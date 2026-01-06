import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as T
import math
import numpy as np

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class Block(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, up=False):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)
        if up:
            self.conv1 = nn.Conv2d(2 * in_ch, out_ch, 3, padding=1)
            self.transform = nn.ConvTranspose2d(out_ch, out_ch, 4, 2, 1)
        else:
            self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
            self.transform = nn.Conv2d(out_ch, out_ch, 4, 2, 1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bnorm1 = nn.BatchNorm2d(out_ch)
        self.bnorm2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU()

    def forward(self, x, t):
        h = self.bnorm1(self.relu(self.conv1(x)))
        time_emb = self.relu(self.time_mlp(t))
        time_emb = time_emb[(..., ) + (None, ) * 2]
        h = h + time_emb
        h = self.bnorm2(self.relu(self.conv2(h)))
        return self.transform(h)

class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, time_emb_dim=None, groups=8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if time_emb_dim is not None else None
        
        self.block1 = nn.Sequential(
            nn.Conv2d(dim, dim_out, 3, padding=1),
            nn.GroupNorm(groups, dim_out),
            nn.SiLU(),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(dim_out, dim_out, 3, padding=1),
            nn.GroupNorm(groups, dim_out),
            nn.SiLU(),
        )
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        scale_shift = None
        if self.mlp is not None and time_emb is not None:
            time_emb = self.mlp(time_emb)
            time_emb = time_emb.view(time_emb.shape[0], -1, 1, 1)
            scale_shift = time_emb.chunk(2, dim=1)

        h = self.block1(x)
        if scale_shift is not None:
            scale, shift = scale_shift
            h = h * (scale + 1) + shift
        
        h = self.block2(h)
        return h + self.res_conv(x)

class Attention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: t.view(b, self.heads, -1, h * w).permute(0, 1, 3, 2), qkv)
        q = q * self.scale
        sim = torch.einsum("b h i d, b h j d -> b h i j", q, k)
        attn = sim.softmax(dim=-1)
        out = torch.einsum("b h i j, b h j d -> b h i d", attn, v)
        out = out.permute(0, 1, 3, 2).reshape(b, -1, h, w)
        return self.to_out(out)

class SimpleUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, base_channels=64, channel_mults=(1, 1, 2, 3, 4)):
        super().__init__()
        self.init_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(base_channels),
            nn.Linear(base_channels, base_channels * 4),
            nn.GELU(),
            nn.Linear(base_channels * 4, base_channels * 4),
        )
        
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        
        channels = [base_channels]
        curr_channel = base_channels
        
        for i, mult in enumerate(channel_mults):
            out_channel = base_channels * mult
            is_attn = out_channel in [256, 512] 
            self.downs.append(nn.ModuleList([
                ResnetBlock(curr_channel, out_channel, time_emb_dim=base_channels * 4),
                ResnetBlock(out_channel, out_channel, time_emb_dim=base_channels * 4),
                Attention(out_channel) if is_attn else nn.Identity(),
                nn.Conv2d(out_channel, out_channel, 4, 2, 1) if i < len(channel_mults) - 1 else nn.Identity()
            ]))
            channels.append(out_channel)
            curr_channel = out_channel
            
        self.mid_block1 = ResnetBlock(curr_channel, curr_channel, time_emb_dim=base_channels * 4)
        self.mid_attn = Attention(curr_channel)
        self.mid_block2 = ResnetBlock(curr_channel, curr_channel, time_emb_dim=base_channels * 4)
        
        for i, mult in reversed(list(enumerate(channel_mults))):
            out_channel = base_channels * mult
            is_attn = out_channel in [256, 512]
            skip_channel = channels.pop()
            self.ups.append(nn.ModuleList([
                ResnetBlock(curr_channel + skip_channel, out_channel, time_emb_dim=base_channels * 4),
                ResnetBlock(out_channel, out_channel, time_emb_dim=base_channels * 4),
                Attention(out_channel) if is_attn else nn.Identity(),
                nn.ConvTranspose2d(out_channel, out_channel, 4, 2, 1) if i > 0 else nn.Identity()
            ]))
            curr_channel = out_channel
            
        self.final_conv = nn.Conv2d(curr_channel, out_channels, 1)

    def forward(self, x, time):
        t = self.time_mlp(time)
        x = self.init_conv(x)
        skips = [x]
        
        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            x = block2(x, t)
            x = attn(x)
            skips.append(x)
            x = downsample(x)
            
        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)
        
        skips.pop() 
        
        for block1, block2, attn, upsample in self.ups:
            skip = skips.pop()
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
            x = torch.cat([x, skip], dim=1)
            x = block1(x, t)
            x = block2(x, t)
            x = attn(x)
            x = upsample(x)
            
        return self.final_conv(x)

class DiffusionModule(nn.Module):
    def __init__(self, timesteps=1000, beta_start=1e-4, beta_end=0.02, pmc_t=200):
        super().__init__()
        self.timesteps = timesteps
        self.pmc_t = pmc_t
        
        self.betas = torch.linspace(beta_start, beta_end, timesteps)
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        
        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            self._extract(self.sqrt_recip_alphas, t, x_t.shape) * x_t -
            self._extract(self.sqrt_recip_alphas, t, x_t.shape) * self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * noise
        )

    def p_sample(self, model, x, t, t_index):
        betas_t = self._extract(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
        sqrt_recip_alphas_t = self._extract(self.sqrt_recip_alphas, t, x.shape)
        
        model_mean = sqrt_recip_alphas_t * (x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t)
        
        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = self._extract(self.posterior_variance, t, x.shape)
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(posterior_variance_t) * noise

    def p_sample_fast(self, model, x, t):
        noise_pred = model(x, t)
        alpha_t = self._extract(self.alphas_cumprod, t, x.shape)
        beta_t = self._extract(self.betas, t, x.shape)
        
        pred_x0 = (x - torch.sqrt(1 - alpha_t) * noise_pred) / torch.sqrt(alpha_t)
        return pred_x0

    def _extract(self, a, t, x_shape):
        batch_size = t.shape[0]
        out = a.to(t.device).gather(0, t)
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        modules = []
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ))
        
        rates = [6, 12, 18]
        for rate in rates:
            modules.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=rate, dilation=rate, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            ))
            
        modules.append(nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ))
        
        self.convs = nn.ModuleList(modules)
        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

    def forward(self, x):
        res = []
        for conv in self.convs:
            out = conv(x)
            if out.shape[2:] != x.shape[2:]:
                out = F.interpolate(out, size=x.shape[2:], mode='bilinear', align_corners=False)
            res.append(out)
        res = torch.cat(res, dim=1)
        return self.project(res)

class AnomalyDetector(nn.Module):
    def __init__(self, in_channels=1024 + 512):
        super().__init__()
        self.aspp = ASPP(in_channels, 256)
        self.res_block = ResnetBlock(256, 256)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.aspp(x)
        x = self.res_block(x)
        x = self.deconv(x)
        return x

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()

class MTDiff(nn.Module):
    def __init__(self, device='cuda'):
        super().__init__()
        self.device = device
        
        self.unet_p1 = SimpleUNet(channel_mults=(1, 1, 2, 3, 4)).to(device)
        self.unet_p2 = SimpleUNet(channel_mults=(1, 2, 2, 4)).to(device)
        self.unet_p3 = SimpleUNet(channel_mults=(1, 2, 4)).to(device)
        
        self.diffusion = DiffusionModule(pmc_t=200).to(device)
        
        resnet = models.resnet50(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:6]).to(device).eval()
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
            
        self.detector = AnomalyDetector().to(device)
        self.focal_loss = FocalLoss()

    def get_gaussian_downsample(self, x, scale_idx):
        if scale_idx == 0:
            return x
        elif scale_idx == 1:
            x_blur = T.GaussianBlur(3, sigma=(0.1, 2.0))(x)
            return F.interpolate(x_blur, scale_factor=0.5, mode='bilinear', align_corners=False)
        elif scale_idx == 2:
            x_blur = T.GaussianBlur(3, sigma=(0.1, 2.0))(x)
            x_down = F.interpolate(x_blur, scale_factor=0.25, mode='bilinear', align_corners=False)
            return x_down

    def forward_train_diffusion(self, x):
        xp1 = self.get_gaussian_downsample(x, 0)
        xp2 = self.get_gaussian_downsample(x, 1)
        xp3 = self.get_gaussian_downsample(x, 2)
        
        t = torch.randint(1, self.diffusion.pmc_t + 1, (x.shape[0],), device=self.device).long()
        
        noise1 = torch.randn_like(xp1)
        noise2 = torch.randn_like(xp2)
        noise3 = torch.randn_like(xp3)
        
        x_t1 = self.diffusion.q_sample(xp1, t, noise1)
        x_t2 = self.diffusion.q_sample(xp2, t, noise2)
        x_t3 = self.diffusion.q_sample(xp3, t, noise3)
        
        pred_noise1 = self.unet_p1(x_t1, t)
        pred_noise2 = self.unet_p2(x_t2, t)
        pred_noise3 = self.unet_p3(x_t3, t)
        
        loss = F.mse_loss(pred_noise1, noise1) + \
               F.mse_loss(pred_noise2, noise2) + \
               F.mse_loss(pred_noise3, noise3)
               
        return loss

    @torch.no_grad()
    def reconstruct_pmc(self, x, scale_idx):
        if scale_idx == 0:
            model = self.unet_p1
        elif scale_idx == 1:
            model = self.unet_p2
        else:
            model = self.unet_p3
            
        x_scale = self.get_gaussian_downsample(x, scale_idx)
        
        t_start = torch.full((x.shape[0],), self.diffusion.pmc_t, device=self.device).long()
        noise = torch.randn_like(x_scale)
        x_t = self.diffusion.q_sample(x_scale, t_start, noise)
        
        for t in reversed(range(0, self.diffusion.pmc_t)):
            t_batch = torch.full((x.shape[0],), t, device=self.device).long()
            x_t = self.diffusion.p_sample(model, x_t, t_batch, t)
            
        return x_t 

    def extract_features(self, x):
        features = []
        h = x
        for name, module in self.feature_extractor.named_children():
            h = module(h)
            if name == 'layer2' or name == 'layer3':
                features.append(h)
        return features 

    def compute_cosine_distance(self, feat1, feat2):
        assert feat1.shape == feat2.shape
        cos_sim = F.cosine_similarity(feat1, feat2, dim=1, eps=1e-6)
        return 1 - cos_sim

    def get_aggregated_heatmap(self, x):
        rec_p1 = self.reconstruct_pmc(x, 0)
        rec_p2 = self.reconstruct_pmc(x, 1)
        rec_p3 = self.reconstruct_pmc(x, 2)
        
        feat_in_p1 = self.extract_features(x)
        feat_rec_p1 = self.extract_features(rec_p1)
        
        feat_in_p2 = self.extract_features(self.get_gaussian_downsample(x, 1))
        feat_rec_p2 = self.extract_features(rec_p2)
        
        feat_in_p3 = self.extract_features(self.get_gaussian_downsample(x, 2))
        feat_rec_p3 = self.extract_features(rec_p3)
        
        diff_maps = []
        
        for i, (fin, frec) in enumerate(zip(feat_in_p1, feat_rec_p1)):
             d = self.compute_cosine_distance(fin, frec).unsqueeze(1)
             diff_maps.append(F.interpolate(d, size=(256, 256), mode='bilinear', align_corners=False))
             
        for i, (fin, frec) in enumerate(zip(feat_in_p2, feat_rec_p2)):
             d = self.compute_cosine_distance(fin, frec).unsqueeze(1)
             diff_maps.append(F.interpolate(d, size=(256, 256), mode='bilinear', align_corners=False))

        for i, (fin, frec) in enumerate(zip(feat_in_p3, feat_rec_p3)):
             d = self.compute_cosine_distance(fin, frec).unsqueeze(1)
             diff_maps.append(F.interpolate(d, size=(256, 256), mode='bilinear', align_corners=False))

        F_agg = torch.cat(diff_maps, dim=1)
        return F_agg

    def forward_train_detector(self, x_normal, x_anomaly_synthetic, mask_gt):
        F_agg = self.get_aggregated_heatmap(x_anomaly_synthetic)
        
        pred_mask = self.detector(F_agg)
        
        if pred_mask.shape[2:] != mask_gt.shape[2:]:
            pred_mask = F.interpolate(pred_mask, size=mask_gt.shape[2:], mode='bilinear', align_corners=False)
            
        l1_loss = F.l1_loss(pred_mask, mask_gt)
        f_loss = self.focal_loss(pred_mask, mask_gt)
        
        return l1_loss + f_loss

    def inference(self, x):
        F_agg = self.get_aggregated_heatmap(x)
        pred_mask = self.detector(F_agg)
        return pred_mask