import torch
import torch.nn as nn
'''
https://www.kaggle.com/code/vikramsandu/ddpm-from-scratch-in-pytorch
'''



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DiffusionForwardProcess:
    def __init__(self, num_time_steps, beta_start, beta_end):
        self.betas = torch.linspace(beta_start, beta_end, num_time_steps) # 동일한 간격으로 start-end 범위 숫자 생성
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alpha_bars = torch.sqrt(self.alpha_bars)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1 - self.alpha_bars)


    def add_noise(self, original_x, noise, t):
        # noise 를 원래 data 에 더하는게 forward process
        # t번째
        sqrt_alpha_bar_t = self.sqrt_alpha_bars.to(device)[t]
        sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alpha_bars.to(device)[t]

        sqrt_alpha_bar_t = sqrt_alpha_bar_t[:, None, None, None]
        sqrt_one_minus_alpha_bar_t = sqrt_one_minus_alpha_bar_t[:, None, None, None]

        return (sqrt_alpha_bar_t * original_x) + (sqrt_one_minus_alpha_bar_t * noise)



class DiffusionReverseProcess:
    def __init__(self, num_time_steps, beta_start, beta_end):
        self.betas = torch.linspace(beta_start, beta_end, num_time_steps).to(device) # 동일한 간격으로 start-end 범위 숫자 생성
        self.alphas = (1 - self.betas).to(device)
        self.alpha_bars = torch.cumprod(self.alphas, dim=0).to(device)

    def sample_prev_timestep(self, x_t, noise_pred, t):
        # Given x_t, sample x_(t-1) 이게 x_t1
        # Denoising 하는거

        # Original image Prediction at time step t
        x_t1 = (x_t - (torch.sqrt(1 - self.alpha_bars[t]) * noise_pred)) / torch.sqrt(self.alpha_bars[t])
        x_t1 = torch.clamp(x_t1, -1., 1.)

        # mean of x_(t-1)
        mean = (x_t - ((1 - self.alphas[t]) * noise_pred) / (torch.sqrt(1 - self.alpha_bars[t]))) / torch.sqrt(self.alphas[t])


        if t==0:
            return mean, x_t1
        else:
            variance = ((1 - self.alpha_bars[t-1])/(1 - self.alpha_bars[t])) * self.betas[t]
            sigma = variance ** 0.5
            z = torch.randn(x_t.shape).to(device)
            return mean + sigma*z, x_t1



def get_time_embedding(time_steps: torch.Tensor, t_emb_dim: int) -> torch.Tensor:
    assert t_emb_dim%2 ==0, "time embeddin gmust be divisible by 2."

    factor = 2 * torch.arange(start=0, end=t_emb_dim//2, dtype=torch.float32, device= device) / t_emb_dim
    factor = 10000**factor

    t_emb = time_steps[:, None]  # B -> (B, 1)
    t_emb = t_emb / factor  # (B, 1) -> (B, t_emb_dim//2)
    t_emb = torch.cat([torch.sin(t_emb), torch.cos(t_emb)], dim=1)  # (B , t_emb_dim)
    return t_emb



class NormActConv(nn.Module):
    def __init__(self, in_channels, out_channels, num_groups=1, kernel_size=3, norm=True, act=True):
        super(NormActConv, self).__init__()

        self.g_norm = nn.GroupNorm(num_groups, in_channels) if norm is True else nn.Identity()
        self.act = nn.SiLU() if act is True else nn.Identity()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size-1)//2)

    def forward(self, x):
        x = self.g_norm(x)
        x = self.act(x)
        x = self.conv(x)
        return x



class TimeEmbedding(nn.Module):
    def __init__(self, n_out, t_emb_dim=128):
        super(TimeEmbedding, self).__init__()

        self.te_block = nn.Sequential(
            nn.SiLU(),
            nn.Linear(t_emb_dim, n_out)
        )
    def forward(self,x):
        return self.te_block(x)



class SelfAttentionBlock(nn.Module):
    def __init__(self, num_channels, num_groups=1, num_heads=1, norm=True):
        super(SelfAttentionBlock, self).__init__()

        self.g_norm = nn.GroupNorm(num_groups, num_channels) if norm is True else nn.Identity()
        self.attn = nn.MultiheadAttention(num_channels, num_heads, batch_first=True)

    def forward(self,x):
        batch_size, channels, h, w = x.shape
        x = x.reshape(batch_size, channels, h*w)
        x = self.g_norm(x)
        x = x.transpose(1,2)
        x, _ = self.attn(x, x, x)
        x = x.transpose(1,2).reshape(batch_size, channels, h, w)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels, out_channels, k=2, use_conv=True, use_mpool=True):
        super(Downsample, self).__init__()
        self.use_conv = use_conv
        self.use_mpool = use_mpool

        self.dconv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.Conv2d(in_channels, out_channels//2 if use_mpool else out_channels, kernel_size=4, stride=k, padding=1)
        ) if use_conv else nn.Identity()

        self.mpool = nn.Sequential(
            nn.MaxPool2d(k, k),
            nn.Conv2d(in_channels, out_channels//2 if use_conv else out_channels, kernel_size=1, stride=1, padding=0)
        ) if use_mpool else nn.Identity()

    def forward(self, x):
        # downsampling 을 conv 가지고 할건지 maxpool 로 할건지 아니면 둘다로 할건지
        # 둘다로 하면 outchannel 의 반 만큼씩 줄여서 합쳐.
        if not self.use_conv:
            return self.mpool(x)
        if not self.use_mpool:
            return self.dconv(x)
        return torch.cat([self.dconv(x), self.mpool(x)], dim=1)


class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels, k=2, use_conv=True, use_upsample=True):
        super(Upsample, self).__init__()
        self.use_conv = use_conv
        self.use_upsample = use_upsample

        self.uconv = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels//2 if use_upsample else out_channels, kernel_size=4, stride=k, padding=1),
            nn.Conv2d(out_channels//2 if use_upsample else out_channels, out_channels//2 if use_upsample else out_channels, kernel_size=1, stride=1, padding=0)
        ) if use_conv else nn.Identity()

        self.up = nn.Sequential(
            nn.Upsample(scale_factor=k, mode= 'bilinear', align_corners=False),
            nn.Conv2d(in_channels, out_channels//2 if use_conv else out_channels, kernel_size=1, stride=1, padding=0)
        ) if use_upsample else nn.Identity()

    def forward(self, x):
        # 얘도 conv로 할건지 upsample 로 할건지
        # 둘다로 하면 outchannel의 반 만큼해서 합쳐
        if not self.use_conv:
            return self.up(x)
        if not self.use_upsample:
            return self.uconv(x)
        return torch.cat([self.uconv(x), self.up(x)], dim=1)







class DownC(nn.Module):
    """
    Perform Down-convolution on the input using following approach.
    1. Conv + TimeEmbedding
    2. Conv
    3. Skip-connection from input x.
    4. Self-Attention
    5. Skip-Connection from 3.
    6. Downsampling
    """
    def __init__(self, in_channels, out_channels, t_emb_dim=128, num_layers=2, down_sample=True):
        super(DownC, self).__init__()
        self.num_layers = num_layers
        self.conv1 = nn.ModuleList([
            NormActConv(in_channels if i==0 else out_channels, out_channels) for i in range(num_layers) ])
        self.conv2 = nn.ModuleList([
            NormActConv(out_channels, out_channels) for _ in range(num_layers) ])
        self.te_block = nn.ModuleList([
            TimeEmbedding(out_channels, t_emb_dim) for _ in range(num_layers) ])
        self.attn_block = nn.ModuleList([
            SelfAttentionBlock(out_channels) for _ in range(num_layers) ])
        self.down_block = Downsample(out_channels, out_channels) if down_sample else nn.Identity()
        self.res_block = nn.ModuleList([
            nn.Conv2d(in_channels if i==0 else out_channels, out_channels, kernel_size=1) for i in range(num_layers) ])

    def forward(self, x, t_emb):
        out = x
        for i in range(self.num_layers):
            resnet_input = out

            out = self.conv1[i](out)
            out = out + self.te_block[i](t_emb)[:, :, None, None]
            out = self.conv2[i](out)
            out = out + self.res_block[i](resnet_input)

            # out_attn = self.attn_block[i](out)
            # out = out + out_attn

        out = self.down_block(out)
        return out




class MidC(nn.Module):
    '''
    1. Resnet Block with Time Embedding
    2. A Series of Self-Attention + Resnet Block with Time-Embedding
    '''
    def __init__(self, in_channels, out_channels, t_emb_dim, num_layers):
        super(MidC, self).__init__()
        self.num_layers = num_layers
        self.conv1 = nn.ModuleList([
            NormActConv(in_channels if i==0 else out_channels, out_channels) for i in range(num_layers+1) ])
        self.conv2 = nn.ModuleList([
            NormActConv(out_channels, out_channels) for _ in range(num_layers + 1) ])
        self.te_block = nn.ModuleList([
            TimeEmbedding(out_channels, t_emb_dim) for _ in range(num_layers + 1) ])
        self.attn_block = nn.ModuleList([
            SelfAttentionBlock(out_channels) for _ in range(num_layers) ])
        self.res_block = nn.ModuleList([
            nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, kernel_size=1) for i in range(num_layers + 1) ])


    def forward(self, x, t_emb):
        out = x

        resnet_input = out
        out = self.conv1[0](out)
        out = out + self.te_block[0](t_emb)[:, :, None, None]
        out = self.conv2[0](out)
        out = out + self.res_block[0](resnet_input)

        for i in range(self.num_layers):
            # # Self Attention
            # out_attn = self.attn_block[i](out)
            # out = out + out_attn
            # Resnet Block
            resnet_input = out
            out = self.conv1[i + 1](out)
            out = out + self.te_block[i + 1](t_emb)[:, :, None, None]
            out = self.conv2[i + 1](out)
            out = out + self.res_block[i + 1](resnet_input)

        return out





class UpC(nn.Module):
    """
    Exactly same as DownC Block except Downsampling block is replaced by Upsampling.
    Perform Up-convolution on the input using following approach.
    1. Upsampling
    2. Conv + TimeEmbedding
    3. Conv
    4. Skip-connection from 1.
    5. Self-Attention
    6. Skip-Connection from 3.
    """
    def __init__(self, in_channels, out_channels, t_emb_dim, num_layers, up_sample=True):
        super(UpC, self).__init__()
        self.num_layers = num_layers
        self.conv1 = nn.ModuleList([
            NormActConv(in_channels if i==0 else out_channels, out_channels) for i in range(num_layers) ])
        self.conv2 = nn.ModuleList([
            NormActConv(out_channels, out_channels) for _ in range(num_layers) ])
        self.te_block = nn.ModuleList([
            TimeEmbedding(out_channels, t_emb_dim) for _ in range(num_layers) ])
        self.attn_block = nn.ModuleList([
            SelfAttentionBlock(out_channels) for _ in range(num_layers) ])
        self.up_block =Upsample(in_channels, in_channels//2) if up_sample else nn.Identity()
        self.res_block = nn.ModuleList([
            nn.Conv2d(in_channels if i==0 else out_channels, out_channels, kernel_size=1) for i in range(num_layers) ])

    def forward(self, x, down_out, t_emb):
        # Upsampling
        x = self.up_block(x)
        x = torch.cat([x, down_out], dim=1)

        out = x
        for i in range(self.num_layers):
            resnet_input = out

            # Resnet Block
            out = self.conv1[i](out)
            out = out + self.te_block[i](t_emb)[:, :, None, None]
            out = self.conv2[i](out)
            out = out + self.res_block[i](resnet_input)

            # # Self Attention
            # out_attn = self.attn_block[i](out)
            # out = out + out_attn

        return out




class Unet(nn.Module):
    def __init__(self, img_channels=3, down_ch=[32, 64, 128, 256], mid_ch=[256, 256, 128], up_ch=[256, 128, 64, 16], down_sample = [True, True, False],
                 t_emb_dim=128, num_downc_layers=1, num_midc_layers=1, num_upc_layers=1):
        super(Unet, self).__init__()
        self.im_channels = img_channels
        self.down_ch = down_ch
        self.mid_ch = mid_ch
        self.up_ch = up_ch
        self.t_emb_dim = t_emb_dim
        self.down_sample = down_sample
        self.num_downc_layers = num_downc_layers
        self.num_midc_layers = num_midc_layers
        self.num_upc_layers = num_upc_layers

        self.up_sample = list(reversed(self.down_sample))  # [False, True, True]

        # Initial Convolution
        self.cv1 = nn.Conv2d(self.im_channels, self.down_ch[0], kernel_size=3, padding=1)

        # Initial Time Embedding Projection
        self.t_proj = nn.Sequential(nn.Linear(self.t_emb_dim, self.t_emb_dim), nn.SiLU(), nn.Linear(self.t_emb_dim, self.t_emb_dim))

        # DownC Blocks
        self.downs = nn.ModuleList([
            DownC(self.down_ch[i], self.down_ch[i + 1], self.t_emb_dim, self.num_downc_layers, self.down_sample[i]) for i in range(len(self.down_ch) - 1)])

        # MidC Block
        self.mids = nn.ModuleList([
            MidC(self.mid_ch[i], self.mid_ch[i + 1], self.t_emb_dim, self.num_midc_layers) for i in range(len(self.mid_ch) - 1)])

        # UpC Block
        self.ups = nn.ModuleList([
            UpC(self.up_ch[i], self.up_ch[i + 1], self.t_emb_dim, self.num_upc_layers, self.up_sample[i]) for i in range(len(self.up_ch) - 1)])

        # Final Convolution
        self.cv2 = nn.Sequential(
            nn.GroupNorm(8, self.up_ch[-1]),
            nn.Conv2d(self.up_ch[-1], self.im_channels, kernel_size=3, padding=1))

    def forward(self, x, t):

        out = self.cv1(x)
        # Time Projection
        t_emb = get_time_embedding(t, self.t_emb_dim)
        t_emb = self.t_proj(t_emb)
        # DownC outputs
        down_outs = []
        for down in self.downs:
            down_outs.append(out)
            out = down(out, t_emb)

        # # MidC outputs
        # for mid in self.mids:
        #     out = mid(out, t_emb)

        # UpC Blocks
        for up in self.ups:
            down_out = down_outs.pop()
            out = up(out, down_out, t_emb)

        # Final Conv
        out = self.cv2(out)

        return out



model = Unet().to(device)
x = torch.randn(4, 3, 224, 224).to(device)
t = torch.randint(0, 10, (4,)).to(device)
model(x, t).shape


num_time_steps =1000
beta_start = 1e-4
beta_end = 0.02
num_layers=1 #DownC
t_emb_dim=128
img_channels = 3 #RGB