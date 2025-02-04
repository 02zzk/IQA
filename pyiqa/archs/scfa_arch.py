# _*coding:utf-8_*_
# _*coding:utf-8_*_
"""TOP-IQ metric, proposed by
TOPIQ: A Top-down Approach from Semantics to Distortions for Image Quality Assessment.
Chaofeng Chen, Jiadi Mo, Jingwen Hou, Haoning Wu, Liang Liao, Wenxiu Sun, Qiong Yan, Weisi Lin.
Transactions on Image Processing, 2024.
Paper link: https://arxiv.org/abs/2308.03060

"""
from torch import nn

import torch.nn.init as init
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import timm
from .constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, OPENAI_CLIP_MEAN, OPENAI_CLIP_STD
from pyiqa.utils.registry import ARCH_REGISTRY
from pyiqa.utils.download_util import DEFAULT_CACHE_DIR
from pyiqa.archs.arch_util import dist_to_mos, load_pretrained_network, uniform_crop
import time
import copy
from .clip_model import load
from .topiq_swin import create_swin
from facexlib.utils.face_restoration_helper import FaceRestoreHelper
import warnings
import torch
from pyiqa.archs.arch_util import get_url_from_name
import sys
import os
# sys.path.append('/data1/zengzk/IQA_PyTorch_main')
sys.path.append('/data1/zengzk/IQA-PyTorch-main/IQA-PyTorch-main')
from SAM.segment_anything import sam_model_registry
from svt.svt import DTCWTForward, DTCWTInverse

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class SVT_channel_mixing(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # 将模型和所有参数移到指定的设备
        self.device = device  # 确保使用的设备是 'cuda' 或 'cpu'
        self.to(device)  # 确保模型的所有部分都在同一设备上
        # 将模型和所有参数移到指定的设备
        if dim == 64:  # [b, 64,56,56]
            self.hidden_size = dim
            self.num_blocks = 4
            self.block_size = self.hidden_size // self.num_blocks
            assert self.hidden_size % self.num_blocks == 0
            self.complex_weight_ll = nn.Parameter(torch.randn(dim, 56, 56, dtype=torch.float32) * 0.02)
            self.complex_weight_lh_1 = nn.Parameter(
                torch.randn(2, self.num_blocks, self.block_size, self.block_size, dtype=torch.float32) * 0.02)
            self.complex_weight_lh_2 = nn.Parameter(
                torch.randn(2, self.num_blocks, self.block_size, self.block_size, dtype=torch.float32) * 0.02)
            self.complex_weight_lh_b1 = nn.Parameter(
                torch.randn(2, self.num_blocks, self.block_size, dtype=torch.float32) * 0.02)
            self.complex_weight_lh_b2 = nn.Parameter(
                torch.randn(2, self.num_blocks, self.block_size, dtype=torch.float32) * 0.02)

        if dim == 128:  # [b, 128,28,28]
            self.hidden_size = dim
            self.num_blocks = 4
            self.block_size = self.hidden_size // self.num_blocks
            assert self.hidden_size % self.num_blocks == 0
            self.complex_weight_ll = nn.Parameter(torch.randn(dim, 28, 28, dtype=torch.float32) * 0.02)
            self.complex_weight_lh_1 = nn.Parameter(
                torch.randn(2, self.num_blocks, self.block_size, self.block_size, dtype=torch.float32) * 0.02)
            self.complex_weight_lh_2 = nn.Parameter(
                torch.randn(2, self.num_blocks, self.block_size, self.block_size, dtype=torch.float32) * 0.02)
            self.complex_weight_lh_b1 = nn.Parameter(
                torch.randn(2, self.num_blocks, self.block_size, dtype=torch.float32) * 0.02)
            self.complex_weight_lh_b2 = nn.Parameter(
                torch.randn(2, self.num_blocks, self.block_size, dtype=torch.float32) * 0.02)

        if dim == 96:  # 96 for large model, 64 for small and base model
            self.hidden_size = dim
            self.num_blocks = 4
            self.block_size = self.hidden_size // self.num_blocks
            assert self.hidden_size % self.num_blocks == 0
            self.complex_weight_ll = nn.Parameter(torch.randn(dim, 56, 56, dtype=torch.float32) * 0.02)
            self.complex_weight_lh_1 = nn.Parameter(
                torch.randn(2, self.num_blocks, self.block_size, self.block_size, dtype=torch.float32) * 0.02)
            self.complex_weight_lh_2 = nn.Parameter(
                torch.randn(2, self.num_blocks, self.block_size, self.block_size, dtype=torch.float32) * 0.02)
            self.complex_weight_lh_b1 = nn.Parameter(
                torch.randn(2, self.num_blocks, self.block_size, dtype=torch.float32) * 0.02)
            self.complex_weight_lh_b2 = nn.Parameter(
                torch.randn(2, self.num_blocks, self.block_size, dtype=torch.float32) * 0.02)
        if dim == 192:
            self.hidden_size = dim
            self.num_blocks = 4
            self.block_size = self.hidden_size // self.num_blocks
            assert self.hidden_size % self.num_blocks == 0
            self.complex_weight_ll = nn.Parameter(torch.randn(dim, 28, 28, dtype=torch.float32) * 0.02)
            self.complex_weight_lh_1 = nn.Parameter(
                torch.randn(2, self.num_blocks, self.block_size, self.block_size, dtype=torch.float32) * 0.02)
            self.complex_weight_lh_2 = nn.Parameter(
                torch.randn(2, self.num_blocks, self.block_size, self.block_size, dtype=torch.float32) * 0.02)
            self.complex_weight_lh_b1 = nn.Parameter(
                torch.randn(2, self.num_blocks, self.block_size, dtype=torch.float32) * 0.02)
            self.complex_weight_lh_b2 = nn.Parameter(
                torch.randn(2, self.num_blocks, self.block_size, dtype=torch.float32) * 0.02)

        self.xfm = DTCWTForward(J=1, biort='near_sym_b', qshift='qshift_b').to(self.device)
        self.ifm = DTCWTInverse(biort='near_sym_b', qshift='qshift_b').to(self.device)
        self.softshrink = 0.0

        if dim > 192:
            self.hidden_size = dim
            self.num_blocks = 4
            self.block_size = self.hidden_size // self.num_blocks
            assert self.hidden_size % self.num_blocks == 0
            self.complex_weight_ll = nn.Parameter(torch.randn(dim, 28, 28, dtype=torch.float32) * 0.02)
            self.complex_weight_lh_1 = nn.Parameter(
            torch.randn(2, self.num_blocks, self.block_size, self.block_size, dtype=torch.float32) * 0.02)
            self.complex_weight_lh_2 = nn.Parameter(
            torch.randn(2, self.num_blocks, self.block_size, self.block_size, dtype=torch.float32) * 0.02)
            self.complex_weight_lh_b1 = nn.Parameter(
            torch.randn(2, self.num_blocks, self.block_size, dtype=torch.float32) * 0.02)
            self.complex_weight_lh_b2 = nn.Parameter(
            torch.randn(2, self.num_blocks, self.block_size, dtype=torch.float32) * 0.02)

    def multiply(self, input, weights):
        device = self.device  # 获取 input 张量所在的设备
        input = input.to(device)
        weights = weights.to(device)  # 将 weights 移动到与 input 相同的设备
        return torch.einsum('...bd,bdk->...bk', input, weights)

    def forward(self, x):   # h w刪除

        # 获取输入张量 x 所在的设备
        device = x.device
        # 确保输入张量 x 在正确的设备上
        x = x.to(device)
        B,C,H,W=x.shape
        # 处理输入特征图的形状
        x = x.to(torch.float32)

        # 开始计时
        # start_time = time.time()
        xl, xh = self.xfm(x)
        # 结束计时
        # end_time = time.time()
        # 计算耗时
        # elapsed_time = end_time - start_time
        # print(f"小波变换耗时: {elapsed_time:.6f} 秒")

        xh[0] = torch.permute(xh[0], (5, 0, 2, 3, 4, 1))
        xh[0] = xh[0].reshape(xh[0].shape[0], xh[0].shape[1], xh[0].shape[2], xh[0].shape[3], xh[0].shape[4],
                              self.num_blocks, self.block_size)

        x_real = xh[0][0]
        x_imag = xh[0][1]

        x_real_1 = F.relu(
            self.multiply(x_real, self.complex_weight_lh_1[0]) - self.multiply(x_imag, self.complex_weight_lh_1[1]) +
            self.complex_weight_lh_b1[0])
        x_imag_1 = F.relu(
            self.multiply(x_real, self.complex_weight_lh_1[1]) + self.multiply(x_imag, self.complex_weight_lh_1[0]) +
            self.complex_weight_lh_b1[1])

        x_real_2 = self.multiply(x_real_1, self.complex_weight_lh_2[0]) - self.multiply(x_imag_1,
                                                                                        self.complex_weight_lh_2[1]) + \
                   self.complex_weight_lh_b2[0]
        x_imag_2 = self.multiply(x_real_1, self.complex_weight_lh_2[1]) + self.multiply(x_imag_1,
                                                                                        self.complex_weight_lh_2[0]) + \
                   self.complex_weight_lh_b2[1]

        xh[0] = torch.stack([x_real_2, x_imag_2], dim=-1).float()
        xh[0] = F.softshrink(xh[0], lambd=self.softshrink) if self.softshrink else xh[0]
        xh[0] = xh[0].reshape(B, xh[0].shape[1], xh[0].shape[2], xh[0].shape[3], self.hidden_size, xh[0].shape[6])
        xh[0] = torch.permute(xh[0], (0, 4, 1, 2, 3, 5))


        x = self.ifm((xl, xh))

        return x

model_type = "vit_b"
checkpoint_path = "/data1/zengzk/wafer_IQA/wafer_2024/segment_anything/checkpoints/sam_vit_b_01ec64.pth"
# 输入文件夹
input_folder = "/data1/zengzk/wafer_IQA/wafer_2024/datasets/wafer/1024*1024"  # 输入文件夹路径，包含所有图像

class SAMFeatureExtractor(nn.Module):
    def __init__(self, model_type="vit_b", checkpoint_path=None):
        super(SAMFeatureExtractor, self).__init__()


        # 通过预训练模型路径加载 SAM 模型，并将其加载到指定设备
        self.sam_model = sam_model_registry[model_type](checkpoint=checkpoint_path).to(device)  # 确保模型加载到正确设备
        self.image_encoder = self.sam_model.image_encoder  # 提取图像编码器部分

        # 冻结SAM模型的所有参数
        for param in self.sam_model.parameters():
            param.requires_grad = False  # 冻结权重，防止训练过程中更新这些参数

    def forward(self, x):
        # 确保输入数据也在正确的设备上
        x = x.to(device)  # 确保输入张量在正确的设备上
        features = self.image_encoder(x)  # 提取特征
        return features

default_model_urls = {

    'SAMNet_nr_koniq_res50': get_url_from_name('cfanet_nr_koniq_res50-9a73138b.pth'),
}

sam_feature_extractor = SAMFeatureExtractor(model_type="vit_b", checkpoint_path=checkpoint_path)

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])  # 多次使用相同的模块

def _get_activation_fn(activation):  # 根据提供的字符串参数返回对应的激活函数。
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="gelu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def forward(self, src):
        src2 = self.norm1(src)
        q = k = src2
        src2, self.attn_map = self.self_attn(q, k, value=src2)
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="gelu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def forward(self, tgt, memory):
        memory = self.norm2(memory)
        tgt2 = self.norm1(tgt)
        tgt2, self.attn_map = self.multihead_attn(query=tgt2,
                                                  key=memory, value=memory)
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    def forward(self, src):
        output = src

        for layer in self.layers:
            output = layer(output)

        return output

class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers

    def forward(self, tgt, memory):
        output = tgt

        for layer in self.layers:
            output = layer(output, memory)

        return output

class GatedConv(nn.Module):
    def __init__(self, weightdim, ksz=3):
        super().__init__()

        self.splitconv = nn.Conv2d(weightdim, weightdim * 2, 1, 1, 0)
        self.act = nn.GELU()

        self.weight_blk = nn.Sequential(
            nn.Conv2d(weightdim, 64, 1, stride=1),
            nn.GELU(),
            nn.Conv2d(64, 64, ksz, stride=1, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 1, ksz, stride=1, padding=1),
            nn.Sigmoid()
        )
    def forward(self, x):
        x1, x2 = self.splitconv(x).chunk(2, dim=1)
        weight = self.weight_blk(x2)
        x1 = self.act(x1)
        return x1 * weight

class SVTFusionModule(nn.Module):
    def __init__(self,in_ch, out_ch,kernel_size=1):
        super(SVTFusionModule, self).__init__()  # 确保父类初始化
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size)
    def forward(self,x):
        return self.conv(x)

@ARCH_REGISTRY.register()
class SVTNet(nn.Module):
    def __init__(self,
                 semantic_model_name='resnet50',
                 model_name='clip_nr_koniq_res50',
                 backbone_pretrain=True,
                 in_size=None,
                 use_ref=False,
                 num_class=1,
                 num_crop=1,
                 crop_size=256,
                 inter_dim=256,
                 num_heads=4,
                 num_attn_layers=1,
                 dprate=0.1,
                 activation='gelu',
                 pretrained=True,
                 pretrained_model_path=None,
                 out_act=False,
                 block_pool='weighted_avg',
                 test_img_size=None,
                 align_crop_face=True,
                 default_mean=IMAGENET_DEFAULT_MEAN,  #後續考慮修改
                 default_std=IMAGENET_DEFAULT_STD,
                 fusion_method='concat',
                 brightness=torch.tensor(1.0, dtype=torch.float32),
                 angle = torch.tensor(0.0, dtype=torch.float32),

    ):
        super().__init__()

        self.in_size = in_size

        self.model_name = model_name
        self.semantic_model_name = semantic_model_name
        self.semantic_level = -1
        self.crop_size = crop_size
        self.use_ref = use_ref

        self.num_class = num_class
        self.block_pool = block_pool
        self.test_img_size = test_img_size

        self.align_crop_face = align_crop_face
        self.brightness =brightness
        self.angle = angle
        # self.sam_feature_extractor = SAMFeatureExtractor().to(device)

        # =============================================================
        # define semantic backbone network
        # =============================================================

        self.svt_modules = nn.ModuleList()  # 用来存储所有的 SVT_channel_mixing 模块
        target_channels = [64, 256, 512, 1024, 2048]  # 5个阶段的 ResNet
        in_channels = [128, 512, 1024, 2048, 4096]
        self.target_channels = target_channels  # 目标输出通道数列表
        self.in_channels = in_channels  # 输入通道数列表

        # 初始化卷积层的 ModuleList（包含多个卷积层）
        self.convs = nn.ModuleList()

        # 假设我们有5个阶段（5个卷积层），每个阶段的目标通道数和输入通道数来自于相应的列表
        for i in range(len(self.target_channels)):
            in_ch = self.in_channels[i]  # 获取每个阶段的输入通道数
            out_ch = self.target_channels[i]  # 获取每个阶段的目标输出通道数
            conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)
            self.convs.append(conv)

        # 对所有卷积层进行初始化
        for conv in self.convs:
            init.kaiming_normal_(conv.weight, mode='fan_out', nonlinearity='relu')
            if conv.bias is not None:
                init.zeros_(conv.bias)
        # 在初始化时就实例化所有的 SVT_channel_mixing 模块
        for dim in target_channels:
             svt_module = SVT_channel_mixing(dim=dim)
             self.svt_modules.append(svt_module)

        if 'swin' in semantic_model_name:
            self.semantic_model = create_swin(semantic_model_name, pretrained=True, drop_path_rate=0.0)
            feature_dim = self.semantic_model.num_features
            feature_dim_list = [int(self.semantic_model.embed_dim * 2 ** i) for i in
                                range(self.semantic_model.num_layers)]
            feature_dim_list = feature_dim_list[1:] + [feature_dim]
            all_feature_dim = sum(feature_dim_list)
        elif 'clip' in semantic_model_name:
            semantic_model_name = semantic_model_name.replace('clip_', '')
            self.semantic_model = [load(semantic_model_name, 'cpu')]
            feature_dim_list = self.semantic_model[0].visual.feature_dim_list
            default_mean, default_std = OPENAI_CLIP_MEAN, OPENAI_CLIP_STD
        else:
            self.semantic_model = timm.create_model(semantic_model_name, pretrained=True, features_only=True,
                                                    pretrained_cfg_overlay=dict(
                                                        file='/data1/zengzk/wafer_IQA/wafer_2024/model/resnet50.bin'))
            feature_dim_list = self.semantic_model.feature_info.channels()
            feature_dim = feature_dim_list[self.semantic_level]
            all_feature_dim = sum(feature_dim_list)
            self.fix_bn(self.semantic_model)

        self.default_mean = torch.Tensor(default_mean).view(1, 3, 1, 1)
        self.default_std = torch.Tensor(default_std).view(1, 3, 1, 1)
        self.fusion_mul = 3 if use_ref else 1
        ca_layers = sa_layers = num_attn_layers
        # 定义融合方法：拼接特征还是加法融合
        self.fusion_method = fusion_method
        self.act_layer = nn.GELU() if activation == 'gelu' else nn.ReLU()
        dim_feedforward = min(4 * inter_dim, 2048)

        # gated local pooling and self-attention
        tmp_layer = TransformerEncoderLayer(inter_dim, nhead=num_heads, dim_feedforward=dim_feedforward,    #实例化
                                            normalize_before=True, dropout=dprate, activation=activation)
        self.sa_attn_blks = nn.ModuleList()
        self.dim_reduce = nn.ModuleList()
        self.weight_pool = nn.ModuleList()
        for idx, dim in enumerate(feature_dim_list):
            dim = dim * 3 if use_ref else dim
            if use_ref:
                self.weight_pool.append(
                    nn.Sequential(
                        nn.Conv2d(dim // 3, 64, 1, stride=1),
                        self.act_layer,
                        nn.Conv2d(64, 64, 3, stride=1, padding=1),
                        self.act_layer,
                        nn.Conv2d(64, 1, 3, stride=1, padding=1),
                        nn.Sigmoid()
                    )
                )
            else:
                self.weight_pool.append(GatedConv(dim))

            self.dim_reduce.append(nn.Sequential(
                nn.Conv2d(dim, inter_dim, 1, 1),
                self.act_layer,
            )
            )

            self.sa_attn_blks.append(TransformerEncoder(tmp_layer, sa_layers))   #實例化這個類  src輸入數據並沒有消失

        # cross scale attention
        self.attn_blks = nn.ModuleList()
        tmp_layer = TransformerDecoderLayer(inter_dim, nhead=num_heads, dim_feedforward=dim_feedforward,
                                            normalize_before=True, dropout=dprate, activation=activation)
        for i in range(len(feature_dim_list) - 1):
            self.attn_blks.append(TransformerDecoder(tmp_layer, ca_layers))

        # attention pooling and MLP layers
        self.attn_pool = TransformerEncoderLayer(inter_dim, nhead=num_heads, dim_feedforward=dim_feedforward,
                                                 normalize_before=True, dropout=dprate, activation=activation)

        linear_dim = inter_dim
        self.score_linear = [
            nn.LayerNorm(linear_dim),
            nn.Linear(linear_dim, linear_dim),
            self.act_layer,
            nn.LayerNorm(linear_dim),
            nn.Linear(linear_dim, linear_dim),
            self.act_layer,
            nn.Linear(linear_dim, self.num_class),
        ]

        # make sure output is positive, useful for 2AFC datasets with probability labels
        if out_act and self.num_class == 1:
            self.score_linear.append(nn.Softplus())

        if self.num_class > 1:
            self.score_linear.append(nn.Softmax(dim=-1))

        self.score_linear = nn.Sequential(*self.score_linear)

        self.h_emb = nn.Parameter(torch.randn(1, inter_dim // 2, 32, 1))
        self.w_emb = nn.Parameter(torch.randn(1, inter_dim // 2, 1, 32))

        nn.init.trunc_normal_(self.h_emb.data, std=0.02)
        nn.init.trunc_normal_(self.w_emb.data, std=0.02)
        self._init_linear(self.dim_reduce)
        self._init_linear(self.sa_attn_blks)
        self._init_linear(self.attn_blks)
        self._init_linear(self.attn_pool)

        if pretrained_model_path is not None:
            load_pretrained_network(self, pretrained_model_path, False, weight_keys='params')
        # elif pretrained:
        #     load_pretrained_network(self, default_model_urls[model_name], True, weight_keys='params')

        self.eps = 1e-8
        self.crops = num_crop

        if 'gfiqa' in model_name:
            self.face_helper = FaceRestoreHelper(
                1,
                face_size=512,
                crop_ratio=(1, 1),
                det_model='retinaface_resnet50',
                save_ext='png',
                use_parse=True,
                model_rootpath=DEFAULT_CACHE_DIR,
            )

    def _init_linear(self, m):
        for module in m.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight.data)
                nn.init.constant_(module.bias.data, 0)

    def preprocess(self, x):
        x = (x - self.default_mean.to(x)) / self.default_std.to(x)
        return x

    def fix_bn(self, model):
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                for p in m.parameters():
                    p.requires_grad = False
                m.eval()

    def get_swin_feature(self, model, x):
        b, c, h, w = x.shape
        x = model.patch_embed(x)
        if model.absolute_pos_embed is not None:
            x = x + model.absolute_pos_embed
        x = model.pos_drop(x)
        feat_list = []
        for ly in model.layers:
            x = ly(x)
            feat_list.append(x)

        h, w = h // 8, w // 8
        for idx, f in enumerate(feat_list):
            feat_list[idx] = f.transpose(1, 2).reshape(b, f.shape[-1], h, w)
            if idx < len(feat_list) - 2:
                h, w = h // 2, w // 2

        return feat_list



    def dist_func(self, x, y, eps=1e-12):
        return torch.sqrt((x - y) ** 2 + eps)

    def forward_cross_attention(self, x,y=None,brightness=None, angle=None):
        if not self.training:
            if self.model_name == 'cfanet_iaa_ava_swin':
                x = TF.resize(x, [384, 384], antialias=True)  # swin require square inputs
            elif self.test_img_size is not None:
                x = TF.resize(x, self.test_img_size, antialias=True)

        x = self.preprocess(x)
        if self.use_ref:
            y = self.preprocess(y)
        if 'swin' in self.semantic_model_name:
            dist_feat_list = self.get_swin_feature(self.semantic_model, x)
            if self.use_ref:
                ref_feat_list = self.get_swin_feature(self.semantic_model, y)
            self.semantic_model.eval()
        elif 'clip' in self.semantic_model_name:
            visual_model = self.semantic_model[0].visual.to(x.device)
            dist_feat_list = visual_model.forward_features(x)
            if self.use_ref:
                ref_feat_list = visual_model.forward_features(y)
        else:
            dist_feat_list = self.semantic_model(x)
            if self.use_ref:
                ref_feat_list = self.semantic_model(y)
            self.fix_bn(self.semantic_model)
            self.semantic_model.eval()
        target_channels = [64, 256, 512, 1024, 2048]  # 这里假设是5个阶段的 ResNet

        # for i, feat in enumerate(dist_feat_list):
        #     print('dist_feat_list', feat.shape)
        for i, (feat, svt_module) in enumerate(zip(dist_feat_list, self.svt_modules)):
            # 获取当前尺度的通道数
            #dim = feat.shape[1]  # 获取通道数（C）
            # # 实例化 SVT_channel_mixing 模块
            # svt_module = SVT_channel_mixing(dim=dim).to(device)
            feat = feat.to(device)  # 确保 feat 在同一设备上
            # 将每个尺度特征传入对应的 SVT_module 进行处理
            #dist_feat_list[i] = svt_module(feat)
            feat_svt = svt_module(feat)
            _, _, height, width = feat_svt.shape
            feat_combined =torch.cat([feat_svt,feat],dim=1)
            conv = self.convs[i]
            conv = conv.to(device)  # 将卷积层移到 GPU 上
            adjusted_feature = conv(feat_combined)  # 调整后的特征形状为 [B, out_channels, H, W]
            if torch.var(feat_combined) == 0:
                print("feat_combined 是常数张量。")
            # 将叠加后的特征存回 dist_feat_list
            dist_feat_list[i] = adjusted_feature

        start_level = 0
        end_level = len(dist_feat_list)

        b, c, th, tw = dist_feat_list[end_level - 1].shape
        pos_emb = torch.cat(
            (self.h_emb.repeat(1, 1, 1, self.w_emb.shape[3]), self.w_emb.repeat(1, 1, self.h_emb.shape[2], 1)), dim=1)

        token_feat_list = []
        for i in reversed(range(start_level, end_level)):
            tmp_dist_feat = dist_feat_list[i]
            if hasattr(self, 'sam_block'):               #这两行不考虑
               tmp_dist_feat = self.sam_block(tmp_dist_feat)

            # gated local pooling
            if self.use_ref:
                tmp_ref_feat = ref_feat_list[i]
                diff = self.dist_func(tmp_dist_feat, tmp_ref_feat)

                tmp_feat = torch.cat([tmp_dist_feat, tmp_ref_feat, diff], dim=1)
                weight = self.weight_pool[i](diff)
                tmp_feat = tmp_feat * weight
            else:
                tmp_feat = self.weight_pool[i](tmp_dist_feat)

            if tmp_feat.shape[2] > th and tmp_feat.shape[3] > tw:
                tmp_feat = F.adaptive_avg_pool2d(tmp_feat, (th, tw))

            # self attention
            tmp_pos_emb = F.interpolate(pos_emb, size=tmp_feat.shape[2:], mode='bicubic', align_corners=False)
            tmp_pos_emb = tmp_pos_emb.flatten(2).permute(2, 0, 1)

            tmp_feat = self.dim_reduce[i](tmp_feat)
            tmp_feat = tmp_feat.flatten(2).permute(2, 0, 1)
            tmp_feat = tmp_feat + tmp_pos_emb

            tmp_feat = self.sa_attn_blks[i](tmp_feat)
            token_feat_list.append(tmp_feat)

        # high level -> low level: coarse to fine
        query = token_feat_list[0]
        query_list = [query]
        for i in range(len(token_feat_list) - 1):
            key_value = token_feat_list[i + 1]
            query = self.attn_blks[i](query, key_value)
            query_list.append(query)

        final_feat = self.attn_pool(query)
        out_score = self.score_linear(final_feat.mean(dim=0))

        return out_score

    def preprocess_face(self, x):
        warnings.warn(
            f'The faces will be aligned, cropped and resized to 512x512 with facexlib. Currently, this metric does not support batch size > 1 and gradient backpropagation.',
            UserWarning)
        # warning message
        device = x.device
        assert x.shape[0] == 1, f'Only support batch size 1, but got {x.shape[0]}'
        self.face_helper.clean_all()
        self.face_helper.input_img = x[0].permute(1, 2, 0).cpu().numpy() * 255
        self.face_helper.input_img = self.face_helper.input_img[..., ::-1]
        if self.face_helper.get_face_landmarks_5(only_center_face=True, eye_dist_threshold=5) > 0:
            self.face_helper.align_warp_face()
            x = self.face_helper.cropped_faces[0]
            x = torch.from_numpy(x[..., ::-1].copy()).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            return x.to(device)
        else:
            assert False, f'No face detected in the input image.'

    def forward(self, x, y=None,brightness=None, angle=None,return_mos=True, return_dist=False):
        if self.use_ref:
            assert y is not None, f'Please input y when use reference is True.'
        else:
            y = None

        if 'gfiqa' in self.model_name:
            if self.align_crop_face:
                x = self.preprocess_face(x)
            else:
                x = nn.functional.interpolate(x, size=(512, 512), mode='bicubic', align_corners=False)

        if self.crops > 1 and not self.training:
            bsz = x.shape[0]
            if y is not None:
                x, y = uniform_crop([x, y], self.crop_size, self.crops)
            else:
                x = uniform_crop([x], self.crop_size, self.crops)[0]

            #score = self.forward_cross_attention(x, y,brightness, angle)
            score = self.forward_cross_attention(x, y)
            score = score.reshape(bsz, self.crops, self.num_class)
            score = score.mean(dim=1)
        else:
            #score = self.forward_cross_attention(x, y,brightness, angle)
            score = self.forward_cross_attention(x, y)

        mos = dist_to_mos(score)
        return_list = []
        if return_mos:
            return_list.append(mos)
        if return_dist:
            return_list.append(score)

        if len(return_list) > 1:
            return return_list
        else:
            return return_list[0]