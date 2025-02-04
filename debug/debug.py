import torch
from pyiqa.archs.zzknet_svt_arch import ZZKNet_svt
model=ZZKNet_svt(crop_size=384,
             pretrained=False,
             num_crop=1,
             num_attn_layers=1,
             semantic_model_name='resnet50',
             block_pool='weighted_avg',
use_ref=False
             ).cuda()
data=torch.zeros((1,3,384,384)).cuda()
res=model(data)
print(res.shape)
