# import pyiqa
# import torch
#
# # list all available metrics
# print(pyiqa.list_models())
#
# device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
#
# # create metric with default setting
# iqa_metric = pyiqa.create_metric('lpips', device=device)
# # Note that gradient propagation is disabled by default. set as_loss=True to enable it as a loss function.
# iqa_loss = pyiqa.create_metric('lpips', device=device, as_loss=True)
#
# # create metric with custom setting
# iqa_metric = pyiqa.create_metric('psnr', test_y_channel=True, color_space='ycbcr').to(device)
#
# # check if lower better or higher better
# print(iqa_metric.lower_better)
#
# # example for iqa score inference
# # Tensor inputs, img_tensor_x/y: (N, 3, H, W), RGB, 0 ~ 1
# score_fr = iqa_metric(img_tensor_x, img_tensor_y)
# score_nr = iqa_metric(img_tensor_x)
#
# # img path as inputs.
# score_fr = iqa_metric('./ResultsCalibra/dist_dir/I03.bmp', './ResultsCalibra/ref_dir/I03.bmp')
#
# # For FID metric, use directory or precomputed statistics as inputs
# # refer to clean-fid for more details: https://github.com/GaParmar/clean-fid
# fid_metric = pyiqa.create_metric('fid')
# score = fid_metric('./ResultsCalibra/dist_dir/', './ResultsCalibra/ref_dir')
# score = fid_metric('./ResultsCalibra/dist_dir/', dataset_name="FFHQ", dataset_res=1024, dataset_split="trainval70k")

import pyiqa
import torch
import torchvision.transforms as transforms
from PIL import Image

# 检查 CUDA 是否可用
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# 创建 LPIPS 度量
iqa_metric = pyiqa.create_metric('lpips', device=device)

# 加载和转换图像
img_x = Image.open('./ResultsCalibra/dist_dir/I04.bmp').convert('RGB')
img_y = Image.open('./ResultsCalibra/ref_dir/I04.bmp').convert('RGB')

# 定义转换，以将图像转换为张量
transform = transforms.Compose([
    transforms.Resize((2048, 2048)),  # 根据需要调整高度和宽度
    transforms.ToTensor(),
])

# 应用转换
img_tensor_x = transform(img_x).unsqueeze(0).to(device)  # 添加批次维度并移动到设备
img_tensor_y = transform(img_y).unsqueeze(0).to(device)

# 计算分数
score_fr = iqa_metric(img_tensor_x, img_tensor_y)
print("两张图像之间的分数:", score_fr.item())