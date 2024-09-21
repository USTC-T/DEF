import os
import torch
from torchvision.utils import save_image
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F

from model import TCL_AODNet  # 根据需要导入你的模型
from option_train import opt  # 确保 opt 配置文件已正确加载

# 创建输出文件夹
output_dir = './outs'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 设置设备
device = torch.device(opt.device if torch.cuda.is_available() else 'cpu')

# 加载模型
net = TCL_AODNet.dehaze_net().to(device)
net.eval()

# 加载预训练的模型权重
model_path = '/home/tangcheng/GithubProject/DEA-Net/exps/AOD/tcl/outdoor/1000iter_outdoor/saved_model/9490.pk'  # 替换为实际的模型权重路径
# 加载检查点文件
checkpoint = torch.load(model_path, map_location=device)

# 检查是否有 'model' 字段
if 'model' in checkpoint:
    net.load_state_dict(checkpoint['model'])
    print("Model weights loaded successfully.")
else:
    print("Error: 'model' key not found in the checkpoint. Please check the .pk file.")
# checkpoint = torch.load(model_path, map_location=device)
# net.load_state_dict(checkpoint['model'])

# 图片预处理
transform = transforms.Compose([
    transforms.ToTensor(),
])

def pad_img(x, patch_size):
    _, _, h, w = x.size()
    mod_pad_h = (patch_size - h % patch_size) % patch_size
    mod_pad_w = (patch_size - w % patch_size) % patch_size
    x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
    return x

# 推理函数
def infer_image(img_path, output_dir):
    # 加载图片
    img = Image.open(img_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device)  # 添加 batch 维度

    # 图片填充
    H, W = img_tensor.shape[2:]
    img_tensor = pad_img(img_tensor, 4)

    # 推理
    with torch.no_grad():
        output = net(img_tensor).clamp(0, 1)
        output = output[:, :, :H, :W]  # 去除填充

    # 保存推理结果
    img_name = os.path.basename(img_path)
    save_image(output, os.path.join(output_dir, f'result_{img_name}'))
    print(f"Saved result to {os.path.join(output_dir, f'result_{img_name}')}")


if __name__ == "__main__":
    # 手动设置图片路径
    img_paths = ['/home/tangcheng/GithubProject/DEA-Net/dataset/outdoor/test/hazy/0046.png']  # 替换为你的实际图片路径
    
    for img_path in img_paths:
        infer_image(img_path, output_dir)
