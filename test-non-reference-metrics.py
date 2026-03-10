import os
import glob
import torch
from pyiqa import create_metric
from PIL import Image
import torchvision.transforms as transforms

image_dir = '.../.../.../sr_output_dir'  
image_list = glob.glob(os.path.join(image_dir, '*.png'))  

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


clipiqa_metric = create_metric('clipiqa', device=device)
musiq_metric = create_metric('musiq', device=device)
maniqa_metric1 = create_metric('maniqa', device=device)
maniqa_metric2 = create_metric('maniqa-pipal', device=device)
niqe_metric = create_metric('niqe', device=device)

def load_image(image_path):
    transform = transforms.Compose([
        transforms.ToTensor()  #
    ])
    image = Image.open(image_path).convert('RGB')  
    return transform(image).unsqueeze(0) 



clipiqa_scores = []
musiq_scores = []
maniqa1_scores = []
maniqa2_scores = []
niqe_scores = []

for img_path in image_list:
    # 加载图像并计算指标
    img = load_image(img_path).to(device)

    clipiqa_scores.append(clipiqa_metric(img).item())
    musiq_scores.append(musiq_metric(img).item())
    maniqa1_scores.append(maniqa_metric1(img).item())
    maniqa2_scores.append(maniqa_metric2(img).item())
    niqe_scores.append(niqe_metric(img).item())

print(f'CLIPIQA : {torch.mean(torch.tensor(clipiqa_scores)):.4f}')
print(f'MUSIQ : {torch.mean(torch.tensor(musiq_scores)):.4f}')
print("There are tow methods of calculating MANIQA:")
print(f'MANIQA-1 : {torch.mean(torch.tensor(maniqa1_scores)):.4f}')
print(f'MANIQA-2 : {torch.mean(torch.tensor(maniqa2_scores)):.4f}')
print(f'NIQE : {torch.mean(torch.tensor(niqe_scores)):.4f}')