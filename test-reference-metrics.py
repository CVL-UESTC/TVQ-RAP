import os
import torch
from pyiqa import create_metric
from PIL import Image
import torchvision.transforms as transforms


real_dir = '.../.../.../gt_dir'  
fake_dir = '.../.../.../sr_output_dir' 
real_images = sorted(os.listdir(real_dir))  
fake_images = sorted(os.listdir(fake_dir))  


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 图像预处理
def load_image(image_path):
    transform = transforms.Compose([
        transforms.ToTensor(),              
    ])
    image = Image.open(image_path).convert('RGB')  
    return transform(image).unsqueeze(0)          



lpips_metric = create_metric('lpips', device=device)
dists_metric = create_metric('dists', device=device)
psnr_metric = create_metric('psnr', test_y_channel=True, color_space='ycbcr',device=device)
ssim_metric = create_metric('ssim',test_y_channel=True, color_space='ycbcr', device=device)
fid_metric = create_metric('fid', device=device)


lpips_scores = []
dists_scores = []
psnr_scores = []
ssim_scores = []

for real_img_name, fake_img_name in zip(real_images, fake_images):
    real_path = os.path.join(real_dir, real_img_name)
    fake_path = os.path.join(fake_dir, fake_img_name)

    img1 = load_image(real_path).to(device)
    img2 = load_image(fake_path).to(device)

    
    lpips_scores.append(lpips_metric(img1, img2).item())
    dists_scores.append(dists_metric(img1, img2).item())
    psnr_scores.append(psnr_metric(img1, img2).item())
    ssim_scores.append(ssim_metric(img1, img2).item())

fid_score = fid_metric(real_dir, fake_dir)


print(f'LPIPS : {torch.mean(torch.tensor(lpips_scores)):.4f}')
print(f'DISTS : {torch.mean(torch.tensor(dists_scores)):.4f}')
print(f'PSNR : {torch.mean(torch.tensor(psnr_scores)):.4f}')
print(f'SSIM : {torch.mean(torch.tensor(ssim_scores)):.4f}')
print(f'FID Score: {fid_score.item():.4f}')
