# python 2.7, pytorch 0.3.1

import os, sys
sys.path.insert(1, '../')
import torch
import torchvision
import numpy as np
import subprocess
import random
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from PIL import Image
import cv2
from pietorch import data_convertors
from pietorch.DuRN_US import cleaner
from pietorch.pytorch_ssim import ssim as ssim
from skimage.measure import compare_psnr as psnr
from skimage.measure import compare_ssim as ski_ssim
import scipy.misc
#------- Option --------
tag = 'DuRN-US'
# Choose a dataset.
data_name = 'RESIDE' # 'DCPDNData' or 'RESIDE'
#-----------------------

if data_name == 'RESIDE':
    testroot = "../data/"+data_name+"/sots_indoor_test/"
    test_list_pth = "../lists/RESIDE_indoor/sots_test_list.txt"
elif data_name == 'DCPDNData':    
    testroot = "../data/"+data_name+"/TestA/"
    test_list_pth = '../lists/'+data_name+'/testA_list.txt'
else:
    print('Unknown dataset name.')

Pretrained = '../trainedmodels/'+data_name+'/'+tag+'_model.pt'    
show_dst = '../cleaned_images/'+data_name+'/'+tag+'/'
#subprocess.check_output(['mkdir', '-p', show_dst])

# Set transformer, convertor, and data_loader
transform = transforms.ToTensor()
convertor = data_convertors.ConvertImageSet(testroot, test_list_pth, data_name,
                                            transform=transform)
dataloader = DataLoader(convertor, batch_size=1, shuffle=False, num_workers=1)

# Make the network
cleaner = cleaner().cuda()
cleaner.load_state_dict(torch.load(Pretrained))
cleaner.eval()

I_HAZE = "D:\Image_dataset\# I-HAZY NTIRE 2018\hazy1\\"
I_HAZE_GT = "D:\Image_dataset\# I-HAZY NTIRE 2018\GT1\\"
O_HAZE="D:\Image_dataset\# O-HAZY NTIRE 2018\hazy1\\"
O_HAZE_GT="D:\Image_dataset\# O-HAZY NTIRE 2018\GT1\\"
SOTSI = "C:\\Users\FQL\Desktop\RESIDE-standard\SOTS\indoor\hazy\\"
SOTSI_GT = "C:\\Users\FQL\Desktop\RESIDE-standard\SOTS\indoor\gt1\\"
SOTSO = "F:\SOTS\hazy\\"
SOTSO_GT = "F:\SOTS\clear\\"
HSTSS = "F:\HSTS\HAZY\\"
HSTSS_GT = "F:\HSTS\CLEAR\\" #jpg

HSTSR = "F:\HSTS\REAL\\"
RTTS = "F:\RTTS\\"
origin_dir = "F:\haze\\"
pathname = "F:\haze\\"
output_dir = "F:\Durn-US\I-HAZE\\"
img_name_list = os.listdir(pathname)
img_sum = len(img_name_list)
time_sum = 0
for img_path in img_name_list:
    (imageName, extension) = os.path.splitext(img_path)
    img = Image.open(os.path.join(pathname, img_path)).convert('RGB')
    im_w, im_h = img.size
    if im_w % 4 != 0 or im_h % 4 != 0:
        img = img.resize((int(im_w // 4 * 4), int(im_h // 4 * 4))) 
    in_data = transforms.ToTensor()(img)
    in_data = in_data.unsqueeze_(0)
    start = cv2.getTickCount()
    with torch.no_grad():
        pred = cleaner(Variable(in_data.cuda()))
    out_img = pred.data[0].cpu().numpy().squeeze().transpose((1,2,0))
    end  = cv2.getTickCount()
    time_sum+=(end - start) / cv2.getTickFrequency()
    scipy.misc.toimage(out_img).save(os.path.join(output_dir+ imageName+ ".jpg"))
print(time_sum, img_sum)
print(time_sum/img_sum)
