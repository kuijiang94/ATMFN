import os
from PIL import Image
from PIL import ImageChops
import numpy as np
import math
#import cv2

if __name__ == '__main__':
    data_path = 'F:\\jiangkui\\shiyan\\face\\GAN\\src\\lfw\\data\\lfw1\\'
    #file = os.listdir(data_path)
    save_path = 'F:\\jiangkui\\shiyan\\face\\GAN\\src\\lfw\\data\\lfw1\\'
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    #count =0
    #num =0
    files = [
        os.path.join(data_path, filename)
        for filename in os.listdir(data_path)
        if 'png' in filename or 'tif' in filename  or 'jpg' in filename]
    #count =0
    for filename in files:
        #if count%5==0:
        pic_path = os.path.join(data_path, filename)
        img = Image.open(pic_path)
        #w,h,c = img.shape
        img = img.crop([img.size[0]//2-48,img.size[1]//2-48,img.size[0]//2-48+96,img.size[1]//2-48+96])
        
        img.save(os.path.join(save_path, filename))