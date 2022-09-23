import os
import numpy as np
from PIL import Image
from os.path import join, exists
from utils.data_utils import getPaths
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import cv2
from utils.measure_utils import plot_debug
import pandas as pd
import fnmatch
import cv2
from globals import CLASSES

DATASET                 = 'DataSet_ConchasAbanico' #['SUIM', 'DataSet_ConchasAbanico']
HOME_COLAB_DRIVE        = '/content/drive/MyDrive/DATA/{}'.format(DATASET)
HOME_LOCAL              = ''
HOME_LOCAL_DATASET_WIN  = 'C:/Users/David/Desktop/DATASETS/{}'.format(DATASET)
HOME_TO_USE             = HOME_LOCAL_DATASET_WIN

images_dir = os.path.join(HOME_TO_USE, "train_val/images/")
# masks_process_dir = os.path.join(HOME_TO_USE, "TEST/masks_process2/" )
masks_dir = os.path.join(HOME_TO_USE, "train_val/masks/")
# masks_gen = os.path.join(HOME_TO_USE, "train_val/masks/" )
    
def getPaths(root, data_dir):
    # read image files from directory
    exts = ['*.png','*.JPG', '*.JPEG', '*.bmp']
    # exts = ['*.PNG','*.jpg','*.JPG', '*.JPEG', '*.bmp']
    image_paths = []
    for pattern in exts:
        for d, s, fList in os.walk(data_dir):
            # print('fList: ', len(fList))
            for filename in fList:
                if (fnmatch.fnmatch(filename, pattern)):
                    fname_ = os.path.join(root,d,filename)
                    image_paths.append(fname_)
    return image_paths

def getFileNames(paths):
    names=[]
    for p in paths:
        n = p.split('/')[-1][:-4]
        names.append(n)
    return names

n_classes=6

images_paths = getPaths(HOME_TO_USE, images_dir)
mask_paths = getPaths(HOME_TO_USE, masks_dir)
for p in images_paths:
    print(p)
print(len(images_paths))

for p in mask_paths:
    print(p)
print(len(mask_paths))


images_names = getFileNames(images_paths)
mask_names = getFileNames(mask_paths)

images_without_masks = [im for im in images_names if im not in mask_names]

print('----images: \n', images_names)
print('----masks: \n', mask_names)
print('----images_without_masks: \n', images_without_masks)

w,h = 320, 240
img = np.zeros((h,w,3),dtype=np.uint8)
for im in images_without_masks:
    Image.fromarray(img).convert("RGB").save("{}.png".format(masks_dir+im))


# if __name__ == "__main__":
#     mask_2_classes(mask_paths)
