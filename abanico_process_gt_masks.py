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
from globals import CLASSES

DATASET                 = 'DataSet_ConchasAbanico' #['SUIM', 'DataSet_ConchasAbanico']
HOME_COLAB_DRIVE        = '/content/drive/MyDrive/DATA/{}'.format(DATASET)
HOME_LOCAL              = ''
HOME_LOCAL_DATASET_WIN  = 'C:/Users/David/Desktop/DATASETS/{}'.format(DATASET)
HOME_TO_USE             = HOME_LOCAL_DATASET_WIN

masks_dir = os.path.join(HOME_TO_USE, "TEST/masks/")
masks_process_dir = os.path.join(HOME_TO_USE, "TEST/masks_process2/" )
    
CAB_dir = masks_process_dir + "CAB/"
VCA_dir = masks_process_dir + "VCA/"
VPP_dir = masks_process_dir + "VPP/"
CAR_dir = masks_process_dir + "CAR/"
CAN_dir = masks_process_dir + "CAN/" 
if not exists(masks_process_dir): os.makedirs(masks_process_dir)
if not exists(CAB_dir): os.makedirs(CAB_dir)
if not exists(VCA_dir): os.makedirs(VCA_dir)
if not exists(VPP_dir): os.makedirs(VPP_dir)
if not exists(CAR_dir): os.makedirs(CAR_dir)
if not exists(CAN_dir): os.makedirs(CAN_dir)

n_classes=6
#Define a function to perform additional preprocessing after datagen.
#For example, scale images, convert masks to categorical, etc. 
def preprocess_data(mask, num_class):
    #Scale images
    # img = img / 255. #This can be done in ImageDataGenerator but showing it outside as an example
    #Convert mask to one-hot
    # labelencoder = LabelEncoder()
    # h, w = mask.shape  
    # mask = mask.reshape(-1,1)
    # mask = labelencoder.fit_transform(mask)
    # mask = mask.reshape(h, w)

    classes_idx = np.unique(mask)
    # print("Classes in mask: ", clases_idx)
    mask = to_categorical(mask, num_class)
      
    return mask, classes_idx


mask_paths = getPaths(HOME_TO_USE, masks_dir)
for p in mask_paths:
    print(p)
print(len(mask_paths))

    
def mask_2_classes(mask_paths):
    for i,p in enumerate(mask_paths):
        # read and scale inputs
        img = Image.open(p)
        # img = img.resize((im_w, im_h))
        # img = img.convert('RGB')
        img = np.array(img)
        # img = cv2.imread(p, 0)
        im_h, im_w = img.shape
        print("\nUnique values in the mask are: ", np.unique(img))
        print('img {}, {}'.format(i, img.shape))
        # img = np.array(img)/255.
        # img = np.expand_dims(img, axis=0)
        # if i==1: break
        # plt.show()

        mask, _ = preprocess_data(img,n_classes)
        # mask_classes = np.zeros(len(classes)).tolist()
        # print('mask: ', mask.shape)

        img_name = p.split('/')[-1][:-4]
        print('img_name: ', img_name)

        # imgplot = plt.imshow(mask[:,:,0])
        # fig, ax = plt.subplots(nrows=1, ncols=n_classes+1)
        # plt.title(img_name)
        # ax[0].imshow(mask[:,:,0]) #background
        # ax[1].imshow(mask[:,:,1]) #CAB
        # ax[2].imshow(mask[:,:,2]) #VCA
        # ax[3].imshow(mask[:,:,3]) #VPP
        # ax[4].imshow(mask[:,:,4]) #CAR
        # ax[5].imshow(mask[:,:,5]) #CAN
        # ax[6].imshow(img)
        # plt.show()
        
        CAB = np.reshape(mask[:,:,1], (im_h, im_w))
        VCA = np.reshape(mask[:,:,2], (im_h, im_w))
        VPP = np.reshape(mask[:,:,3], (im_h, im_w))
        CAR = np.reshape(mask[:,:,4], (im_h, im_w))
        CAN = np.reshape(mask[:,:,5], (im_h, im_w))
        # Image.fromarray(np.uint8(CAB*255.)).save(CAB_dir+img_name + '.bmp')
        # Image.fromarray(np.uint8(VCA*255.)).save(VCA_dir+img_name + '.bmp')
        # Image.fromarray(np.uint8(VPP*255.)).save(VPP_dir+img_name + '.bmp')
        # Image.fromarray(np.uint8(CAR*255.)).save(CAR_dir+img_name + '.bmp')
        # Image.fromarray(np.uint8(CAN*255.)).save(CAN_dir+img_name + '.bmp')

if __name__ == "__main__":
    mask_2_classes(mask_paths)
