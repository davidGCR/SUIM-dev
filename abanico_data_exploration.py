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
masks_process_dir = os.path.join(HOME_TO_USE, "TEST/masks_process/" )

def count_by_class(csv_df_file, class_name='CAB'):
    if os.path.exists(csv_df_file):
        samples_x_clases_df = pd.read_csv(csv_df_file)
        # print(samples_x_clases_df.head(5))
    else:
        n_classes=6
        mask_paths = getPaths(HOME_TO_USE, masks_dir)
        samples_x_clases_df = explore_data(mask_paths, n_classes=n_classes, plot=False)
    counter = samples_x_clases_df[class_name].value_counts().get(key=1)
    # print('counter: ', type(counter), counter)
    # print('counter.get(key=0): ', counter.get(key=1))
    return counter

def explore_data(mask_paths, n_classes=6, plot=False):
    samples_x_clases_df = pd.DataFrame(columns=CLASSES) 
                                        # index=list(range(len(mask_paths))))
    for i,p in enumerate(mask_paths):
        # read and scale inputs
        img = Image.open(p)
        img = np.array(img)
        im_h, im_w = img.shape
        classes_idx = np.unique(img)
        mask_classes = len(CLASSES)*[0] #list of Zeros
        # image_name = p.split('/')[-1][0:-4]
        image_name = p.split('/')[-1]
        print(image_name)
        for k in range(len(mask_classes)):
            if k+1 in classes_idx:
                mask_classes[k] = 1
        samples_x_clases_df.loc[image_name] = mask_classes
        print("Classes in mask: ", classes_idx)
        print("mask_classes: ", mask_classes)
        # imgplot = plt.imshow(mask[:,:,0])
        if plot:
            fig, ax = plt.subplots(nrows=1, ncols=n_classes+1)
            plt.title(img_name)
            ax[0].imshow(mask[:,:,0]) #background
            ax[1].imshow(mask[:,:,1]) #CAB
            ax[2].imshow(mask[:,:,2]) #VCA
            ax[3].imshow(mask[:,:,3]) #VPP
            ax[4].imshow(mask[:,:,4]) #CAR
            ax[5].imshow(mask[:,:,5]) #CAN
            ax[6].imshow(img)
            plt.show()
    samples_x_clases_df.to_csv('test_samples_x_clases.csv', index=True)
    # print(samples_x_clases_df.head(20))
    return samples_x_clases_df
    
if __name__ == "__main__":
    n_classes=6
    mask_paths = getPaths(HOME_TO_USE, masks_dir)
    for p in mask_paths:
        print(p)
    print(len(mask_paths))
    explore_data(mask_paths, n_classes=n_classes, plot=False)
