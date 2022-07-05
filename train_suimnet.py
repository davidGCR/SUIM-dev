"""
# Training pipeline of the SUIM-Net
# Paper: https://arxiv.org/pdf/2004.01241.pdf  
"""
from __future__ import print_function, division
import os
import math
from os.path import join, exists
from keras import callbacks
# local libs
from models.suim_net import SUIM_Net
from utils.data_utils import trainDataGenerator
HOME_COLAB='/content/SUIM-dev'

## dataset directory
dataset_name = "suim"
train_dir = os.path.join(HOME_COLAB, "/mnt/data1/ImageSeg/suim/train_val/")

## ckpt directory
ckpt_dir = os.path.join(HOME_COLAB, "ckpt/")
base_ = 'VGG' # or 'RSB'
if base_=='RSB':
    im_res_ = (320, 240, 3) 
    ckpt_name = "suimnet_rsb.hdf5"
else: 
    im_res_ = (320, 256, 3)
    ckpt_name = "suimnet_vgg.hdf5"
model_ckpt_name = join(ckpt_dir, ckpt_name)
if not exists(ckpt_dir): os.makedirs(ckpt_dir)

## initialize model
suimnet = SUIM_Net(base=base_, im_res=im_res_, n_classes=5)
model = suimnet.model
print (model.summary())
## load saved model
#model.load_weights(join("ckpt/saved/", "***.hdf5"))


# batch_size = 8
batch_size = 1
num_epochs = 50
# setup data generator
data_gen_args = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')

model_checkpoint = callbacks.ModelCheckpoint(model_ckpt_name, 
                                   monitor = 'loss', 
                                   verbose = 1, mode= 'auto',
                                   save_weights_only = True,
                                   save_best_only = True)

# data generator
train_gen = trainDataGenerator(batch_size, # batch_size 
                              train_dir,# train-data dir
                              "images", # image_folder 
                              "masks", # mask_folder
                              data_gen_args, # aug_dict
                              image_color_mode="rgb", 
                              mask_color_mode="rgb",
                              target_size = (im_res_[1], im_res_[0]))

import matplotlib.pyplot as plt
import numpy as np
fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(15,15))

for i in range(4):

  # convert to unsigned integers for plotting
  image = next(train_gen)[0].astype('uint8')

  # changing size from (1, 200, 200, 3) to (200, 200, 3) for plotting the image
  image = np.squeeze(image)

  # plot raw pixel data
  ax[i].imshow(image)
  ax[i].axis('off')

## fit model
# model.fit_generator(train_gen, 
#                     steps_per_epoch = 5000,
#                     epochs = num_epochs,
#                     callbacks = [model_checkpoint])

