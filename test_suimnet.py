"""
# Test script for the SUIM-Net
    # for 5 object categories: HD, FV, RO, RI, WR 
# Paper: https://arxiv.org/pdf/2004.01241.pdf  
"""
from __future__ import print_function, division
import os
import ntpath
import numpy as np
from PIL import Image
from os.path import join, exists
# local libs
from models.suim_net import SUIM_Net
from utils.data_utils import getPaths
# HOME_COLAB='/content/SUIM-dev'

DATASET             = 'DataSet_ConchasAbanico' #['SUIM', 'DataSet_ConchasAbanico']
HOME_COLAB_DRIVE    = '/content/drive/MyDrive/DATA/{}'.format(DATASET)
HOME_LOCAL          = ''
HOME_TO_USE         = HOME_COLAB_DRIVE


## experiment directories
#test_dir = "/mnt/data1/ImageSeg/suim/TEST/images/"
# test_dir = os.path.join(HOME_COLAB, "data/test/images/")
test_dir = os.path.join(HOME_TO_USE, "data/test/images/")

## sample and ckpt dir
#folder to save predicted segmentations
# samples_dir = "data/test/output/" 
samples_dir = os.path.join(HOME_TO_USE, "data/test/output/" )
if DATASET == "SUIM":
    RO_dir = samples_dir + "RO/"
    FB_dir = samples_dir + "FV/"
    WR_dir = samples_dir + "WR/"
    HD_dir = samples_dir + "HD/"
    RI_dir = samples_dir + "RI/" 
    if not exists(samples_dir): os.makedirs(samples_dir)
    if not exists(RO_dir): os.makedirs(RO_dir)
    if not exists(FB_dir): os.makedirs(FB_dir)
    if not exists(WR_dir): os.makedirs(WR_dir)
    if not exists(HD_dir): os.makedirs(HD_dir)
    if not exists(RI_dir): os.makedirs(RI_dir)
elif DATASET == "DataSet_ConchasAbanico":
    # CAB: Concha de abanico
    # VCA: Valva de concha de abanico (muerta)
    # VPP: Valva de pico de pato
    # CAR: Caracol
    # CAN: Cangrejo
    CAB_dir = samples_dir + "CAB/"
    VCA_dir = samples_dir + "VCA/"
    VPP_dir = samples_dir + "VPP/"
    CAR_dir = samples_dir + "CAR/"
    CAN_dir = samples_dir + "CAN/" 
    if not exists(samples_dir): os.makedirs(samples_dir)
    if not exists(test_dir): os.makedirs(test_dir)
    if not exists(CAB_dir): os.makedirs(CAB_dir)
    if not exists(VCA_dir): os.makedirs(VCA_dir)
    if not exists(VPP_dir): os.makedirs(VPP_dir)
    if not exists(CAR_dir): os.makedirs(CAR_dir)
    if not exists(CAN_dir): os.makedirs(CAN_dir)

## input/output shapes
base_ = 'VGG' # or 'RSB'
if base_=='RSB':
    im_res_ = (320, 240, 3) 
    ckpt_name = "suimnet_rsb5.hdf5"
else: 
    im_res_ = (320, 256, 3)
    ckpt_name = "suimnet_vgg.hdf5"
    # ckpt_name = "suimnet_vgg_100.hdf5"
suimnet = SUIM_Net(base=base_, im_res=im_res_, n_classes=5)
model = suimnet.model
# print (model.summary())
model.load_weights(os.path.join(HOME_TO_USE, "ckpt/", ckpt_name))


im_h, im_w = im_res_[1], im_res_[0]
def testGenerator():
    # test all images in the directory
    assert exists(test_dir), "local image path doesnt exist"
    imgs = []
    for p in getPaths(HOME_TO_USE, test_dir):
        # read and scale inputs
        img = Image.open(p).resize((im_w, im_h))
        img = np.array(img)/255.
        img = np.expand_dims(img, axis=0)
        # inference
        out_img = model.predict(img)
        # print('Out_img size: ', out_img.shape)
        # thresholding
        out_img[out_img>0.5] = 1.
        out_img[out_img<=0.5] = 0.
        print ("tested: {0}".format(p))
        # get filename
        img_name = ntpath.basename(p).split('.')[0] + '.bmp'
        # save individual output masks
        if DATASET == "SUIM":
            ROs = np.reshape(out_img[0,:,:,0], (im_h, im_w))
            FVs = np.reshape(out_img[0,:,:,1], (im_h, im_w))
            HDs = np.reshape(out_img[0,:,:,2], (im_h, im_w))
            RIs = np.reshape(out_img[0,:,:,3], (im_h, im_w))
            WRs = np.reshape(out_img[0,:,:,4], (im_h, im_w))
            Image.fromarray(np.uint8(ROs*255.)).save(RO_dir+img_name)
            Image.fromarray(np.uint8(FVs*255.)).save(FB_dir+img_name)
            Image.fromarray(np.uint8(HDs*255.)).save(HD_dir+img_name)
            Image.fromarray(np.uint8(RIs*255.)).save(RI_dir+img_name)
            Image.fromarray(np.uint8(WRs*255.)).save(WR_dir+img_name)
        elif DATASET == "DataSet_ConchasAbanico":
            CABs = np.reshape(out_img[0,:,:,0], (im_h, im_w))
            VCAs = np.reshape(out_img[0,:,:,1], (im_h, im_w))
            VPPs = np.reshape(out_img[0,:,:,2], (im_h, im_w))
            CARs = np.reshape(out_img[0,:,:,3], (im_h, im_w))
            CANs = np.reshape(out_img[0,:,:,4], (im_h, im_w))
            Image.fromarray(np.uint8(CABs*255.)).save(CAB_dir+img_name)
            Image.fromarray(np.uint8(VCAs*255.)).save(VCA_dir+img_name)
            Image.fromarray(np.uint8(VPPs*255.)).save(VPP_dir+img_name)
            Image.fromarray(np.uint8(CARs*255.)).save(CAR_dir+img_name)
            Image.fromarray(np.uint8(CANs*255.)).save(CAN_dir+img_name)
            

# test images
testGenerator()


