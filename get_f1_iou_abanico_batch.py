"""
# Script for evaluating F score and mIOU 
"""
from __future__ import print_function, division
import ntpath
import numpy as np
from PIL import Image
import os
# local libs
from utils.data_utils import getPaths
from utils.measure_utils import db_eval_boundary, IoU_bin
from globals import CLASSES
from abanico_data_exploration import count_by_class
import matplotlib.pyplot as plt

DATASET             = 'DataSet_ConchasAbanico' #['SUIM', 'DataSet_ConchasAbanico']
HOME_COLAB_DRIVE    = '/content/drive/MyDrive/DATA/{}'.format(DATASET)
HOME_LOCAL          = ''
HOME_LOCAL_DATASET_WIN  = 'C:/Users/David/Desktop/DATASETS/{}'.format(DATASET)
# HOME_TO_USE         = HOME_LOCAL_DATASET_WIN
HOME_TO_USE         = HOME_COLAB_DRIVE

## experiment directories
obj_cat         = "CAB" # sub-dir  ["RI/", "FV/", "WR/", "RO/", "HD/"]
test_dir_process= os.path.join(HOME_TO_USE,"data/test/masks_process") if HOME_TO_USE==HOME_COLAB_DRIVE else os.path.join(HOME_TO_USE,"TEST/masks_process")
test_dir        = os.path.join(HOME_TO_USE, "TEST/masks/")
#test_dir = "/mnt/data1/ImageSeg/suim/TEST/masks/"
real_mask_dir   = os.path.join(test_dir_process, obj_cat) # real labels
gen_mask_dir    = os.path.join(HOME_TO_USE, "data/test/output", obj_cat) if HOME_TO_USE==HOME_COLAB_DRIVE else os.path.join(HOME_TO_USE, "TEST/output", obj_cat)# generated labels
## input/output shapes
im_res = (320, 240)

# count_by_class('test_samples_x_clases.csv')

def AP(iou_scores, threshold, clase):
    TP = 0
    FP = 0
    precisions = []
    recalls    = []
    # iou_scores.sort(reverse=True)
    # print('iou_scores sorted: ', iou_scores)
    for iou in iou_scores:
        if iou >= threshold:
            TP += 1
        else:
            FP += 1
        precision = TP/(TP+FP)
        recall = TP/count_by_class('test_samples_x_clases.csv',clase)
        precisions.append(precision)
        recalls.append(recall)
    for i in range(len(precisions)-1, 0, -1):
        if precisions[i] > precisions[i-1]:
            precisions[i-1] = precisions[i]
    return precisions, recalls
    

# for reading and scaling input images
def read_and_bin(im_path):
    img = Image.open(im_path).resize(im_res)
    img = np.array(img)/255.
    img[img >= 0.5] = 1
    img[img < 0.5] = 0
    return img

# accumulate F1/iou values in the lists
Ps, Rs, F1s, IoUs = [], [], [], []
gen_paths = sorted(getPaths(HOME_TO_USE, gen_mask_dir))
real_paths = sorted(getPaths(HOME_TO_USE, real_mask_dir))
print('gen_paths: ', gen_paths)

for gen_p, real_p in zip(gen_paths, real_paths):
    gen, real = read_and_bin(gen_p), read_and_bin(real_p)
    if (np.sum(real)>0):
        precision, recall, F1 = db_eval_boundary(real, gen)
        iou = IoU_bin(real, gen)
        #print ("{0}:>> P: {1}, R: {2}, F1: {3}, IoU: {4}".format(gen_p, precision, recall, F1, iou))
        Ps.append(precision)
        Rs.append(recall)
        F1s.append(F1)
        IoUs.append(iou)

print("IoUs: ", IoUs, len(IoUs))
# print F-score and mIOU in [0, 100] scale
print ("Avg. F: {0}".format(100.0*np.mean(F1s)))
print ("Avg. IoU: {0}".format(100.0*np.mean(IoUs)))

precisions, recalls = AP(IoUs, threshold=0.5, clase='CAB')
plt.plot(recalls, precisions, linewidth=4, color="red")
plt.xlabel("Recall", fontsize=12, fontweight='bold')
plt.ylabel("Precision", fontsize=12, fontweight='bold')
plt.title("Precision-Recall Curve", fontsize=15, fontweight="bold")
plt.show()
    

