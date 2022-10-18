import matplotlib.pyplot as plt
import SimpleITK as sitk
import numpy as np
import os
import glob
from skimage.measure import label, regionprops
from skimage.segmentation import clear_border
from skimage.feature import peak_local_max
import random
from multiprocessing import Pool
from medpy.metric.binary import obj_fpr, obj_tpr
from sklearn.metrics import roc_curve, auc
import pandas as pd
from skimage.segmentation import clear_border

def compute_fpr_tpr(ref, pred):
    fpr = {}
    tpr = {}

    for index, target in enumerate(['GTVt', 'GTVn']): # calculate GTVt and GTVn
        ref_tmp = ref==index+1
        pred_tmp = pred==index+1
        print(index, '-', ref_tmp.sum(), '-', pred_tmp.sum())       
        #print(ref_tmp.shape)
        if ref_tmp.sum()< 500 and pred_tmp.sum()< 500: #set minimal ref volume for each target 
            fpr[target] = 0
            tpr[target] = 1
        elif (ref_tmp.sum()< 10 and pred_tmp.sum()>500):
            print(index, 'no ref but pred_tmp prediced')
            fpr[target] = 1
            tpr[target] = 0
        else: 
            if pred_tmp.sum()>10:
                fpr[target] = obj_fpr(pred_tmp, ref_tmp)
                tpr[target] = obj_tpr(pred_tmp, ref_tmp)
            else:
                print(index, 'pred_tmp not enough pixels')
                fpr[target] = 1
                tpr[target] = 0

#             continue
#             fpr[target] = np.nan
#             tpr[target] = np.nan
    return fpr, tpr

def aggregate_scores(ref_path, pred_path):
    ref_files = sorted(glob.glob(os.path.join(ref_path, '*.nii.gz')))
    fpr = []
    tpr = [] 
    
    for ref_file in ref_files:
        pred_file = ref_file.replace(ref_path, pred_path)
        print(pred_file)
        try:
            ref = sitk.GetArrayFromImage(sitk.ReadImage(ref_file))
            pred = sitk.GetArrayFromImage(sitk.ReadImage(pred_file))
        except e:
            print("reference and prediciton file name not match")
        cleared_ref = clear_border(ref)
        cleared_pred = clear_border(pred) 
        _fpr, _tpr = compute_fpr_tpr(ref, pred)
        fpr.append(_fpr)
        tpr.append(_tpr)
        
    df_fpr = pd.DataFrame.from_dict(fpr)
    df_tpr = pd.DataFrame.from_dict(tpr)
    
    return df_fpr, df_tpr

"""
roc_auc = {}
roc_auc["GTVt"] = auc(sorted(fpr["GTVt"]), sorted(tpr["GTVt"]))
roc_auc["GTVn"] = auc(sorted(fpr["GTVn"]), sorted(tpr["GTVn"]))


plt.figure()
plt.plot(
    sorted(tpr["GTVt"]),
    sorted(fpr["GTVt"]),
    label="GTVt ROC curve (area = {0:0.2f})".format(roc_auc["GTVt"]),
    color="green",
    linestyle=":",
    linewidth=2,
)

plt.scatter(
    sorted(fpr["GTVn"]),
    sorted(tpr["GTVn"]),
    label="GTVn ROC curve (area = {0:0.2f})".format(roc_auc["GTVn"]),
    color="navy",
    linestyle=":",
    linewidth=2,
)


plt.plot([0, 1], [0, 1], "k--", lw=1)
plt.xlim([-0.02, 1.01])
plt.ylim([0, 1.01])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver operating characteristic ")
plt.legend(loc="lower right")
plt.show()
"""
ref_path = 'ref'
pred_path = 'unet_pred'
fpr, tpr = aggregate_scores(ref_path, pred_path)


if __name__ == '__main__':
    work_path = '/data/jintao/nnUNet/nnUNet_raw_data_base/nnUNet_raw_data/Task102_auh_od2/labelsOd'
    gtv_path = glob.glob(os.path.join(work_path, '*'))
    # for path in gtv_path:
    #     compute_roc(path)

    
    # #compare(files[0])
    with Pool(64) as p:
        p.map(simulate_user_feeds, gtv_path)