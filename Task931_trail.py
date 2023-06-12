
import numpy as np
from collections import OrderedDict

from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.paths import nnUNet_raw_data
import SimpleITK as sitk
import shutil

"""

"""

if __name__ == "__main__":
    """
    REMEMBER TO CONVERT LABELS BACK TO BRATS CONVENTION AFTER PREDICTION!
    """

    task_name = "Task931_trail"
    data_dir = "/mnt/data/jintao/hndata/Task901_all/"
    split_table = "data_split_GTV_trail.json"

    with open('data_split_GTV_trail.json') as file:
        split = json.load(file)

    train_patients = split['train']
    test_patients = split['test']
    od_patients = split['other_sites']

    target_base = join(nnUNet_raw_data, task_name)
    
    cur = data_dir
    imagedir = join(cur, 'images')
    labeldir = join(cur, 'labels')

    for split_group in ['train', 'test', 'other_sites']: 
        if split_group == 'train':
            target_images = join(target_base, "imagesTr")
            target_labels = join(target_base, "labelsTr")
        elif split_group == 'test':
            target_images = join(target_base, "imagesTs")
            target_labels = join(target_base, "labelsTs")
        elif split_group == 'other_sites':
            target_images = join(target_base, "imagesOd")
            target_labels = join(target_base, "labelsOd")

        maybe_mkdir_p(target_images)
        maybe_mkdir_p(target_labels)
        patients = split[split_group]

        for p in patients:
            print(imagedir)
            patient_name = p
            ct = join(imagedir, p+"_0000.nii.gz")
            pt = join(imagedir, p+"_0001.nii.gz")
            t1 = join(imagedir, p+"_0002.nii.gz")
            t2 = join(imagedir, p+"_0003.nii.gz")
            gtv = join(labeldir, patient_name+".nii.gz")

            print(ct)
            assert all([
                isfile(ct),
                isfile(pt),
                isfile(t1),
                isfile(t2),
                isfile(gtv)
            ]), "%s" % patient_name

            shutil.copy(ct, join(target_images, patient_name + "_0000.nii.gz"))
            shutil.copy(pt, join(target_images, patient_name + "_0001.nii.gz"))
            shutil.copy(pt, join(target_images, patient_name + "_0002.nii.gz"))
            shutil.copy(pt, join(target_images, patient_name + "_0003.nii.gz"))
            shutil.copy(gtv, join(target_labels, patient_name + ".nii.gz"))

        #copy_BraTS_segmentation_and_convert_labels(seg, join(target_labelsTr, patient_name + ".nii.gz"))

    json_dict = OrderedDict()
    json_dict['name'] = "trail_study"
    json_dict['description'] = "nothing"
    json_dict['tensorImageSize'] = "4D"
    json_dict['reference'] = "NA"
    json_dict['licence'] = "NA"
    json_dict['release'] = "0.0"
    json_dict['modality'] = {
        "0": "CT",
        "1": "PT",
        "2": "T1dr",
        "3": "T2dr"
    }
    json_dict['labels'] = {
        "0": "background",
        "1": "GTVt",
        "2": "GTVn"
    }
    json_dict['numTraining'] = len(train_patients)
    json_dict['numTest'] = len(test_patients)
    json_dict['numOd'] = len(od_patients)
    json_dict['training'] = [{'image': "./imagesTr/%s.nii.gz" % i, "label": "./labelsTr/%s.nii.gz" % i} for i in
                             train_patients]


    json_dict['test'] =  [{'image': "./imagesTs/%s.nii.gz" % i, "label": "./labelsTs/%s.nii.gz" % i} for i in
                          test_patients]
    json_dict['od'] =  [{'image': "./imagesOd/%s.nii.gz" % i, "label": "./labelsOd/%s.nii.gz" % i} for i in
                          od_patients]
   
    save_json(json_dict, join(target_base, "dataset.json"))