import SimpleITK as sitk
import numpy as np
import os
import glob
from skimage.measure import label, regionprops
from skimage.segmentation import clear_border
from skimage.feature import peak_local_max
import random
from multiprocessing import Pool
def small_volumes(gtv_path):
    img = sitk.ReadImage(gtv_path)
    arr = sitk.GetArrayFromImage(img)

    cleared = clear_border(arr)

    label_image = label(cleared)
    #coords = peak_local_max(arr)
    #print(coords)
    #coords = np.asarray(np.column_stack(np.ma.where(arr >0)))
    #print(coords)
    for region in regionprops(label_image):
        if region.area<100:
            print(gtv_path)


def simulate_user_feeds(gtv_path):
    iter_limit = 10000
    out_put_path = gtv_path.replace('labelsTr', 'imagesTr')
    out_put_path = out_put_path.replace('.nii.gz','_0004.nii.gz')
    #print(out_put_path)
    img = sitk.ReadImage(gtv_path)
    arr = sitk.GetArrayFromImage(img)

    cleared = clear_border(arr)
    #print(cleared.shape)
    label_image = label(cleared)
    #coords = peak_local_max(arr)
    #print(coords)
    #coords = np.asarray(np.column_stack(np.ma.where(arr >0)))
    #print(coords)

    new_arr = np.zeros(arr.shape)
    for region in regionprops(label_image):
        
        bbox = region.bbox
        point_not_found = True
        iterations = 0
        volume = region.area
        if volume>100: # only label volume larger than 100 pixels
            
            while point_not_found and iterations<iter_limit:
                radius = random.choice([2,3,4,5]) 
                if iterations > (iter_limit/3) and iterations % 100==0:
                    print(iterations, gtv_path)
                    if volume < 500:
                        radius = 2
                
                iterations+=1
                x = random.choice(np.linspace(bbox[0], bbox[3], dtype=int))
                y = random.choice(np.linspace(bbox[1], bbox[4], dtype=int))
                z = random.choice(np.linspace(bbox[2], bbox[5], dtype=int))
                points_inside = True
                
                temp_arr = np.zeros(arr.shape)
                for xi in range(x-radius, x+radius, 1):
                    for yi in range(y-radius, y+radius, 1):
                        for zi in range(z-radius, z+radius, 1):
                            # check if xi, yi, zi inside a tumor.
                                if arr[xi,yi,zi]==0:
                                    points_inside=False
                                else:
                                    temp_arr[xi,yi,zi] = arr[xi,yi,zi]
                                    
                if points_inside:
                    point_not_found = False
            
            new_arr = new_arr + temp_arr
        if iterations > iter_limit:
            print(gtv_path)   
        #print(region.area)
        #print(region._label_image)
    new_img = sitk.GetImageFromArray(new_arr)
    new_img.CopyInformation(img)
    sitk.WriteImage(new_img,out_put_path)

if __name__ == '__main__':
    work_path = '/mnt/data/jintao/nnUNet/nnUNet_raw_data_base/nnUNet_raw_data/Task001_user/labelsTr'

    # '/mnt/data/jintao/nnUNet/nnUNet_raw_data_base/nnUNet_raw_data/Task931_trail/'
    gtv_path = glob.glob(os.path.join(work_path, '*'))
    # for path in gtv_path:
    #     simulate_user_feeds(path)

    
    # #compare(files[0])
    with Pool(80) as p:
        p.map(simulate_user_feeds, gtv_path)
        #p.map(small_volumes, gtv_path)