import numpy as np
import pydicom
from PIL import Image
import glob
import os
import pydicom.uid
import tensorflow as tf
from matplotlib import pyplot as plt
import cv2

#Function to read dicom files and return a list of the names of the dicom files
def get_names(path):
    names = []
    for _, _, filenames in os.walk(path):
        for filename in filenames:
            _, ext = os.path.splitext(filename)
            if ext in ['.dcm']:
                names.append(filename)
    return names


#Function to convert dicom files to jpg
def convert_dcm_jpg(cdir, name):
    im = pydicom.dcmread(cdir + '/' + name)
    im = im.pixel_array.astype(float)
    rescale_image = (np.maximum(im, 0)/im.max())*255
    final_image = np.uint8(rescale_image)

    final_image = Image.fromarray(final_image)
    return final_image

"""
Ideal workflow:
Step 1: Define input directory and output directory
Step 2: Utilize the function get_names to get the names of the dicom files, as a list
Step 3: Utilize the function convert_dcm_jpg to convert the dicom files to jpg
Example of Step 3:

 for name in names:
     final_image = convert_dcm_jpg(cdir, name)
     final_image.save(out_dir + '/' + name + '.jpg')

     
"""     