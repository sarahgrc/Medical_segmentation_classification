
import pydicom as dicom
import shutil
import dicom2nifti
import numpy as np
import torch
from scipy.ndimage import morphology

def dicom_2_tensor(dicom_folder : str, length:str)-> torch.tensor:
    """

    :param dicom_folder: folder where dicom files of 1 person are stored
    :param length: number of cuts to save from dicom
    :return: torch tensor of size lxwx legth
    """
    return None

def equalise_hist(img):
    return None

