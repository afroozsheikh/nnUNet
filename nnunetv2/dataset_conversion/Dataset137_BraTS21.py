import multiprocessing
import shutil
from multiprocessing import Pool

import SimpleITK as sitk
import numpy as np
from batchgenerators.utilities.file_and_folder_operations import *
from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
from nnunetv2.paths import nnUNet_raw


def copy_BraTS_segmentation_and_convert_labels_to_nnUNet(in_file: str, out_file: str) -> None:
    # use this for segmentation only!!!
    # nnUNet wants the labels to be continuous. BraTS is 0, 1, 2, 4 -> we make that into 0, 1, 2, 3
    img = sitk.ReadImage(in_file)
    img_npy = sitk.GetArrayFromImage(img)

    uniques = np.unique(img_npy)
    for u in uniques:
        if u not in [0, 1, 2, 3]:
            raise RuntimeError('unexpected label')

    seg_new = np.zeros_like(img_npy)
    # seg_new[img_npy == 4] = 3
    # seg_new[img_npy == 2] = 1
    # seg_new[img_npy == 1] = 2

    '''
    GD-enhancing tumor (ET — label 3)
    edematous/invaded tissue (ED — label 2)
    necrotic tumor core (NCR — label 1)

    The WT describes the complete extent of the disease, as it entails the TC (1, 3) and the peritumoral edematous/invaded tissue (ED) -->2
    The TC entails the ET (3), as well as the necrotic (NCR) (1) parts of the tumor.
    
    'background': 0,
    'whole tumor': (1, 2, 3),
    'tumor core': (1, 3),
    'enhancing tumor': (3, )

    '''
    seg_new[img_npy == 3] = 3
    seg_new[img_npy == 2] = 2
    seg_new[img_npy == 1] = 1

    img_corr = sitk.GetImageFromArray(seg_new)
    img_corr.CopyInformation(img)
    sitk.WriteImage(img_corr, out_file)


def convert_labels_back_to_BraTS(seg: np.ndarray):
    new_seg = np.zeros_like(seg)
    # new_seg[seg == 1] = 2
    # new_seg[seg == 3] = 4
    # new_seg[seg == 2] = 1
    new_seg[seg == 1] = 1
    new_seg[seg == 3] = 3
    new_seg[seg == 2] = 2

    return new_seg


def load_convert_labels_back_to_BraTS(filename, input_folder, output_folder):
    a = sitk.ReadImage(join(input_folder, filename))
    b = sitk.GetArrayFromImage(a)
    c = convert_labels_back_to_BraTS(b)
    d = sitk.GetImageFromArray(c)
    d.CopyInformation(a)
    sitk.WriteImage(d, join(output_folder, filename))


def convert_folder_with_preds_back_to_BraTS_labeling_convention(input_folder: str, output_folder: str, num_processes: int = 12):
    """
    reads all prediction files (nifti) in the input folder, converts the labels back to BraTS convention and saves the
    """
    maybe_mkdir_p(output_folder)
    nii = subfiles(input_folder, suffix='.nii.gz', join=False)
    with multiprocessing.get_context("spawn").Pool(num_processes) as p:
        p.starmap(load_convert_labels_back_to_BraTS, zip(nii, [input_folder] * len(nii), [output_folder] * len(nii)))


if __name__ == '__main__':
    # brats_data_dir = '/home/isensee/drives/E132-Rohdaten/BraTS_2021/training'
    brats_data_dir = '/mnt/mydrive/Datasets/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData'

    task_id = 1
    task_name = "BraTS2023"

    foldername = "Dataset%03.0d_%s" % (task_id, task_name)

    # setting up nnU-Net folders
    out_base = join(nnUNet_raw, foldername)
    imagestr = join(out_base, "imagesTr")
    labelstr = join(out_base, "labelsTr")
    maybe_mkdir_p(imagestr)
    maybe_mkdir_p(labelstr)

    case_ids = subdirs(brats_data_dir, prefix='BraTS', join=False)

    for c in case_ids:
        shutil.copy(join(brats_data_dir, c, c + "-t1n.nii.gz"), join(imagestr, c + '_0000.nii.gz'))
        shutil.copy(join(brats_data_dir, c, c + "-t1c.nii.gz"), join(imagestr, c + '_0001.nii.gz'))
        shutil.copy(join(brats_data_dir, c, c + "-t2w.nii.gz"), join(imagestr, c + '_0002.nii.gz'))
        shutil.copy(join(brats_data_dir, c, c + "-t2f.nii.gz"), join(imagestr, c + '_0003.nii.gz'))


        copy_BraTS_segmentation_and_convert_labels_to_nnUNet(join(brats_data_dir, c, c + "-seg.nii.gz"),
                                                             join(labelstr, c + '.nii.gz'))

    '''
    All BraTS mpMRI scans are available as NIfTI files (.nii.gz) and describe 
    a) native (T1)  
    b) post-contrast T1-weighted (T1Gd), 
    c) T2-weighted (T2), 
    d) T2 Fluid Attenuated Inversion Recovery (T2-FLAIR) volumes
    '''
    generate_dataset_json(out_base,
                          channel_names={0: 'T1', 1: 'T1ce', 2: 'T2', 3: 'Flair'},
                          labels={
                              'background': 0,
                              'whole tumor': (1, 2, 3),
                            #   'tumor core': (2, 3),
                            'tumor core': (1, 3),
                              'enhancing tumor': (3, )
                          },
                          num_training_cases=len(case_ids),
                          file_ending='.nii.gz',
                        #   regions_class_order=(1, 2, 3),
                            regions_class_order=(2, 1, 3),
                          license='see https://www.synapse.org/#!Synapse:syn25829067/wiki/610863',
                          reference='see https://www.synapse.org/#!Synapse:syn25829067/wiki/610863',
                          dataset_release='1.0')
