#    Copyright 2021 HIP Applied Computer Vision Lab, Division of Medical Image Computing, German Cancer Research Center
#    (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
import multiprocessing
import re
from multiprocessing import Pool
from typing import Type, Tuple
import sys

import numpy as np
import pandas as pd
from batchgenerators.utilities.file_and_folder_operations import *

from nnunetv2.imageio.base_reader_writer import BaseReaderWriter
from nnunetv2.imageio.reader_writer_registry import determine_reader_writer_from_dataset_json
from nnunetv2.paths import nnUNet_raw
from nnunetv2.utilities.label_handling.label_handling import LabelManager
from nnunetv2.utilities.utils import get_identifiers_from_splitted_dataset_folder, \
    get_filenames_of_train_images_and_targets


def verify_labels(label_file: str, readerclass: Type[BaseReaderWriter], expected_labels: List[int]) -> bool:
    rw = readerclass()
    seg, properties = rw.read_seg(label_file)
    found_labels = np.sort(pd.unique(seg.ravel()))  # np.unique(seg)
    unexpected_labels = [i for i in found_labels if i not in expected_labels]

    # line = "#" * 50

    # print(f"{line}\nrw:{rw}\n\n\nseg:{seg}\n\nproperties:{properties}\n\n\nfound lables:{found_labels}\n\nunexpected:{unexpected_labels}\n{line}")
    


    
    if len(found_labels) == 0 and found_labels[0] == 0:
        print('WARNING: File %s only has label 0 (which should be background). This may be intentional or not, '
              'up to you.' % label_file)
    if len(unexpected_labels) > 0:
        print("Error: Unexpected labels found in file %s.\nExpected: %s\nFound: %s" % (label_file, expected_labels,
                                                                                       found_labels))
        return False
    return True


def check_case_origin(label_file: str, readerclass: type) -> Tuple[str, bool, List[int]]:
    rw = readerclass()
    try:
        # We only need the properties of the segmentation file
        _, properties_seg = rw.read_seg(label_file)

        if 'sitk_stuff' in properties_seg.keys():
            origin_seg = properties_seg['sitk_stuff']['origin']

            if len(origin_seg) != 3:
                return label_file, False, list(origin_seg)
            else:
                return label_file, True, list(origin_seg)
        else:
            # If not using SimpleITKIO, this check is not relevant for this error
            return label_file, True, [] # Assume OK if not sitk_stuff
    except Exception as e:
        # Catch any errors during reading to prevent process crashes
        return label_file, False, f"Error during reading: {e}"

def check_cases(image_files: List[str], label_file: str, expected_num_channels: int,
                readerclass: Type[BaseReaderWriter]) -> bool:

    pid = os.getpid()
    ret = True

    try:
        rw = readerclass()

        images, properties_image = rw.read_images(image_files)
        segmentation, properties_seg = rw.read_seg(label_file)

        # check for nans
        if np.any(np.isnan(images)):
            print(f'Images contain NaN pixel values. You need to fix that by '
                f'replacing NaN values with something that makes sense for your images!\nImages:\n{image_files}')
            ret = False
        if np.any(np.isnan(segmentation)):
            print(f'Segmentation contains NaN pixel values. You need to fix that.\nSegmentation:\n{label_file}')
            ret = False

        # check shapes
        shape_image = images.shape[1:]
        shape_seg = segmentation.shape[1:]
        if shape_image != shape_seg:
            print('Error: Shape mismatch between segmentation and corresponding images. \nShape images: %s. '
                '\nShape seg: %s. \nImage files: %s. \nSeg file: %s\n' %
                (shape_image, shape_seg, image_files, label_file))
            ret = False

        # check spacings
        spacing_images = properties_image['spacing']
        spacing_seg = properties_seg['spacing']
        if not np.allclose(spacing_seg, spacing_images):
            print('Error: Spacing mismatch between segmentation and corresponding images. \nSpacing images: %s. '
                '\nSpacing seg: %s. \nImage files: %s. \nSeg file: %s\n' %
                (spacing_images, spacing_seg, image_files, label_file))
            ret = False

        # check modalities
        if not len(images) == expected_num_channels:
            print('Error: Unexpected number of modalities. \nExpected: %d. \nGot: %d. \nImages: %s\n'
                % (expected_num_channels, len(images), image_files))
            ret = False

        # nibabel checks
        if 'nibabel_stuff' in properties_image.keys():
            # this image was read with NibabelIO
            affine_image = properties_image['nibabel_stuff']['original_affine']
            affine_seg = properties_seg['nibabel_stuff']['original_affine']
            if not np.allclose(affine_image, affine_seg):
                print('WARNING: Affine is not the same for image and seg! \nAffine image: %s \nAffine seg: %s\n'
                    'Image files: %s. \nSeg file: %s.\nThis can be a problem but doesn\'t have to be. Please run '
                    'nnUNetv2_plot_overlay_pngs to verify if everything is OK!\n'
                    % (affine_image, affine_seg, image_files, label_file))

        # sitk checks
        if 'sitk_stuff' in properties_image.keys():


            # this image was read with SimpleITKIO
            # spacing has already been checked, only check direction and origin
            origin_image = properties_image['sitk_stuff']['origin']
            origin_seg = properties_seg['sitk_stuff']['origin']



            # if len(origin_image) != 3:
            #     print(f"[PID {pid}] WARNING: Image origin not 3D. File: {image_files}. Origin: {origin_image} (length {len(origin_image)})", file=sys.stderr)
            # if len(origin_seg) != 3:
            #     print(f"[PID {pid}] ERROR: Segmentation origin not 3D. File: {label_file}. Origin: {origin_seg} (length {len(origin_seg)})", file=sys.stderr)
            #     # Since the original error is due to shape mismatch in np.allclose,
            #     # setting ret to False here correctly flags the issue.
            #     ret = False
            # else: # Only compare if both are 3D to avoid the ValueError
            #     if not np.allclose(origin_image, origin_seg):
            #         print(f'[PID {pid}] WARNING: Origin mismatch between segmentation and corresponding images. Origin images: {origin_image}. Origin seg: {origin_seg}. Image files: {image_files}. Seg file: {label_file}', file=sys.stderr)


            
            # origin_image = np.array(properties_image['sitk_stuff']['origin'])[:3]
            # origin_seg = np.array(properties_seg['sitk_stuff']['origin'])[:3]

            # if len(origin_image) != 3 or len(origin_seg) != 3:
            #     abnormal_files+=1
            #     print(f"Expected origin to be 3D, got: {origin_image} ({len(origin_image)})")
            # print(f"num abnormals:{abnormal_files}")
            # assert len(origin_image) == 3, f"Expected origin to be 3D, got: {origin_image} ({len(origin_image)})"
            # assert len(origin_seg) == 3, f"Expected seg origin to be 3D, got: {origin_seg} ({len(origin_seg)})"

            if not np.allclose(origin_image, origin_seg):
                print('Warning: Origin mismatch between segmentation and corresponding images. \nOrigin images: %s. '
                    '\nOrigin seg: %s. \nImage files: %s. \nSeg file: %s\n' %
                    (origin_image, origin_seg, image_files, label_file))

            
            direction_image = properties_image['sitk_stuff']['direction']
            direction_seg = properties_seg['sitk_stuff']['direction']
            if not np.allclose(direction_image, direction_seg):
                print(f'[PID {pid}] WARNING: Direction mismatch between segmentation and corresponding images. Direction images: {direction_image}. Direction seg: {direction_seg}. Image files: {image_files}. Seg file: {label_file}', file=sys.stderr)

                # print('Warning: Direction mismatch between segmentation and corresponding images. \nDirection images: %s. '
                #     '\nDirection seg: %s. \nImage files: %s. \nSeg file: %s\n' %
                #     (direction_image, direction_seg, image_files, label_file))

    except Exception as e:
        print(f"[PID {pid}] UNCAUGHT EXCEPTION processing {label_file} (image: {image_files}): {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr) # Print full traceback for this specific file
        ret = False # Mark as failed

    return ret


def verify_dataset_integrity(folder: str, num_processes: int = 8) -> None:
    """
    folder needs the imagesTr, imagesTs and labelsTr subfolders. There also needs to be a dataset.json
    checks if the expected number of training cases and labels are present
    for each case, if possible, checks whether the pixel grids are aligned
    checks whether the labels really only contain values they should
    :param folder:
    :return:
    """
    assert isfile(join(folder, "dataset.json")), f"There needs to be a dataset.json file in folder, folder={folder}"
    dataset_json = load_json(join(folder, "dataset.json"))

    if not 'dataset' in dataset_json.keys():
        assert isdir(join(folder, "imagesTr")), f"There needs to be a imagesTr subfolder in folder, folder={folder}"
        assert isdir(join(folder, "labelsTr")), f"There needs to be a labelsTr subfolder in folder, folder={folder}"

    # make sure all required keys are there
    dataset_keys = list(dataset_json.keys())
    required_keys = ['labels', "channel_names", "numTraining", "file_ending"]
    assert all([i in dataset_keys for i in required_keys]), 'not all required keys are present in dataset.json.' \
                                                            '\n\nRequired: \n%s\n\nPresent: \n%s\n\nMissing: ' \
                                                            '\n%s\n\nUnused by nnU-Net:\n%s' % \
                                                            (str(required_keys),
                                                             str(dataset_keys),
                                                             str([i for i in required_keys if i not in dataset_keys]),
                                                             str([i for i in dataset_keys if i not in required_keys]))

    expected_num_training = dataset_json['numTraining']
    num_modalities = len(dataset_json['channel_names'].keys()
                         if 'channel_names' in dataset_json.keys()
                         else dataset_json['modality'].keys())
    file_ending = dataset_json['file_ending']

    dataset = get_filenames_of_train_images_and_targets(folder, dataset_json)

    # check if the right number of training cases is present
    assert len(dataset) == expected_num_training, 'Did not find the expected number of training cases ' \
                                                               '(%d). Found %d instead.\nExamples: %s' % \
                                                               (expected_num_training, len(dataset),
                                                                list(dataset.keys())[:5])

    # check if corresponding labels are present
    if 'dataset' in dataset_json.keys():
        # just check if everything is there
        ok = True
        missing_images = []
        missing_labels = []
        for k in dataset:
            for i in dataset[k]['images']:
                if not isfile(i):
                    missing_images.append(i)
                    ok = False
            if not isfile(dataset[k]['label']):
                missing_labels.append(dataset[k]['label'])
                ok = False
        if not ok:
            raise FileNotFoundError(f"Some expected files were missing. Make sure you are properly referencing them "
                                    f"in the dataset.json. Or use imagesTr & labelsTr folders!\nMissing images:"
                                    f"\n{missing_images}\n\nMissing labels:\n{missing_labels}")
    else:
        # old code that uses imagestr and labelstr folders
        labelfiles = subfiles(join(folder, 'labelsTr'), suffix=file_ending, join=False)
        # because of -seg! +4 -seg --> 4
        label_identifiers = [i[:-len(file_ending)] for i in labelfiles]
        # label_identifiers = [i[:-(len(file_ending)+4)] for i in labelfiles]
        print(f"label_identifiers: {label_identifiers[:5]}")
        labels_present = [i in label_identifiers for i in dataset.keys()]
        missing = [i for j, i in enumerate(dataset.keys()) if not labels_present[j]]
        assert all(labels_present), f'not all training cases have a label file in labelsTr. Fix that. Missing: {missing}'

    labelfiles = [v['label'] for v in dataset.values()]
    image_files = [v['images'] for v in dataset.values()]
    
    # print(f"----\nimagefiles: {image_files}")

    # no plans exist yet, so we can't use PlansManager and gotta roll with the default. It's unlikely to cause
    # problems anyway
    label_manager = LabelManager(dataset_json['labels'], regions_class_order=dataset_json.get('regions_class_order'))
    expected_labels = label_manager.all_labels
    if label_manager.has_ignore_label:
        expected_labels.append(label_manager.ignore_label)
    labels_valid_consecutive = np.ediff1d(expected_labels) == 1
    assert all(
        labels_valid_consecutive), f'Labels must be in consecutive order (0, 1, 2, ...). The labels {np.array(expected_labels)[1:][~labels_valid_consecutive]} do not satisfy this restriction'

    # determine reader/writer class
    # print(f"dataset keys: {dataset.keys()}")
    print(f"****\n{dataset[dataset.keys().__iter__().__next__()]}")
    # print(f"****\n{len(image_files)}, {len(labelfiles)}, {labelfiles}, {image_files}")
    reader_writer_class = determine_reader_writer_from_dataset_json(dataset_json, dataset[dataset.keys().__iter__().__next__()]['images'][0])

    # check whether only the desired labels are present
    with multiprocessing.get_context("spawn").Pool(num_processes) as p:
        result = p.starmap(
            verify_labels,
            zip(labelfiles, [reader_writer_class] * len(labelfiles), [expected_labels] * len(labelfiles))
        )
        if not all(result):
            raise RuntimeError(
                'Some segmentation images contained unexpected labels. Please check text output above to see which one(s).')

        
        # check whether shapes and spacings match between images and labels
        result = p.starmap(
            check_cases,
            zip(image_files, labelfiles, [num_modalities] * expected_num_training,
                [reader_writer_class] * expected_num_training)
        )
        if not all(result):
            raise RuntimeError(
                'Some images have errors. Please check text output above to see which one(s) and what\'s going on.')


    # check for nans
    # check all same orientation nibabel
    print('\n####################')
    print('verify_dataset_integrity Done. \nIf you didn\'t see any error messages then your dataset is most likely OK!')
    print('####################\n')


if __name__ == "__main__":
    # investigate geometry issues
    example_folder = join(nnUNet_raw, 'Dataset250_COMPUTING_it0')
    num_processes = 6
    verify_dataset_integrity(example_folder, num_processes)
