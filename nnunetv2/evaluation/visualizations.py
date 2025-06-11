import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from glob import glob
import os
from tqdm import tqdm
from scipy.ndimage.measurements import label # DeprecationWarning will be shown for this
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.backends.backend_pdf import PdfPages
import sys
import json


def key_to_label_or_region(key: str):
    try:
        return int(key)
    except ValueError:
        key = key.replace('(', '')
        key = key.replace(')', '')
        split = key.split(',')
        return tuple([int(i) for i in split if len(i) > 0])


def load_summary_json_for_visualization(filename: str):
    with open(filename, 'r') as f:
        results = json.load(f)

    if 'metric_per_case' in results:
        for i in range(len(results["metric_per_case"])):
            if 'metrics' in results["metric_per_case"][i]:
                results["metric_per_case"][i]['metrics'] = \
                    {key_to_label_or_region(k): results["metric_per_case"][i]['metrics'][k]
                     for k in results["metric_per_case"][i]['metrics'].keys()}
    return results





def generate_visualization_pdf(
    pdf_filename: str,
    predictions_folder: str,
    labels_folder: str,
    images_folder: str,
    sample_mean_dice_scores: dict,
    samples_all_metrics_per_case: dict, 
    sort_by_dice: bool,
    slice_index: int = 75
):

   
    pdf_pages = PdfPages(pdf_filename)

    image_titles = ['T1', 'T1ce', 'T2', 'FLAIR', 'Label (GT)', 'Prediction']

    if sort_by_dice:
        sorted_cases_items = sorted(sample_mean_dice_scores.items(), key=lambda item: item[1])

    else:
        sorted_cases_items = sorted(sample_mean_dice_scores.items())

    print(f"Total cases to visualize: {len(sorted_cases_items)}")
    metrics_order = ['Dice', 'FN', 'FP', 'IoU', 'TN', 'TP', 'n_pred', 'n_ref']
    regions_for_table = {
        (1, 2, 3): "Whole Tumor (1,2,3)",
        (1, 3): "Tumor Core (1,3)",
        (3,): "Enhancing Tumor (3)"
    }

    for fname, mean_value in tqdm(sorted_cases_items[:10], desc="Generating PDF pages"):
        mean_dice = sample_mean_dice_scores[fname]
        all_metrics_this_case = samples_all_metrics_per_case[fname]
        

        #--------------------------------------------------------------------------------------------------------#
        modalities_slices = []
        for modality_num in range(4):
            img_path = os.path.join(images_folder, f"{fname}_000{modality_num}.nii.gz")    
            try:
                img_data = nib.load(img_path).get_fdata().astype(np.float32)
                modalities_slices.append(img_data[:, :, slice_index])
                # modalities_slices = [img_data[:, :, slice_index, i] for i in range(4)]

            except Exception as e:
                print(f"Error loading or processing image {img_path}: {e}. Skipping visualization.")
                continue
        

        #--------------------------------------------------------------------------------------------------------#
        gt_path = os.path.join(labels_folder, f"{fname}.nii.gz")
        if not os.path.exists(gt_path):
            print(f"Warning: Ground truth label not found for {fname} at {gt_path}. Skipping visualization.")
            continue
        try:
            label_gt_data = nib.load(gt_path).get_fdata().astype(np.float32)
            gt_slice_data = label_gt_data[:, :, slice_index]
        except Exception as e:
            print(f"Error loading or processing ground truth {gt_path}: {e}. Skipping visualization.")
            continue

        #--------------------------------------------------------------------------------------------------------#
        pred_nii_path = os.path.join(predictions_folder, f"{fname}.nii.gz")
        if not os.path.exists(pred_nii_path):
            print(f"Warning: Final NIfTI prediction not found for {fname} at {pred_nii_path}. Skipping visualization.")
            continue
        try:
            pred_data = nib.load(pred_nii_path).get_fdata().astype(np.uint8)
            pred_slice_data = pred_data[:, :, slice_index]
        except Exception as e:
            print(f"Error loading or processing prediction {pred_nii_path}: {e}. Skipping visualization.")
            continue

        #--------------------------------------------------------------------------------------------------------#
        imgs_to_display = modalities_slices + [gt_slice_data] + [pred_slice_data]

        label_cmap = ListedColormap(['lightgray', '#DC143C', '#FFD700', '#00BFFF'])
        label_colors = label_cmap.colors
        legend_elements = [
            Patch(facecolor=label_colors[1], label='Enhancing Tumor'),
            Patch(facecolor=label_colors[2], label='Tumor Core'),
            Patch(facecolor=label_colors[3], label='Whole Tumor')
        ]
        fig = plt.figure(figsize=(24, 10)) 
        gs = fig.add_gridspec(2, 6, height_ratios=[4, 1])
        ax_images = [fig.add_subplot(gs[0, i]) for i in range(6)]
        ax_table = fig.add_subplot(gs[1, :])

        for j in range(6):
            ax_images[j].imshow(imgs_to_display[j], cmap=label_cmap if j >= 4 else 'gray')
            ax_images[j].axis('off')
            ax_images[j].set_title(image_titles[j], fontsize=10)
        
        ax_images[-1].legend(handles=legend_elements, loc='lower right', fontsize=8, frameon=True)
    

        #-------------------------------      Prepare and Plot Metrics Table    ---------------------------------------------#
        table_data = []
        metric_dict = {metric_name: [] for metric_name in metrics_order}

        for region_key, display_name in regions_for_table.items():
            row_data = []
            metrics_for_region = all_metrics_this_case.get(region_key, {}) 
            for metric_name in metrics_order:
                value = metrics_for_region.get(metric_name)
                formatted_value = f"{value:.4f}" if isinstance(value, (int, float)) and not np.isnan(value) else "-"
                metric_dict[metric_name].append(float(formatted_value))
                row_data.append(formatted_value)
                
            table_data.append(row_data)
    
        mean_row_data = []
        for metric_name in metrics_order:
            values = [x for x in metric_dict[metric_name] if isinstance(x, (int, float, np.number))]
            mean_row_data.append(f"{np.mean(values):.4f}" if values else '-------')

        table_data.append(mean_row_data)

        table_rows_labels = list(regions_for_table.values()) + ["Mean Values"]
        
        table = ax_table.table(cellText=table_data,
                                rowLabels=table_rows_labels,
                                colLabels=metrics_order,
                                loc='center',
                                cellLoc='center',
                                # bbox=[0, 0, 1, 1]
                                bbox=[0.2, 0.25, 0.6, 0.5]
                                ) 

        table.auto_set_font_size(False)
        table.set_fontsize(12)
        # table.scale(1.2, 1.2) 
        table.scale(2, 2)   

        ax_table.axis('off') 

        prediction_file_basename = os.path.basename(pred_nii_path)

        fig_title_text = f"Case: {fname} | Mean Dice: {mean_dice:.4f} | Mean IoU: {np.mean(metric_dict['IoU']):.4f}"
        fig.suptitle(fig_title_text, fontsize=14, y=0.88) 

        plt.tight_layout()
        plt.subplots_adjust(top=0.93, bottom=0.05, hspace=0.1) 

        pdf_pages.savefig(fig)
        plt.close(fig)


    pdf_pages.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate a PDF visualization report of BraTS2023 segmentation results.")
    parser.add_argument('--pred_folder', type=str,
                        help='Path to the folder containing nnUNet predicted .nii.gz segmentations',
                        default='/mnt/mydrive/Results/Dataset001_BraTS2023/nnUNetTrainer__nnUNetPlans__2d/fold_0/validation/')
    parser.add_argument('--image_folder', type=str,
                        help='Path to the original BraTS image modalities (e.g., /path/to/BraTS2023/images/). '
                             'Assumes structure like IMAGES_PATH/case_name.nii.gz where each .nii.gz contains 4 modalities.')
    parser.add_argument('--gt_folder', type=str,
                        help='Path to the ground truth label folder (e.g., /mnt/mydrive/Datasets/Dataset001_BraTS2023/labelsTr/). '
                             'Assumes structure like GT_PATH/case_name.nii.gz.')
    parser.add_argument('--summary_json_path', type=str,
                        help='Path to the summary.json file generated by evaluate_predictions.py.')
    parser.add_argument('--output_file_name', type=str,
                        help='Name for the output PDF file (e.g., results_report.pdf)')
    parser.add_argument('--sorted', action='store_true',
                        help='Sort the cases in the PDF by their mean Dice score (ascending).')
    parser.add_argument('--slice_index', type=int, default=75,
                        help='Axial slice index to visualize for all cases (default: 75)')

    args = parser.parse_args()

    print(f"Loading summary from {args.summary_json_path}...")
    try:
        summary_data_raw = load_summary_json_for_visualization(args.summary_json_path)
        print(f"Summary of first case:\n{summary_data_raw['metric_per_case'][0]}")
    except FileNotFoundError:
        print(f"Error: summary.json not found at {args.summary_json_path}.")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading or parsing summary.json: {e}")
        sys.exit(1)

    sample_mean_dice_scores = {}
    samples_all_metrics_per_case = {} 

    brats_regions_for_mean_dice = [(1, 2, 3), (1, 3), (3,)]

    for case_data in summary_data_raw['metric_per_case']:
        pred_file = case_data['prediction_file']
        case_name = os.path.basename(pred_file).replace(".nii.gz", "")
        print(f"{'-'*20} Case: {case_name} {'-'*20}")

        metrics = case_data['metrics'] # These keys are already converted to tuples

        dice_scores_for_case = []
        for region_key in brats_regions_for_mean_dice:
            if region_key in metrics and 'Dice' in metrics[region_key]:
                dice_scores_for_case.append(metrics[region_key]['Dice'])
        
        print(f"Dice scores per label: {dice_scores_for_case}")
        mean_dice_this_case = np.nanmean([d for d in dice_scores_for_case if not np.isnan(d)])
        
        sample_mean_dice_scores[case_name] = mean_dice_this_case
        samples_all_metrics_per_case[case_name] = metrics # Store the full metrics dictionary for the table


    print(f"\nGenerating visualization PDF: {args.output_file_name}")
    generate_visualization_pdf(
        args.output_file_name,
        args.pred_folder,
        args.gt_folder,
        args.image_folder,
        sample_mean_dice_scores,
        samples_all_metrics_per_case,
        args.sorted,
        args.slice_index
    )
    print("Visualization PDF generation complete.")
