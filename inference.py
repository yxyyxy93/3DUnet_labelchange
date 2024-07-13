import os
from utils_func.Read_CSV import read_csv_to_3d_array, save_3d_array_to_csv

# Import all functions from test.py
from test import (load_checkpoint, load_test_dataset, evaluate_model, reassemble_chunks,
                  SimpleCSVLoader, process_data, process_ultrasound_data, calculate_precision_recall,
                  plot_precision_recall_curve, add_or_remove_ones, save_2d_array_as_image)

# Set mode for testing
os.environ['MODE'] = 'test'

import config
import re


if __name__ == "__main__":
    # Initialize model
    import numpy as np

    import model_unet3d
    from utils_func import criteria

    import config

    # # ------------------------------------------------
    # Parameters
    fold_number = 1
    model_filename = "d_best.pth.tar"
    modified_results_dir = config.results_dir[8:]
    process_from_start = True  # User-defined flag to choose processing mode
    # Load and preprocess test data
    testdata = SimpleCSVLoader(config.test_data_path)
    # Use regex to extract the "_1" part
    match = re.search(r'_(\d+)\.csv', config.label_exp_dir)
    if match:
        extracted_part = match.group(1)
        extracted_part = f'_{extracted_part}'
        print(f"Extracted part: {extracted_part}")
    else:
        print("No match found")
    save_path = f"D:\python_work\WovenComposite_defects\3dUnet_ultrasound_defect_LabelChange_depthchannel\dataset" \
                f"\inference\infer{extracted_part}"

    testdata.load_and_preprocess()
    segment_data, original_size = testdata.segment_dataset(chunk_size=(17, 17), step=config.step)
    # Function call
    reassembled_data = process_ultrasound_data(fold_number=fold_number,
                                               model_filename=model_filename,
                                               segment_data=segment_data,
                                               original_size=original_size,
                                               save_path="",
                                               process_from_start=process_from_start,
                                               step=config.step)

    # work on the sections
    testdata = SimpleCSVLoader("/mnt/raid5/xiaoyu/Ultrasound_data/dataset_woven_["
                               "#090]8_0-1defect/test/x0_60_y0_60_090_2.csv")
    testdata.load_and_preprocess()
    segment_data, original_size = testdata.segment_dataset(chunk_size=(17, 17), step=1)
    process_ultrasound_data(fold_number=fold_number,
                            model_filename=model_filename,
                            segment_data=segment_data,
                            original_size=original_size,
                            save_path=f"/mnt/raid5/xiaoyu/Ultrasound_data/dataset_woven_["
                                      f"#090]8_0-1defect/test/x0_60_y0_60_090_2_{modified_results_dir}.csv",
                            process_from_start=process_from_start,
                            step=config.step)

    testdata = SimpleCSVLoader("/mnt/raid5/xiaoyu/Ultrasound_data/dataset_woven_["
                               "#090]8_0-1defect/test/x0_120_y0_120_090_2.csv")
    testdata.load_and_preprocess()
    segment_data, original_size = testdata.segment_dataset(chunk_size=(17, 17), step=1)
    process_ultrasound_data(fold_number=fold_number,
                            model_filename=model_filename,
                            segment_data=segment_data,
                            original_size=original_size,
                            save_path=f"/mnt/raid5/xiaoyu/Ultrasound_data/dataset_woven_["
                                      f"#090]8_0-1defect/test/x0_120_y0_120_090_2_{modified_results_dir}.csv",
                            process_from_start=process_from_start,
                            step=config.step)
    # ------------------------------------------------
    # Read the label data
    label = read_csv_to_3d_array(config.label_exp_dir)
    # label = add_or_remove_ones(label, num_ones_to_add=3)
    label = np.max(label, axis=2)
    # Call the function
    save_2d_array_as_image(label, f"label{extracted_part}")

    # ---------------------------------
    # # Define the criterion and validation criterion
    # criterion = getattr(criteria, config.loss_function)(smooth=1e4).to(config.device)
    # val_crite = getattr(criteria, config.val_function)().to(config.device)

    # loss_DL = criterion(torch.tensor(DL_output, dtype=torch.float32).to(config.device), label_torch)
    # score_DL = val_crite(torch.tensor(DL_output, dtype=torch.float32).to(config.device), label_torch)
    # print(f"Loss of DL: {loss_DL}, Score of DL: {score_DL}")

    # ## Convert to tensors
    # # label_tensor_2d = torch.tensor(label.max(axis=2), dtype=torch.float32).to(config.device)

    # # Example usage
    # zeros_output_tensor = torch.zeros_like(AST_output_tensor)
    # ones_output_tensor = torch.ones_like(AST_output_tensor)
    # loss_zeros = criterion(zeros_output_tensor, label_torch)
    # score_zeros = val_crite(zeros_output_tensor, label_torch)
    # loss_ones = criterion(ones_output_tensor, label_torch)
    # score_ones = val_crite(ones_output_tensor, label_torch)
    # print(f"Loss of all-zero matrix: {loss_zeros}, Score of all-zero matrix: {score_zeros}")
    # print(f"Loss of all-one matrix: {loss_ones}, Score of all-one matrix: {score_ones}")

    DL_output = np.max(DL_output, axis=2)
    save_2d_array_as_image(DL_output, f"DL_pro{extracted_part}")
    precision, recall, thresholds, f1_scores, average_precision = calculate_precision_recall(
        DL_output.flatten(),
        label.flatten(),
        n_interp_points=100)
    print(f"Average Precision: {average_precision:.4f}")
    # Plot the Precision-Recall curve
    plot_precision_recall_curve(precision, recall, average_precision, title=f"PR_curve_DL{extracted_part}")
