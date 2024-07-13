import os
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim.swa_utils import AveragedModel

import dataset
from dataset import imgproc
from utils_func.Read_CSV import read_csv_to_3d_array, save_3d_array_to_csv
from utils_func.criteria import SSIM3D  # Assuming SSIM3D is defined in utils_func.criteria

# Set mode for testing
os.environ['MODE'] = 'test'

import config
import matplotlib.pyplot as plt
import re


def load_checkpoint(checkpoint_path, model, ema_model=None, optimizer=None, scheduler=None):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["state_dict"])
    if ema_model:
        ema_model.load_state_dict(checkpoint["ema_state_dict"])
    if optimizer:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler:
        scheduler.load_state_dict(checkpoint["scheduler"])
    return model, ema_model, checkpoint


def load_test_dataset():
    # "\"Load and prepare the test dataset
    test_dataset = dataset.TestDataset(config.image_dir,
                                       config.label_dir,
                                       config.option_type,
                                       config.dilation_factors)  # Adjust as per your dataset class
    test_loader = DataLoader(test_dataset, batch_size=1,
                             shuffle=False)  # Adjust batch_size and other parameters as needed
    return test_loader


def evaluate_model(test_loader, model_eval, device):
    # "\"\"Evaluate the model on the test dataset.\"\"\"
    model_eval.eval()
    ssim_model = SSIM3D().to(device)  # SSIM model for evaluation
    test_ssim_scores = []
    with torch.no_grad():
        for data in test_loader:
            gt = data["gt"].to(device=config.device, non_blocking=True)
            lr = data["lr"].to(device=config.device, non_blocking=True)
            outputs = model_eval(lr)
            ssim_score = ssim_model(outputs, gt)  # Assuming ground truth is inputs
            test_ssim_scores.append(ssim_score.item())
    return test_ssim_scores


def reassemble_chunks(chunks: list, original_size: tuple = (256, 235, 300),
                      chunk_size: tuple = (16, 16, 256), step: int = 1) -> np.ndarray:
    """
    Reassembles the segmented chunks back to the original size by averaging intersecting predictions,
    with a consistent step size for both segmentation and reassembly.
    """
    reassembled_data = np.zeros(original_size)
    count_matrix = np.zeros(original_size)  # To count the number of predictions at each point
    # Calculate strides for indexing chunks considering the step size
    stride_y = ((original_size[1] - chunk_size[0]) // step) + 1
    stride_x = ((original_size[2] - chunk_size[1]) // step) + 1
    for i in range(0, original_size[1] - chunk_size[0] + 1, step):
        for j in range(0, original_size[2] - chunk_size[1] + 1, step):
            # Adjust chunk index calculation for the step size
            chunk_idx = (i // step) * stride_x + (j // step)
            chunk = chunks[chunk_idx].squeeze()
            reassembled_data[:, i:i + chunk_size[0], j:j + chunk_size[1]] += chunk
            count_matrix[:, i:i + chunk_size[0], j:j + chunk_size[1]] += 1

    # Normalize reassembled_data by count_matrix, safely handling zeros
    count_matrix_with_no_zeros = np.where(count_matrix == 0, 1, count_matrix)
    normalized_reassembled_data = reassembled_data / count_matrix_with_no_zeros

    return normalized_reassembled_data


class SimpleCSVLoader:
    """
    Simple class for reading and preprocessing .csv files.

    Args:
        dataset_dir (str): Directory containing the dataset files (.csv).
    """

    def __init__(self, dataset_dir: str) -> None:
        self.data = None
        self.dataset_dir = dataset_dir

    def load_and_preprocess(self) -> np.ndarray:
        # Load the image data from the .csv file
        image_data = read_csv_to_3d_array(self.dataset_dir)
        image_data = np.transpose(image_data, (2, 0, 1))

        # Normalize the image data
        image_normalized = imgproc.normalize(image_data)

        # Assuming image_noisy has shape [depth, height, width]
        depth, height, width = image_data.shape
        # Initialize an array of zeros with the same shape as image_noisy
        depth_channel = np.zeros_like(image_data, dtype=int)
        # Fill each depth slice with its respective depth index
        for d in range(depth):
            depth_channel[d, :, :] = d
        depth_channel = imgproc.normalize(depth_channel)
        image_noisy_with_depth = np.stack([image_normalized, depth_channel], axis=0)

        self.data = image_noisy_with_depth

        return image_noisy_with_depth

    def segment_dataset(self, chunk_size: tuple = (32, 32, 256), step: int = 1) -> tuple[list[Any], list[Any]]:
        """
            Segments the dataset into smaller chunks with a customizable step.

            Args:
            chunk_size (tuple): Size of each chunk.
            step (int): Step size for sliding the window over the dataset.

            Returns:
            list: A list of segmented chunks, each as a numpy array.
        """
        # Assuming self.data is a 4D array with dimensions corresponding to (batch, depth, height, width)
        depth, height, width = self.data.shape[1], self.data.shape[2], self.data.shape[3]
        segmented_data = []
        for i in range(0, height - chunk_size[0] + 1, step):
            for j in range(0, width - chunk_size[1] + 1, step):
                chunk = self.data[:, :, i:i + chunk_size[0], j:j + chunk_size[1]]
                segmented_data.append(chunk)
        # print(f"Segmented into {len(segmented_data)} chunks, each of size {chunk.shape}, with a step of {step}.")

        original_size = [depth, height, width]
        return segmented_data, original_size


def process_data(model, segment_data, batch_size, device):
    segment_output = []
    batch_segments = []

    for i, segment in enumerate(segment_data):
        segment_tensor = torch.tensor(segment, dtype=torch.float).to(device)
        batch_segments.append(segment_tensor.unsqueeze(0))  # Add batch dimension
        if len(batch_segments) == batch_size or i == len(segment_data) - 1:
            batch_tensor = torch.cat(batch_segments, dim=0)

            batch_output = model(batch_tensor)
            batch_output = batch_output.detach().cpu().numpy()
            segment_output.extend(batch_output)
            batch_segments = []

    #     if i % batch_size == 0:
    #         print(f"Processed {i / batch_size} / {len(segment_data) / batch_size} segments")
    #
    # print("Processing complete.")
    return segment_output


def process_ultrasound_data(fold_number=1,
                            model_filename="d_best.pth.tar",
                            segment_data=None,
                            original_size=None,
                            save_path="/mnt/raid5/xiaoyu/Ultrasound_data/dataset_woven_["
                                      "#090]8_0-1defect/test/exp_test_results.csv",
                            process_from_start=True,
                            step=1):
    # Set mode for testing
    os.environ['MODE'] = 'test'
    if process_from_start:
        # Initialize and load the model
        # show_dataset_info(train_prefetcher, show_sample_slices=False)
        convLSTMmodel = model_unet3d.__dict__[config.d_arch_name](in_channels=config.input_dim,
                                                                  num_classes=config.output_dim)
        convLSTMmodel = convLSTMmodel.to(device=config.device)
        # Create an Exponential Moving Average Model
        ema_avg = lambda averaged_model_parameter, model_parameter, num_averaged: \
            (1 - config.model_ema_decay) * averaged_model_parameter + config.model_ema_decay * model_parameter
        ema_model = AveragedModel(convLSTMmodel, avg_fn=ema_avg)
        ema_model = ema_model.to(device=config.device)
        print(f"Build `{config.d_arch_name}` model successfully.")

        # Load the EMA model
        checkpoint_path = os.path.join(config.results_dir, f"_fold {fold_number}", model_filename)
        convLSTMmodel, ema_model, _ = load_checkpoint(checkpoint_path, convLSTMmodel, ema_model)
        # Process data
        segment_output = process_data(ema_model, segment_data, config.batch_size, config.device)
        # Save output to npz
        np.savez("temp_list", *segment_output)
    else:
        # Load the arrays from the .npz file
        loaded_data = np.load('temp_list.npz')
        segment_output = [loaded_data[key] for key in loaded_data]
    # Reassemble and save the data
    reassembled_data = reassemble_chunks(segment_output, original_size=original_size, chunk_size=(17, 17, 256),
                                         step=step)
    # Assuming original data was in (height, width, depth), revert the reassembled data to this order
    reassembled_data = np.transpose(reassembled_data, (1, 2, 0))
    save_3d_array_to_csv(reassembled_data, save_path)

    return reassembled_data


def calculate_precision_recall(y_score, y_true, n_interp_points=200):
    """
    Calculate the Precision-Recall curve and F1 score.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True binary labels. Must be either 0 or 1.

    y_score : array-like of shape (n_samples,)
        Target scores, can either be probability estimates of the positive
        class, confidence values, or non-thresholded measure of decisions.

    n_interp_points : int, default=200
        Number of interpolated points between existing thresholds.

    Returns
    -------
    precision : ndarray of shape (>2,)
        Precision values for the corresponding thresholds.

    recall : ndarray of shape (>2,)
        Recall values for the corresponding thresholds.

    thresholds : ndarray of shape (n_thresholds,)
        Decreasing thresholds.

    f1_scores : ndarray of shape (>2,)
        F1 scores for the corresponding thresholds.

    average_precision : float
        Average precision score.
    """
    # Ensure inputs are numpy arrays
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    # Define thresholds
    thresholds = np.linspace(1, 0, n_interp_points)
    precision = []
    recall = []
    f1_scores = []

    # Calculate precision, recall, and F1 score for each threshold
    for thresh in thresholds:
        tp = np.sum((y_score >= thresh) & (y_true == 1))
        fp = np.sum((y_score >= thresh) & (y_true == 0))
        fn = np.sum((y_score < thresh) & (y_true == 1))

        prec = tp / (tp + fp) if tp + fp > 0 else 0
        rec = tp / (tp + fn) if tp + fn > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0

        precision.append(prec)
        recall.append(rec)
        f1_scores.append(f1)

    precision = np.array(precision)
    recall = np.array(recall)
    f1_scores = np.array(f1_scores)

    # Calculate average precision score manually using the trapezoidal rule
    average_precision = 0.0
    for i in range(1, len(precision)):
        average_precision += (recall[i] - recall[i - 1]) * (precision[i] + precision[i - 1]) / 2

    return precision, recall, thresholds, f1_scores, average_precision


def plot_precision_recall_curve(precision, recall, average_precision, title="Precision-Recall Curve"):
    # Create a valid filename from the title
    valid_filename = re.sub(r'[^\w\s-]', '', title).strip().replace(' ', '_') + ".jpg"

    plt.figure()
    plt.plot(recall, precision, color='darkorange', lw=2,
             label=f'Precision-Recall curve (AP = {average_precision:.2f})')
    plt.xlabel('Recall', fontsize=14)
    plt.ylabel('Precision', fontsize=14)
    plt.legend(loc="lower left", fontsize=14)
    plt.title(title, fontsize=16)
    plt.savefig(valid_filename, format='jpg')
    plt.show()


def add_or_remove_ones(label, num_ones_to_add=1):
    # Iterate over each 2D slice along the third dimension
    for i in range(label.shape[0]):
        for j in range(label.shape[1]):
            # Find the indices of the last `1` in the third dimension
            ones_indices = np.where(label[i, j, :] == 1)[0]
            if ones_indices.size > 0:
                last_one_index = ones_indices[-1]
                if num_ones_to_add > 0:
                    # Determine the range to add new ones
                    add_range = min(label.shape[2] - last_one_index - 1, num_ones_to_add)
                    label[i, j, last_one_index + 1: last_one_index + 1 + add_range] = 1
                elif num_ones_to_add < 0:
                    # Determine the range to remove ones
                    remove_range = min(last_one_index + 1, -num_ones_to_add)
                    label[i, j, last_one_index - remove_range + 1: last_one_index + 1] = 0
    return label


def save_2d_array_as_image(testdata_normalized, save_name):
    """
    Saves a 2D array as an image with a colorbar.
    Parameters
    ----------
    testdata_normalized : np.ndarray
        The 2D array to be saved as an image.

    save_name : str
        Name of the file to save the image as.
    """
    # Ensure the save name ends with .png
    if not save_name.lower().endswith('.png'):
        save_name += '.png'

    # Save the 2D array as an image with colorbar
    plt.figure(figsize=(8, 6))
    plt.imshow(testdata_normalized, aspect='auto', cmap='jet')
    plt.colorbar()
    plt.title('2D Array Image with Colorbar')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.savefig(save_name, format='png')
    plt.show()


if __name__ == "__main__":
    # Initialize model
    import numpy as np
    from scipy.ndimage import binary_dilation

    import model_unet3d
    import config

    # # ------------------------------------------------
    # Parameters
    fold_number = 1
    model_filename = "d_best.pth.tar"
    modified_results_dir = config.results_dir[8:]
    process_from_start = True  # User-defined flag to choose processing mode

    # Load and preprocess test data
    testdata = SimpleCSVLoader(config.test_data_path)
    DL_dir = "D:\\python_work\\WovenComposite_defects\\3dUnet_ultrasound_defect_LabelChange_depthchannel\\dataset\\test" \
             "\\Inst_amplitude_090_4_unequal_UNet3D_TverskyLoss_1_20000_1_32_10_2024-07-11.csv"

    # Use regex to extract the "_1" part
    match = re.search(r'_(\d+)\.csv', config.label_exp_dir)
    if match:
        extracted_part = match.group(1)
        extracted_part = f'_{extracted_part}'
        print(f"Extracted part: {extracted_part}")
    else:
        print("No match found")

    testdata.load_and_preprocess()

    # Read the label data
    label = read_csv_to_3d_array(config.label_exp_dir)
    # label = add_or_remove_ones(label, num_ones_to_add=3)
    label = np.max(label, axis=2)
    # Call the function
    save_2d_array_as_image(label, f"label{extracted_part}")

    # # for AST method
    # Compute mean and std along the 1st and 2nd dimensions
    ori_data = testdata.data[0, :]
    ori_data = np.transpose(ori_data, (1, 2, 0))
    mean_along_3rd = np.mean(ori_data, axis=(0, 1))
    std_along_3rd = np.std(ori_data, axis=(0, 1))
    # Create a 1D array as mean + std
    mean_plus_std = mean_along_3rd + std_along_3rd
    # Initialize the testdata_normalized array
    testdata_normalized = np.zeros_like(ori_data)
    # Divide each slice of the label by mean_plus_std along the 3rd dimension using broadcasting
    for i in range(ori_data.shape[0]):
        for j in range(ori_data.shape[1]):
            testdata_normalized[i, j, :] = ori_data[i, j, :] / mean_plus_std

    # Normalize the result
    min_val = testdata_normalized.min()
    max_val = testdata_normalized.max()
    testdata_normalized = (testdata_normalized - min_val) / (max_val - min_val)
    testdata_normalized = np.max(testdata_normalized, axis=2)
    save_2d_array_as_image(testdata_normalized, f"AST_pro{extracted_part}")
    # Compute the Precision-Recall curve and F1 score
    precision, recall, thresholds, f1_scores, average_precision = calculate_precision_recall(
        testdata_normalized.flatten(),
        label.flatten(),
        n_interp_points=100)
    print(f"Average Precision: {average_precision:.4f}")
    # Plot the Precision-Recall curve
    plot_precision_recall_curve(precision, recall, average_precision, title=f"PR_curve_AST{extracted_part}")

    # # # for constant threshold
    # # Define the gate
    # gate = 10
    # # Set values to zero up to the gate along the 3rd dimension
    # ori_data[:, :, :gate] = 0
    # # Normalize the result
    # min_val = ori_data.min()
    # max_val = ori_data.max()
    # ori_data_normalized = (ori_data - min_val) / (max_val - min_val)
    # # Compute ROC AUC
    # fpr, tpr, roc_auc = compute_roc_auc(ori_data_normalized, label)
    # # Plot ROC Curve
    # plot_roc_curve(fpr, tpr, roc_auc)

    # ---------------------------------
    # # Define the criterion and validation criterion
    # criterion = getattr(criteria, config.loss_function)(smooth=1e4).to(config.device)
    # val_crite = getattr(criteria, config.val_function)().to(config.device)

    # loss_DL = criterion(torch.tensor(DL_output, dtype=torch.float32).to(config.device), label_torch)
    # score_DL = val_crite(torch.tensor(DL_output, dtype=torch.float32).to(config.device), label_torch)
    # print(f"Loss of DL: {loss_DL}, Score of DL: {score_DL}")

    # ## Convert to tensors
    # # label_tensor_2d = torch.tensor(label.max(axis=2), dtype=torch.float32).to(config.device)

    # # Compute loss and score for AST outputs and reassembled data
    # for AST_output, name in zip([AST_output3db, AST_output6db, AST_output9db, AST_output12db],
    # ['3dB', '6dB', '9dB', '12dB']):
    # AST_output_tensor = torch.tensor(AST_output, dtype=torch.float32).to(config.device)
    # loss_AST = criterion(AST_output_tensor, label_torch)
    # score_AST = val_crite(AST_output_tensor, label_torch)
    # print(f"Final Loss of AST {name}: {loss_AST}, Final Score of AST {name}: {score_AST}")

    # # Example usage
    # zeros_output_tensor = torch.zeros_like(AST_output_tensor)
    # ones_output_tensor = torch.ones_like(AST_output_tensor)
    # loss_zeros = criterion(zeros_output_tensor, label_torch)
    # score_zeros = val_crite(zeros_output_tensor, label_torch)
    # loss_ones = criterion(ones_output_tensor, label_torch)
    # score_ones = val_crite(ones_output_tensor, label_torch)
    # print(f"Loss of all-zero matrix: {loss_zeros}, Score of all-zero matrix: {score_zeros}")
    # print(f"Loss of all-one matrix: {loss_ones}, Score of all-one matrix: {score_ones}")

    DL_output = read_csv_to_3d_array(DL_dir)
    DL_output = np.max(DL_output, axis=2)
    save_2d_array_as_image(DL_output, f"DL_pro{extracted_part}")
    precision, recall, thresholds, f1_scores, average_precision = calculate_precision_recall(
        DL_output.flatten(),
        label.flatten(),
        n_interp_points=100)
    print(f"Average Precision: {average_precision:.4f}")
    # Plot the Precision-Recall curve
    plot_precision_recall_curve(precision, recall, average_precision, title=f"PR_curve_DL{extracted_part}")