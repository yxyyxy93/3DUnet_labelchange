from typing import Tuple, List, Any

import numpy as np
import torch
from torch.utils.data import DataLoader

import dataset
from dataset import imgproc
from utils_func.Read_CSV import read_csv_to_3d_array, save_3d_array_to_csv
from utils_func.criteria import SSIM3D  # Assuming SSIM3D is defined in utils_func.criteria

import os

# Set mode for testing
os.environ['MODE'] = 'test'
import config
import model_unet3d


def print_statistics(tensor, name):
    mean = tensor.mean().item()
    std = tensor.std().item()
    min_val = tensor.min().item()
    max_val = tensor.max().item()
    size = tensor.shape

    print(f"Statistics for {name}:")
    print(f"Mean: {mean}")
    print(f"Standard Deviation: {std}")
    print(f"Min: {min_val}")
    print(f"Max: {max_val}")
    print(f"Size: {size}")
    print("-" * 30)


def load_checkpoint(model_load, checkpoint_path):
    # Load a checkpoint into the model
    checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
    model_load.load_state_dict(checkpoint["state_dict"])
    return model_load


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
            with torch.no_grad():
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
        model = model_unet3d.__dict__[config.d_arch_name](in_channels=config.input_dim, num_classes=config.output_dim)
        model = model.to(device=config.device)
        model_path = os.path.join(config.results_dir, f"_fold {fold_number}", model_filename)
        model = load_checkpoint(model, model_path)
        # Process data
        segment_output = process_data(model, segment_data, config.batch_size, config.device)
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


def process_AST_output(AST_output, threshold=0.5):
    # Convert AST_output from dB to ratio
    # AST_output_ratio = 10 ** (AST_output / 20)
    # AST_output_ratio = AST_output
    # Apply binary threshold with a value of 0.5
    AST_output_ratio = np.where(AST_output > threshold, 1.0, 0.0)
    # Normalize AST_output_ratio to the range 0-1
    AST_output_min = AST_output_ratio.min()
    AST_output_max = AST_output_ratio.max()
    AST_output_normalized = (AST_output_ratio - AST_output_min) / (AST_output_max - AST_output_min)
    # find the max along the 3rd dimension to obtain 2D maps
    AST_output_2d = AST_output_normalized.max(axis=2)
    # Convert to tensor
    AST_output_tensor_2d = torch.tensor(AST_output_2d, dtype=torch.float32).to(config.device)
    return AST_output_tensor_2d


def compute_loss_and_score(output_tensor, label_tensor, criterion, val_crite):
    loss = criterion(output_tensor, label_tensor)
    score = val_crite(output_tensor, label_tensor)
    return loss.item(), score.item()


def compute_mean_depth_indices(array_3d):
    """
    Compute the mean depth index along the 3rd dimension using a weighted average.

    Args:
    array_3d (numpy.ndarray): Input 3D array.

    Returns:
    torch.Tensor: Tensor containing the mean depth indices.
    """
    depths = np.arange(array_3d.shape[2])
    weighted_sum = (array_3d * depths).sum(axis=2)
    sum_weights = array_3d.sum(axis=2)
    mean_depth_indices = np.divide(weighted_sum, sum_weights, out=np.zeros_like(weighted_sum), where=sum_weights != 0)
    # mean_depth_indices = np.argmax(array_3d, axis=2)
    # min_val = mean_depth_indices.min()
    # max_val = mean_depth_indices.max()
    # normalized_tensor = (mean_depth_indices - min_val) / (max_val - min_val)

    normalized_tensor = mean_depth_indices / 255

    return torch.tensor(normalized_tensor, dtype=torch.float32)


def main():
    from utils_func import criteria
    from torch.nn import MSELoss
    import matplotlib.pyplot as plt

    # Load data
    AST_outputs_files = [
        "D:\\python_work\\WovenComposite_defects\\3dUnet_ultrasound_defect_LabelChange_depthchannel\\dataset\\test"
        "\\_snr_100000.00_Inst_amplitude_090_2_STEG_3dB.csv",
        "D:\\python_work\\WovenComposite_defects\\3dUnet_ultrasound_defect_LabelChange_depthchannel\\dataset\\test"
        "\\_snr_100000.00_Inst_amplitude_090_2_STEG_6dB.csv",
        "D:\\python_work\\WovenComposite_defects\\3dUnet_ultrasound_defect_LabelChange_depthchannel\\dataset\\test"
        "\\_snr_100000.00_Inst_amplitude_090_2_STEG_9dB.csv",
        "D:\\python_work\\WovenComposite_defects\\3dUnet_ultrasound_defect_LabelChange_depthchannel\\dataset\\test"
        "\\_snr_100000.00_Inst_amplitude_090_2_STEG_12dB.csv"
    ]
    AST_names = ['3dB', '6dB', '9dB', '12dB']

    # Read the label data
    label = read_csv_to_3d_array(config.label_exp_dir)

    # Convert label to tensor
    label_2d = label.max(axis=2)
    label_tensor_2d = torch.tensor(label_2d, dtype=torch.float32).to(config.device)

    # Compute mean depth index for label
    mean_depth_indices_label = compute_mean_depth_indices(label).to(config.device)

    # Define the criterion and validation criterion
    criterion = getattr(criteria, config.loss_function)(smooth=1e3).to(config.device)
    val_crite = getattr(criteria, config.val_function)().to(config.device)

    # Process each AST output and compute losses and scores
    mse_criterion = MSELoss(reduction='sum').to(config.device)

    for AST_file, name in zip(AST_outputs_files, AST_names):
        AST_output = read_csv_to_3d_array(AST_file)
        AST_output_tensor = process_AST_output(AST_output)

        loss, score = compute_loss_and_score(AST_output_tensor, label_tensor_2d, criterion, val_crite)
        # Compute mean depth index for AST output
        mean_depth_indices_AST = compute_mean_depth_indices(AST_output).to(config.device)
        mse_loss_mean = mse_criterion(mean_depth_indices_AST * AST_output_tensor,
                                      mean_depth_indices_label * label_tensor_2d)
        print(f"Dice Loss of AST {name}: {loss}, Score of AST {name}: {score} MSE Loss Depth: {mse_loss_mean.item()}")

    # Compute loss and score for zero and one matrices
    AST_output_tensor_zeros = torch.zeros_like(label_tensor_2d)
    AST_output_tensor_ones = torch.ones_like(label_tensor_2d)
    loss_zeros, score_zeros = compute_loss_and_score(AST_output_tensor_zeros, label_tensor_2d, criterion, val_crite)
    loss_ones, score_ones = compute_loss_and_score(AST_output_tensor_ones, label_tensor_2d, criterion, val_crite)
    print(f"Loss of all-zero matrix: {loss_zeros}, Score of all-zero matrix: {score_zeros}")
    print(f"Loss of all-one matrix: {loss_ones}, Score of all-one matrix: {score_ones}")

    # Load DL_output
    DL_output = read_csv_to_3d_array(
        "D:\\python_work\\WovenComposite_defects\\3dUnet_ultrasound_defect_LabelChange_depthchannel\\dataset\\test"
        "\\Inst_amplitude_090_2_unequal_UNet3D_DiceLoss_1_20000_3_2024-06-01.csv")

    # Running the main loop for different thresholds
    thresholds = [0.1, 0.3, 0.5, 0.7]

    for threshold in thresholds:
        DL_output_tensor_2d = process_AST_output(DL_output, threshold=threshold)
        loss_DL, score_DL = compute_loss_and_score(DL_output_tensor_2d, label_tensor_2d, criterion, val_crite)

        # Compute mean depth index along the 3rd dimension using weighted average
        mean_depth_indices_DL = compute_mean_depth_indices(DL_output).to(config.device)
        mse_loss_mean = mse_criterion(mean_depth_indices_DL * DL_output_tensor_2d,
                                      mean_depth_indices_label * label_tensor_2d)
        print(
            f"Threshold: {threshold}, Dice Loss of DL: {loss_DL}, Score of DL: {score_DL}; MSE Loss Depth: {mse_loss_mean.item()}")

    DL_output_tensor_2d = DL_output.max(axis=2)
    # Convert to tensor
    DL_output_tensor_2d = torch.tensor(DL_output_tensor_2d, dtype=torch.float32).to(config.device)
    loss_DL, score_DL = compute_loss_and_score(DL_output_tensor_2d, label_tensor_2d, criterion, val_crite)
    print(f"Dice Loss of DL: {loss_DL}, Score of DL: {score_DL}")

    plt.imshow((mean_depth_indices_DL - mean_depth_indices_label) * label_tensor_2d.cpu().numpy(), cmap='viridis')
    plt.colorbar()
    plt.title("Mean Depth Indices of DL Output")
    plt.xlabel("Width")
    plt.ylabel("Height")
    plt.show()

    # print_statistics(DL_output, "DL")


if __name__ == "__main__":
    main()
