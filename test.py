import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import imgproc
from utils_func.Read_CSV import read_csv_to_3d_array, save_3d_array_to_csv

import dataset
from utils_func.criteria import SSIM3D  # Assuming SSIM3D is defined in utils_func.criteria

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

    def segment_dataset(self, chunk_size: tuple = (16, 16, 256), step: int = 1) -> list:
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

        print(f"Segmented into {len(segmented_data)} chunks, each of size {chunk.shape}, with a step of {step}.")

        return segmented_data


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

        if i % batch_size == 0:
            print(f"Processed {i / batch_size} / {len(segment_data) / batch_size} segments")

    print("Processing complete.")
    return segment_output


if __name__ == "__main__":
    # Initialize model
    import numpy as np
    import os
    import model_unet3d

    # Set mode for testing
    os.environ['MODE'] = 'test'
    import config

    # User-defined flag to choose processing mode
    process_from_start = True # Set this to False to load from 'temp_list.npz'
 
    if process_from_start:
        # Your existing code to initialize and load the model
        model = model_unet3d.__dict__[config.d_arch_name](in_channels=config.input_dim,
                                                          num_classes=config.output_dim)
        model = model.to(device=config.device)
        fold_number = 1 # Change as needed
        model_filename = "d_best.pth.tar"
        model_path = os.path.join(config.results_dir, f"_fold {fold_number}", model_filename)
        model = load_checkpoint(model, model_path)
 
        # Load and preprocess test data
        testdata = SimpleCSVLoader("/mnt/raid5/xiaoyu/Ultrasound_data/dataset_woven_[#090]8_0-1defect/test/_snr_100000.00_Inst_amplitude_090_2.csv")
        testdata.load_and_preprocess()
        step = 1
        segment_data = testdata.segment_dataset(chunk_size=(16, 16), step=step) 
 
        # Define batch size
        batch_size = 64  # Adjust based on GPU memory
        # Process data
        segment_output = process_data(model, segment_data, batch_size, config.device)
        # Save output to npz
        np.savez("temp_list", *segment_output)
    else:
        # Load the arrays from the .npz file
        loaded_data = np.load('temp_list.npz')
        # Extract arrays from the loaded data
        segment_output = [loaded_data[key] for key in loaded_data]

    # Reassemble and save the data
    original_size = (256, 241, 281)
    # original_size = (256, 235, 300)
    reassembled_data = reassemble_chunks(segment_output, original_size=original_size, step=step)
    save_path = "/mnt/raid5/xiaoyu/Ultrasound_data/dataset_woven_[#090]8_0-1defect/test/exp_test_results.csv"
    # Assuming original data was in (height, width, depth), revert the reassembled data to this order
    reassembled_data = np.transpose(reassembled_data, (1, 2, 0))
    save_3d_array_to_csv(reassembled_data, save_path)
