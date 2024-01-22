import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import imgproc
from utils_func.Read_CSV import read_csv_to_3d_array, save_3d_array_to_csv

# Set mode for testing
os.environ['MODE'] = 'test'
import dataset
from utils_func.criteria import SSIM3D  # Assuming SSIM3D is defined in utils_func.criteria
import config


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


def reassemble_chunks(chunks: list, original_size: tuple = (256, 200, 300),
                      chunk_size: tuple = (16, 16, 256)) -> np.ndarray:
    """
    Reassembles the segmented chunks back to the original size by averaging intersecting predictions.

    Args:
        chunks (list): List of segmented chunks.
        original_size (tuple): Original size of the dataset.
        chunk_size (tuple): Size of each chunk.

    Returns:
        np.ndarray: Reassembled dataset.
    """
    reassembled_data = np.zeros(original_size)
    count_matrix = np.zeros(original_size)  # To count the number of predictions at each point

    for i in range(original_size[1] - chunk_size[0] + 1):
        for j in range(original_size[2] - chunk_size[1] + 1):
            chunk_idx = i * (original_size[2] - chunk_size[1] + 1) + j
            chunk = chunks[chunk_idx].squeeze()

            reassembled_data[:, :, i:i + chunk_size[0], j:j + chunk_size[1]] += chunk
            count_matrix[:, :, i:i + chunk_size[0], j:j + chunk_size[1]] += 1

    # Averaging the predictions
    reassembled_data /= count_matrix

    return reassembled_data


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

    def segment_dataset(self, chunk_size: tuple = (16, 16, 256)) -> list:
        """
        Segments the dataset into smaller chunks with a step of 1.

        Args:
            chunk_size (tuple): Size of each chunk.

        Returns:
            list: A list of segmented chunks, each as a numpy array.
        """
        depth, height, width = self.data.shape[1], self.data.shape[2], self.data.shape[3]
        segmented_data = []

        for i in range(height - chunk_size[0] + 1):
            for j in range(width - chunk_size[1] + 1):
                chunk = self.data[:, :, i:i + chunk_size[0], j:j + chunk_size[1]]
                segmented_data.append(chunk)

        print(f"Segmented into {len(segmented_data)} chunks, each of size {chunk.shape}.")

        return segmented_data


if __name__ == "__main__":
    # Initialize model
    import numpy as np
    import os
    import model_unet3d

    # Set mode for testing
    os.environ['MODE'] = 'test'
    import config

    # ------------- visualize some samples
    # Initialize model
    model = model_unet3d.__dict__[config.d_arch_name](in_channels=config.input_dim,
                                                      num_classes=config.output_dim)
    model = model.to(device=config.device)
    fold_number = 1  # Change as needed
    model_filename = "d_best.pth.tar"
    model_path = os.path.join(config.results_dir, f"_fold {fold_number}", model_filename)
    # Load model checkpoint
    model = load_checkpoint(model, model_path)

    testdata = SimpleCSVLoader("./dataset/test/_snr_100000.00_Inst_amplitude.csv")
    testdata.load_and_preprocess()
    segment_data = testdata.segment_dataset((16, 16))

    segment_output = []
    # Define your batch size
    batch_size = 32  # You can adjust this based on your GPU memory
    # Temporary list to hold a batch of segments
    batch_segments = []

    for i, segment in enumerate(segment_data):
        # Convert segment to a PyTorch tensor and add to the batch
        segment_tensor = torch.tensor(segment, dtype=torch.float).to(config.device)
        batch_segments.append(segment_tensor.unsqueeze(0))  # Add batch dimension
        # Check if the batch is full or if it's the last segment
        if len(batch_segments) == batch_size or i == len(segment_data) - 1:
            # Stack segments into a batch
            batch_tensor = torch.cat(batch_segments, dim=0)
            # Forward pass through the model
            with torch.no_grad():
                batch_output = model(batch_tensor)
            # Convert the outputs to the desired format and add to the segment_output list
            batch_output = batch_output.detach().cpu()
            segment_output.extend(batch_output.unbind())
            # Clear the batch_segments list for the next batch
            batch_segments = []
        # Print progress every n segments
        if i % batch_size == 0:
            print(f"Processed {i / batch_size} / {len(segment_data) / batch_size} segments")

    print("Processing complete.")

    # np.savez("temp_list", *segment_output
    # Load the arrays from the .npz file
    loaded_data = np.load('temp_list.npz')
    # Extract arrays from the loaded data
    segment_output = [loaded_data[key] for key in loaded_data]

    reassembled_data = reassemble_chunks(segment_output)
    # Save to CSV
    save_path = "./dataset/test/exp_test_results.csv"
    save_3d_array_to_csv(reassembled_data, save_path)
