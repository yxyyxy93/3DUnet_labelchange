import torch
from torch.utils.data import DataLoader
import os

# Set mode for testing
os.environ['MODE'] = 'test'
import config
import dataset
import json
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


if __name__ == "__main__":
    # Initialize model
    import numpy as np
    import glob
    import os
    import model_unet3d

    # Set mode for testing
    os.environ['MODE'] = 'test'
    import config
    from visualization import read_metrics
    from test import load_test_dataset, load_checkpoint
    from dataset import plot_dual_orthoslices

    # ------------- visualize some samples
    # Initialize model
    model = model_unet3d.__dict__[config.d_arch_name](in_channels=config.input_dim,
                                                      num_classes=config.output_dim)
    model = model.to(device=config.device)

    results_dir = "results"
    fold_number = 1  # Change as needed
    model_filename = "d_best.pth.tar"
    # Loop through all subdirectories in results_dir

    for subfolder in glob.glob(os.path.join(results_dir, '*')):
        # Construct the file path
        file_path = os.path.join(subfolder, f"_fold {fold_number}", model_filename)

        results_file = os.path.join(subfolder, f"_fold {fold_number}", 'training_metrics.json')
        if os.path.exists(results_file):
            metrics = read_metrics(results_file)
            all_train_losses = metrics['train_losses']
            all_val_losses = metrics['val_losses']
            all_train_scores = metrics['train_scores']
            all_val_scores = metrics['val_scores']
        else:
            print(f"Metrics file not found.")

        # Find the index of the minimum value in all_val_losses
        min_val_loss_index = np.argmin(all_val_losses)
        # Print the minimum value in all_val_losses and the corresponding value in all_val_scores
        print(subfolder, "======================")
        print('Min validation loss:', all_val_losses[min_val_loss_index])
        print('Validation score:', all_val_scores[min_val_loss_index])

        # # Check if the file exists
        # if os.path.exists(file_path):
        #
        # else:
        #     print(f"File not found: {file_path}")
