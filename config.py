# Copyright 2023
# Xiaoyu Leo Yang
# Nanyang Technological university
# ==============================================================================
import random
import numpy as np
import torch
from torch.backends import cudnn

# Random seed to maintain reproducible results
random.seed(0)
torch.manual_seed(0)
np.random.seed(0)
# Use GPU for training by default
device = torch.device("cuda", 0)
# Turning on when the image size does not change during training can speed up training
cudnn.benchmark = True

# Model arch config
input_dim = 21
hidden_dim = 10
kernel_size = 3
output_size = (50, 50, 200)  #

# Current configuration parameter method
mode = "train_convLSTM"
# mode = "test"

# Experiment name, easy to save weights and log files
exp_name = "convLSTM_baseline"
# exp_name = "SRGan_baseline"

g_arch_name = "ConvLSTM3DClassifier"

# When evaluating the performance of the SR model, whether to verify only the Y channel image data
only_test_y_channel = True
# Image magnification factor
upscale_factor = 1

# How many iterations to print the training result
train_print_frequency = 5
valid_print_frequency = 5

if mode == "train_convLSTM":
    # Dataset address
    image_dir = r'D:\python_work\ConvLSTM_3dultrasound\dataset\sim_data'  # path to the 'sim_data' directory
    label_dir = r'D:\python_work\ConvLSTM_3dultrasound\dataset\sim_struct'  # path to the 'sim_struct' directory

    batch_size = 4
    num_workers = 4

    # The address to load the pretrained model
    pretrained_model_path = "./results/pretrained_models/ConvLSTM_pretrain.pth.tar"

    # The address to load the pretrained model
    pretrained_d_model_weights_path = ""

    # Incremental training and migration training
    resume_d_model_weights_path = f""

    # Incremental training and migration training
    resume = ""

    # Total num epochs
    epochs = 100

    # Optimizer parameter
    model_lr = 1e-5
    model_betas = (0.9, 0.99)
    model_eps = 1e-8
    model_weight_decay = 0.0

    # EMA parameter
    model_ema_decay = 0.5
    # How many iterations to print the training result
    print_frequency = 200

    # Dynamically adjust the learning rate policy
    lr_scheduler_milestones = [int(epochs * 0.125), int(epochs * 0.250), int(epochs * 0.500), int(epochs * 0.750)]
    lr_scheduler_gamma = 0.5

if mode == "test":
    # Test data address
    lr_dir = f"./data/ImageTest/noisy_images"
    sr_dir = f"./results/test/{exp_name}"
    hr_dir = f"./data/ImageTest/origin_images"

    # model_path = r"F:\Xiayang\python_work\SRGAN-PyTorch-ultrasonic\samples\SRResNet_baseline/g_epoch_100.pth.tar"
    # model_path = r".\results\ESRGAN_x2\g_best.pth.tar"
    model_path = r".\results\RRDBNet_x1\g_last.pth.tar"
    # model_path = r"F:\Xiayang\python_work\SRGAN-PyTorch-ultrasonic\samples\SRGan_baseline\g_epoch_100.pth.tar"
