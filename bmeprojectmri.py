
import sys
import os
import glob
import math

import numpy as np
import pandas as pd

from skimage.transform import resize
import nibabel as nib

import tensorflow as tf
import scipy.interpolate as scInterp
from sklearn.metrics import confusion_matrix, accuracy_score
from scipy.interpolate import griddata as interpolate_griddata
import seaborn as sns

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, SGD
from torch.utils.data import TensorDataset, DataLoader

device = torch.device("mps")
print ("Device available:", device)

def check_file_counts(folder_path):
    image_files = [f for f in os.listdir(folder_path) if f.endswith("image.npy")]
    mask_files = [f for f in os.listdir(folder_path) if f.endswith("mask.npy")]

    print(f"Number of image files: {len(image_files)}")
    print(f"Number of mask files: {len(mask_files)}")

    if len(image_files) != len(mask_files):
        print("Discrepancy found in file counts.")
        # Additional code can be added here to investigate which files are missing
    else:
        print("File counts are matching.")

folder_path = 'working/output/train/'
check_file_counts(folder_path)

def fileCountInSubfolder(root_path, subfolder_name):
    path_to_check = os.path.join(root_path, subfolder_name)
    file_list = [file for file in os.listdir(path_to_check) if os.path.isfile(os.path.join(path_to_check, file))]
    return len(file_list)

directory = 'working/output'
dataset_categories = ['test', 'train', 'valid']

for category in dataset_categories:
    total_files = fileCountInSubfolder(directory, category)
    print(f"Total of {total_files} files found in the '{category}' category.")

# Using the specified mask types
mask_types = ["none", "csf", "gm", "wm"]

def visualizeScanWithLayers(scan_layers, scan_image, slice_number, slice_axis, storage_path=None):
    """
    Visualizes a scan alongside its mask layers. Optionally saves the visualization.

    Parameters:
    - scan_layers: Array containing the mask layers for visualization.
    - scan_image: The main scan image to display.
    - slice_number: Identifier for the specific slice of the scan.
    - slice_axis: The axis along which the slice is taken.
    - storage_path (optional): Directory to save the visualization if provided.
    """
    # Setting up the figure for visualization
    plt.figure(figsize=(20, 20))

    # Displaying the main scan image
    plt.subplot(1, len(mask_types) + 1, 1)
    plt.title(f"Scan Slice {slice_number} (Axis {slice_axis})")
    plt.imshow(scan_image[0], cmap='gray')
    plt.axis('off')

    # Iterating through each mask layer for display
    for idx, layer in enumerate(mask_types):
        mask_display = scan_layers[idx]
        plt.subplot(1, len(mask_types) + 1, idx + 2)
        plt.title(f"{slice_number}: {layer} (Axis {slice_axis})")
        plt.imshow(mask_display, cmap='gray')
        plt.axis('off')

    plt.tight_layout()

    # Saving the figure if a save directory is provided
    if storage_path:
        plt.savefig(os.path.join(storage_path, f"Slice_{slice_number}_Axis_{slice_axis}_Visualization.png"))

    plt.show()

class AdvancedConv3DResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        """
        Initializes an advanced 3D convolutional residual block.

        Parameters:
        - in_channels: Number of channels in the input tensor.
        - out_channels: Number of channels in the output tensor.
        """
        super(AdvancedConv3DResBlock, self).__init__()
        # Define the main convolutional path
        self.conv_path = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.GroupNorm(8, out_channels),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        # Shortcut path for matching the dimensions
        self.shortcut = self._make_shortcut(in_channels, out_channels)

        # Final GroupNorm
        self.final_norm = nn.GroupNorm(8, out_channels)

    def _make_shortcut(self, in_channels, out_channels):
        if in_channels != out_channels:
            return nn.Conv3d(in_channels, out_channels, kernel_size=1)
        else:
            return nn.Identity()

    def forward(self, x):
        """
        Forward pass of the advanced 3D convolutional residual block.

        Parameters:
        - x: Input tensor.

        Returns the output tensor after applying the residual block.
        """
        # Apply the main convolutional path and add the result to the shortcut path
        return self.final_norm(self.conv_path(x) + self.shortcut(x))

class ConvBlock(nn.Module):
    """
    A restructured convolution block with two Conv2D-BatchNorm-ReLU sequences.
    """
    def __init__(self, input_channels, output_channels):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1)
        self.batch_norm1 = nn.BatchNorm2d(output_channels)
        self.batch_norm2 = nn.BatchNorm2d(output_channels)

    def forward(self, x):
        x = self.batch_norm1(F.relu(self.conv1(x)))
        x = self.batch_norm2(F.relu(self.conv2(x)))
        return x

class DownBlock(nn.Module):
    """
    A downscaling block combining max pooling and a ConvBlock.
    """
    def __init__(self, input_channels, output_channels):
        super(DownBlock, self).__init__()
        self.pooling = nn.MaxPool2d(2)
        self.conv_block = ConvBlock(input_channels, output_channels)

    def forward(self, x):
        x = self.pooling(x)
        return self.conv_block(x)

class UpBlock(nn.Module):
    """
    An upsampling block combining a transposed convolution and a ConvBlock.
    """
    def __init__(self, input_channels, skip_channels, output_channels):
        super(UpBlock, self).__init__()
        self.upsample = nn.ConvTranspose2d(input_channels, output_channels, kernel_size=2, stride=2)
        self.conv_block = ConvBlock(output_channels + skip_channels, output_channels)

    def forward(self, x, skip_connection):
        x = self.upsample(x)
        x = torch.cat([x, skip_connection], dim=1)
        return self.conv_block(x)

class OutputBlock(nn.Module):
    """
    A final output block applying a 1x1 convolution and softmax activation.
    """
    def __init__(self, input_channels, output_channels):
        super(OutputBlock, self).__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size=1)

    def forward(self, x):
        x = self.conv(x)
        return F.softmax(x, dim=1)

class UNetModel(nn.Module):
    """
    A modified U-Net architecture based CNN model.

    Parameters:
    - model_name: Identifier for the model.
    - input_channels: Number of input channels.
    - num_classes: Number of classes for output segmentation.
    """
    def __init__(self, model_name, input_channels, num_classes):
        super(UNetModel, self).__init__()
        self.model_name = model_name
        self.input_channels = input_channels
        self.num_classes = num_classes

        # Defining U-Net layers
        self.encoder1 = ConvBlock(input_channels, 64)
        self.down1 = DownBlock(64, 128)
        self.down2 = DownBlock(128, 256)
        self.down3 = DownBlock(256, 512)
        self.down4 = DownBlock(512, 1024)
        self.up1 = UpBlock(1024, 512, 512)
        self.up2 = UpBlock(512, 256, 256)
        self.up3 = UpBlock(256, 128, 128)
        self.up4 = UpBlock(128, 64, 64)
        self.final = OutputBlock(64, num_classes)

    def forward(self, x):
        # Forward pass through the U-Net layers
        x1 = self.encoder1(x)

        # Downsampling path
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Upsampling path with skip connections
        u1 = self.up1(x5, x4)
        u2 = self.up2(u1, x3)
        u3 = self.up3(u2, x2)
        u4 = self.up4(u3, x1)

        # Final output layer
        return self.final(u4)

def calculate_loss(criterion, input_images, predictions, ground_truth, operation_mode):
    """
    Calculate custom loss for different tissue channels and the background.
    """
    if operation_mode == "val":
        torch.manual_seed(1102)
        np.random.seed(1102)

    # Generating binary brain map (1 for brain, 0 for background)
    brain_map_binary = (torch.squeeze(input_images, dim=1) > 0).float()

    # Create a mask for applying loss calculation
    ones_tensor = torch.ones(brain_map_binary.shape).to("cpu")
    brain_mask = torch.stack((ones_tensor, brain_map_binary, brain_map_binary, brain_map_binary), dim=1)

    # Isolate regions within the brain for loss computation
    target_regions = torch.mul(brain_mask, predictions)

    # Calculate loss with specified criterion
    computed_loss = criterion(target_regions, ground_truth) / brain_mask.sum()

    return computed_loss, target_regions, brain_mask

def compute_pearson_correlation(isolated_output, target_mask, brain_region_mask):
    """
    Calculate Pearson correlation coefficient in the brain region.
    """
    # Flattening the tensors for correlation calculation
    flat_gt = torch.flatten(target_mask)
    flat_iso = torch.flatten(isolated_output)
    flat_mask = torch.flatten(brain_region_mask)

    # Filter the values within the brain region
    valid_indices = flat_mask.nonzero(as_tuple=True)
    gt_brain_region = flat_gt[valid_indices]
    iso_brain_region = flat_iso[valid_indices]

    # Convert to numpy for correlation calculation
    iso_array = iso_brain_region.cpu().detach().numpy()
    gt_array = gt_brain_region.cpu().detach().numpy()

    # Calculate and return the Pearson correlation coefficient
    return np.corrcoef(iso_array, gt_array)[0][1]



def execute_training(model, model_name, total_epochs, training_loader, validation_loader, model_optimizer, calculate_loss):

    # Creating a directory for the model
    model_directory = f'{model_name}'
    if not os.path.exists(model_directory):
        os.makedirs(model_directory)

    num_train_batches = len(training_loader)
    num_valid_batches = len(validation_loader)

    training_losses = []
    training_pearsons = []

    validation_losses = []
    validation_pearsons = []

    loss_criterion = nn.MSELoss(reduction="sum")
    # TRAINING LOOP
    for epoch in range(total_epochs):
        model.train()

        epoch_train_losses = []
        epoch_train_pearsons = []

        for batch_index, batch_data in enumerate(training_loader):

            # Loading data and moving to device
            images = batch_data['image'].to(device)
            true_masks = batch_data['mask'].to(device)

            # Model predictions
            predictions = model(images)

            # Loss calculation for the batch
            loss, isolated_outputs, brain_masks = calculate_loss(loss_criterion, images, predictions, true_masks, 'train')
            loss_value = loss.item()
            epoch_train_losses.append(loss_value)

            # Pearson coefficient calculation for the batch
            pearson_score = compute_pearson_correlation(isolated_outputs, true_masks, brain_masks)
            epoch_train_pearsons.append(pearson_score)

            # Backpropagation
            model_optimizer.zero_grad()
            loss.backward()
            model_optimizer.step()

            # Progress update
            print(f'Epoch {epoch + 1}/{total_epochs} - Train Batch {batch_index + 1}/{num_train_batches} - Loss: {loss_value}, Pearson: {pearson_score}', end='\r')

        # Averaging metrics over the epoch
        avg_train_loss = np.mean(epoch_train_losses)
        avg_train_pearson = np.mean(epoch_train_pearsons)
        training_losses.append(avg_train_loss)
        training_pearsons.append(avg_train_pearson)

        # VALIDATION LOOP
        model.eval()
        epoch_valid_losses = []
        epoch_valid_pearsons = []
        loss_criterion = nn.MSELoss(reduction="sum")
        with torch.no_grad():
            for batch_index, batch_data in enumerate(validation_loader):
                images = batch_data['image'].to(device)
                true_masks = batch_data['mask'].to(device)

                predictions = model(images)

                loss, isolated_outputs, brain_masks = calculate_loss(loss_criterion, images, predictions, true_masks, 'val')
                loss_value = loss.item()
                epoch_valid_losses.append(loss_value)

                pearson_score = compute_pearson_correlation(isolated_outputs, true_masks, brain_masks)
                epoch_valid_pearsons.append(pearson_score)

                print(f'Epoch {epoch + 1}/{total_epochs} - Valid Batch {batch_index + 1}/{num_valid_batches} - Loss: {loss_value}, Pearson: {pearson_score}', end='\r')

        avg_valid_loss = np.mean(epoch_valid_losses)
        avg_valid_pearson = np.mean(epoch_valid_pearsons)
        validation_losses.append(avg_valid_loss)
        validation_pearsons.append(avg_valid_pearson)

        # Epoch summary
        print(f'Epoch {epoch + 1}/{total_epochs} - Train Loss: {avg_train_loss}, Train Pearson: {avg_train_pearson}, Valid Loss: {avg_valid_loss}, Valid Pearson: {avg_valid_pearson}')

        # Model saving
        torch.save(model.state_dict(), os.path.join(model_directory, f'epoch_{epoch + 1:03}.pth'))

    return training_losses, training_pearsons, validation_losses, validation_pearsons

def compute_dice_coefficient_confusion_matrix(model_output, ground_truth, brain_area_mask):
    """
    Calculate Dice coefficient and return confusion matrix for model evaluation.
    Exclude background from calculations to provide more meaningful Dice scores.
    """
    # Generate prediction and ground truth maps
    prediction_map = torch.argmax(model_output, 1)
    ground_truth_map = torch.argmax(ground_truth, 1)

    # Flatten binary brain mask to exclude background in calculations
    brain_mask_flat = torch.flatten(brain_area_mask[:,1,:,:])

    # Prepare for confusion matrix calculation
    flat_pred_map = torch.flatten(prediction_map).cpu().detach().numpy()
    flat_gt_map = torch.flatten(ground_truth_map).cpu().detach().numpy()
    confusion_mtx = confusion_matrix(flat_gt_map, flat_pred_map)

    dice_scores = []

    # Calculate Dice score for each tissue type
    for tissue_type in range(1, 4):
        tissue_pred = (prediction_map == tissue_type).float()
        tissue_gt = (ground_truth_map == tissue_type).float()

        tissue_pred_flat = torch.flatten(tissue_pred)
        tissue_gt_flat = torch.flatten(tissue_gt)

        valid_indices = brain_mask_flat.nonzero(as_tuple=True)
        tissue_pred_flat = tissue_pred_flat[valid_indices]
        tissue_gt_flat = tissue_gt_flat[valid_indices]

        epsilon = 1e-5  # Small constant to avoid division by zero
        dice = (2.0 * torch.sum(tissue_pred_flat[tissue_gt_flat == 1]) + epsilon) / (torch.sum(tissue_pred_flat) + torch.sum(tissue_gt_flat) + epsilon)
        dice_scores.append(dice)

    average_dice_score = np.mean(dice_scores)
    return average_dice_score, confusion_mtx

