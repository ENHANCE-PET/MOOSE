# Reproducing ENHANCE.PET Dataset Results

This guide explains how to reproduce the ENHANCE.PET dataset results 
using [**nnU-Net v2**](https://github.com/MIC-DKFZ/nnUNet).

---

## 1. Install Required Packages

First, install **PyTorch** according to your system configuration:

- Visit the official [PyTorch website](https://pytorch.org/get-started/locally/).
- Select your operating system, package manager, and CUDA version (if applicable).
- Run the generated installation command.

Once PyTorch is installed, install **nnU-Net v2**:

```bash
pip install nnunetv2
```
## 2. Prepare the Data

After downloading the dataset and installing the required packages:

- Format the dataset according to the official nnU-Net specifications:  
  [nnU-Net Dataset Format Documentation](https://github.com/MIC-DKFZ/nnUNet/blob/273d6147f2d71bafc27a5e1c6fcab9f59ceba7d0/documentation/dataset_format.md)

- Once the dataset is correctly structured, configure the required environment variables as described in the official setup guide:  
  [nnU-Net Path Setup Documentation](https://github.com/MIC-DKFZ/nnUNet/blob/273d6147f2d71bafc27a5e1c6fcab9f59ceba7d0/documentation/setting_up_paths.md)

Ensure all paths are exported correctly before proceeding to preprocessing and training.

## 3. Planning and Preprocessing

Once the dataset is correctly formatted and all required paths are configured, run:

```bash
nnUNetv2_plan_and_preprocess -d NUMBER_OF_DATASET --verify_dataset_integrity
```
Replace NUMBER_OF_DATASET with your dataset ID.

This step:

- Verifies dataset integrity.
- Generates dataset-specific configuration plans.
- Performs preprocessing required for training.

Wait for this process to complete before starting the training step.

## 4. Training
After preprocessing is complete, start training:
```bash
nnUNetv2_train NUMBER_OF_DATASET 3d_fullres all -tr nnUNetTrainer_2000epochs_NoMirroring
```
This command:
- Trains the 3D full-resolution configuration.
- Uses all data for training.
- Trains over 2000 epochs without using mirroring for data augmentation.

## 5. Results and Inference
After training completes:
- The trained model is stored in your nnUNet_results directory.
- Refer to the official [nnU-Net documentation](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/how_to_use_nnunet.md#run-inference) for inference instructions.