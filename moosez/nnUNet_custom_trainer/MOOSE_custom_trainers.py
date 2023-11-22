import torch
from nnunetv2.training.nnUNetTrainer.variants.data_augmentation.nnUNetTrainerDA5 import nnUNetTrainerDA5


class nnUNetTrainer_2000_epochs_DA5NoMirroring(nnUNetTrainerDA5):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 2000

        print(f"Epochs: {self.num_epochs}")

    def configure_rotation_dummyDA_mirroring_and_inital_patch_size(self):
        rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes = \
            super().configure_rotation_dummyDA_mirroring_and_inital_patch_size()
        print(f"Mirroring: {mirror_axes} | {self.inference_allowed_mirroring_axes}")

        mirror_axes = None
        self.inference_allowed_mirroring_axes = None
        print(f"Mirroring: {mirror_axes} | {self.inference_allowed_mirroring_axes}")

        return rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes
