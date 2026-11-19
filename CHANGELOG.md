# Changelog

## Unreleased: every model through nnU-Net's streaming inference

Requires nnU-Net 2.9.0 or newer (`nnunetv2>=2.9.0`).

### How models run
- **One pipeline for every model:** each runs through nnU-Net's streaming predictor (`moosez/streaming.py`). An image is
  read once; each model gets it through its nnU-Net reader and its own preprocessing. The label map comes out on the
  grid and in the orientation of the input image.
- **No label-map upsampling:** nnU-Net resamples the network outputs to the input grid before the argmax. MOOSE no
  longer resamples label maps with nearest neighbour, and no longer splits images into chunks.
- **Models trained on images that MOOSE had resampled itself** (every model before `clin_ct_anatomy`) get the same kind
  of input: cubic B-spline resampling on every axis to the model's spacing. Their outputs go back to the input grid
  with trilinear interpolation.
- **Model loading:** all models of a call load once. While the GPU predicts one image, threads read and preprocess
  the next images and write the outputs of finished ones.
- **Registry fields:** a model's registry entry can set its reader, orientation, input resampling and region of
  interest (`moosez/models.py`). A model trained with the current nnU-Net needs none of them: it runs as its plans say.
  Plans that name `NibabelIOWithReorient` are read with `SimpleITKIOWithReorient`, which gives the same arrays.

### New
- `clin_ct_anatomy`: one model for the 125 structures of `clin_ct_cardiac`, `clin_ct_muscles`, `clin_ct_organs`,
  `clin_ct_peripheral_bones`, `clin_ct_ribs` and `clin_ct_vertebrae`, without brain, skull and trachea. It is the
  default model. Its weights are installed from their archive with `-mi` (see README).
- `-i <files or folders> -o <folder>`: image files without subject folders. Every image goes to every model, and the
  outputs go into one folder.
- `-mi <archive.zip>` installs a model from a zip archive that holds its model folder, and prints the model's name.
- `-m` is optional (default: `clin_ct_anatomy`).
- `MoosePredictor` (Python): models loaded once, any number of images.
- The start banner also cites the streaming-inference paper.

### Changed
- Statistics are opt-in: `-s` writes the volume and intensity statistics (`stats/`). 3.2.2 always wrote them.
- `-herd` starts one process per GPU, each with its share of the subjects (`-herd N`: N GPUs). It was N parallel
  instances.
- `-b` writes one row for the whole run. There are no per-subject rows or plots any more.
- `MOOSE_information.json`: each subject's file is written when its run (or its `-herd` process) ends.
- Statistics are computed from the label map in memory. Volumes are voxel counts times the voxel volume, the same
  values as before. `-sl` files are written by several threads.
- A model named twice runs once.

### Fixed
- `clin_ct_lungs`, `clin_ct_fast_cardiac`, `clin_ct_ALPACA` and `clin_ct_PUMA` were trained on LAS arrays but were
  given RAS arrays. Their left / right labels came out swapped; they now get LAS arrays.
- Long scans: the chunked resampling shifted label maps along the body axis, by up to 4 mm on a 1940-slice CT.
- `clin_mr_FVM` could not load: its archive holds the model folder one level deeper. Archives are now unpacked wherever
  they hold the model folder.
- Intensity statistics of a second modality failed when its grid differed from the label map's only by rounding.
- The 1 s pauses after progress messages are gone.

### Removed
- **Dependency:** Dask.
- **Modules:** `moosez/predict.py`, `moosez/workflows.py`, `moosez/nnUNet_custom_trainer/`.
- **Helpers in `image_processing`:** `ImageResampler`, `ImageChunker`, `image_reorient`, `extract_labels_and_write`.
  `reslice_identity` is now `reslice_label_image`.
- **Helpers in `moosez`:** `moose_subject`.
- **`Model` attributes:** `voxel_spacing`, `transpose_forward`, `transpose_backward`.
