![Moose-logo](Images/moose.png)
## MOOSE 3.0 🦌- Furiously Fast. Brutally Efficient. Unmatched Precision. 💪
[![Documentation Status](https://img.shields.io/readthedocs/moosez/latest.svg?style=flat-square&logo=read-the-docs&color=CC00FF)](https://moosez.rtfd.io/en/latest/?badge=latest) [![PyPI version](https://img.shields.io/pypi/v/moosez?color=FF1493&style=flat-square&logo=pypi)](https://pypi.org/project/moosez/) [![License: GPL v3](https://img.shields.io/badge/License-GPLv3-red.svg?style=flat-square&logo=gnu&color=FF0000)](https://www.gnu.org/licenses/gpl-3.0) [![Discord](https://img.shields.io/badge/Discord-Chat-blue.svg?style=flat-square&logo=discord&color=0000FF)](https://discord.gg/9uTHYhWCA5) [![Monthly Downloads](https://img.shields.io/pypi/dm/moosez?label=Downloads%20(Monthly)&color=9400D3&style=flat-square&logo=python)](https://pypi.org/project/moosez/) [![Daily Downloads](https://img.shields.io/pypi/dd/moosez?label=Downloads%20(Daily)&color=9400D3&style=flat-square&logo=python)](https://pypi.org/project/moosez/)

Welcome to the new and improved MOOSE (v3.0), where speed and efficiency aren't just buzzwords—they're a way of life. 

**💨 3x Faster Than Before**  
Like a moose sprinting through the woods (okay, maybe not that fast), MOOSE 3.0 is built for speed. It's 3x faster than its older sibling, MOOSE 2.0, which was already no slouch. Blink and you'll miss it. ⚡

**💻 Memory: Light as a Feather, Strong as a Bull**  
Forget "Does it fit on my laptop?" The answer is YES. 🕺 Thanks to Dask wizardry, all that data stays in memory. No disk writes, no fuss. Run total-body CT on that 'decent' laptop you bought three years ago and feel like you’ve upgraded. 🥳

**🛠️ Any OS, Anytime, Anywhere**  
Windows, Mac, Linux—we don’t play favorites. 🍏 Mac users, you’re in luck: MOOSE runs natively on MPS, getting you GPU-like speeds without the NVIDIA guilt. 🚀 

**🎯 Trained to Perfection**  
This is our best model yet, trained on a whopping 1.7k datasets. More data, better results. Plus you can run multiple models at the same time - You'll be slicing through images like a knife through warm butter. (Or tofu, if you prefer.) 🧈🔪

**🖥️ The 'Herd' Mode 🖥️**  
Got a powerhouse server just sitting around? Time to let the herd loose! Flip the **Herd Mode** switch and watch MOOSE multiply across your compute like... well, like a herd of moose! 🦌🦌🦌 The more hardware you have, the faster your inference gets done. Scale up, speed up, and make every bit of your server earn its oats. 🌾💨

MOOSE 3.0 isn't just an upgrade—it's a lifestyle. A faster, leaner, and stronger lifestyle. Ready to join the herd? 🦌✨

<div align="center">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://github.com/QIMP-Team/MOOSE/blob/main/Images/MOOSE_storyboard_dark.gif">
  <source media="(prefers-color-scheme: light)" srcset="https://github.com/QIMP-Team/MOOSE/blob/main/Images/MOOSE_storyboard_light.gif">
  <img alt="Shows an illustrated MOOSE story board adopted to different themes" src="https://github.com/QIMP-Team/MOOSE/blob/main/Images/MOOSE_storyboard_light.gif">
</picture>
</div>

## Available Segmentation Models 🧬

MOOSE 3.0 offers a wide range of segmentation models catering to various clinical and preclinical needs. Here are the models currently available:

### Clinical 👫🏽
- [`clin_ct_lungs`](https://github.com/QIMP-Team/MOOSE/blob/f48e4b6f9155f7b50bb042b045550b9cc25f6989/moosez/constants.py#L85)
- [`clin_ct_organs`](https://github.com/QIMP-Team/MOOSE/blob/f48e4b6f9155f7b50bb042b045550b9cc25f6989/moosez/constants.py#L66)
- [`clin_ct_body`](https://github.com/QIMP-Team/MOOSE/blob/f48e4b6f9155f7b50bb042b045550b9cc25f6989/moosez/constants.py#L92)
- [`clin_ct_ribs`](https://github.com/QIMP-Team/MOOSE/blob/f48e4b6f9155f7b50bb042b045550b9cc25f6989/moosez/constants.py#L124)
- [`clin_ct_muscles`](https://github.com/QIMP-Team/MOOSE/blob/f48e4b6f9155f7b50bb042b045550b9cc25f6989/moosez/constants.py#L151)
- [`clin_ct_peripheral_bones`](https://github.com/QIMP-Team/MOOSE/blob/f48e4b6f9155f7b50bb042b045550b9cc25f6989/moosez/constants.py#L163)
- [`clin_ct_fat`](https://github.com/QIMP-Team/MOOSE/blob/f48e4b6f9155f7b50bb042b045550b9cc25f6989/moosez/constants.py#L199)
- [`clin_ct_vertebrae`](https://github.com/QIMP-Team/MOOSE/blob/f48e4b6f9155f7b50bb042b045550b9cc25f6989/moosez/constants.py#L209)
- [`clin_ct_cardiac`](https://github.com/QIMP-Team/MOOSE/blob/f48e4b6f9155f7b50bb042b045550b9cc25f6989/moosez/constants.py#L239)
- [`clin_ct_digestive`](https://github.com/QIMP-Team/MOOSE/blob/f48e4b6f9155f7b50bb042b045550b9cc25f6989/moosez/constants.py#L251)
- [`clin_ct_all_bones_v1`](https://github.com/QIMP-Team/MOOSE/blob/f48e4b6f9155f7b50bb042b045550b9cc25f6989/moosez/constants.py#L264)
- [`clin_pt_fdg_brain_v1`](https://github.com/QIMP-Team/MOOSE/blob/f48e4b6f9155f7b50bb042b045550b9cc25f6989/moosez/constants.py#L311)

### Preclinical 🐁
- [`preclin_ct_legs`](https://github.com/QIMP-Team/MOOSE/blob/f48e4b6f9155f7b50bb042b045550b9cc25f6989/moosez/constants.py#L260)
- [`preclin_mr_all`](https://github.com/QIMP-Team/MOOSE/blob/f48e4b6f9155f7b50bb042b045550b9cc25f6989/moosez/constants.py#L101)

Each model is designed to provide high-quality segmentation with MOOSE 3.0's optimized algorithms and data-centric AI principles.

<div align="center">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://github.com/QIMP-Team/MOOSE/blob/main/Images/MOOSE-Rotational-MIP.gif">
  <source media="(prefers-color-scheme: light)" srcset="https://github.com/QIMP-Team/MOOSE/blob/main/Images/MOOSE-Rotational-MIP.gif">
  <img alt="Shows an illustrated MOOSE story board adopted to different themes" src="https://github.com/QIMP-Team/MOOSE/blob/main/Images/MOOSE-Rotational-MIP.gif">
</picture>
</div>

## Star History 🤩

<a href="https://star-history.com/#QIMP-Team/MOOSE&Date">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=ENHANCE-PET/MOOSE&type=Date&theme=dark" />
    <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=ENHANCE-PET/MOOSE&type=Date" />
    <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=QIMP-Team/MOOSE&type=Date" />
  </picture>
</a>



## Citations ❤️ 

- Shiyam Sundar, L. K., Yu, J., Muzik, O., Kulterer, O., Fueger, B. J., Kifjak, D., Nakuz, T., Shin, H. M., Sima, A. K., Kitzmantl, D., Badawi, R. D., Nardo, L., Cherry, S. R., Spencer, B. A., Hacker, M., & Beyer, T. (2022). Fully-automated, semantic segmentation of whole-body <sup>18</sup>F-FDG PET/CT images based on data-centric artificial intelligence. *Journal of Nuclear Medicine*. https://doi.org/10.2967/jnumed.122.264063
- Isensee, F., Jaeger, P.F., Kohl, S.A.A. et al. nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation. Nat Methods 18, 203–211 (2021). https://doi.org/10.1038/s41592-020-01008-z


## Requirements ✅

Before you dive into the incredible world of MOOSE 3.0, here are a few things you need to ensure for an optimal experience:

- **Operating System**: We've got you covered whether you're on Windows, Mac, or Linux. MOOSE 3.0 has been tested across these platforms to ensure seamless operation.

- **Memory**: MOOSE 3.0 has quite an appetite! Make sure you have at least 16GB of RAM for the smooth running of all tasks.

- **GPU**: If speed is your game, an NVIDIA GPU is the name! MOOSE 3.0 leverages GPU acceleration to deliver results fast. Don't worry if you don't have one, though - it will still work, just at a slower pace.

- **Python**: Ensure that you have Python 3.10 installed on your system. MOOSE 3.0 likes to keep up with the latest, after all!

So, that's it! Make sure you're geared up with these specifications, and you're all set to explore everything MOOSE 3.0 has to offer. 🚀🌐

## Installation Guide 🛠️

Available on Windows, Linux, and MacOS, the installation is as simple as it gets. Follow our step-by-step guide below and set sail on your journey with MOOSE 3.0.

## For Linux (and Intel x86 Mac)🐧

1. First, create a Python environment. You can name it to your liking; for example, 'moose-env'.
   ```bash
   python3.10 -m venv moose-env
   ```

2. Activate your newly created environment.
   ```bash
   source moose-env/bin/activate  # for Linux
   ```

3. Install MOOSE 3.0.
   ```bash
   pip install moosez
   ```

Voila! You're all set to explore with MOOSE 3.0.

## For Macs powered by Apple Silicon (M series chips with MPS) 🍏

1. First, create a Python environment. You can name it to your liking; for example, 'moose-env'.
   ```bash
   python3.10 -m venv moose-env
   ```

2. Activate your newly created environment.
   ```bash
   source moose-env/bin/activate 
   ```

3. Install MOOSE 3.0 and a special fork of PyTorch (MPS specific). You need to install the MPS specific branch for making MOOSE work with MPS
   ```bash
   pip install moosez
   pip install git+https://github.com/LalithShiyam/pytorch-mps.git
   ```
Now you are ready to use MOOSE on Apple Silicon 🏎⚡️.

## For Windows 🪟

1. Create a Python environment. You could name it 'moose-env', or as you wish.
   ```bash
   python3.10 -m venv moose-env
   ```

2. Activate your newly created environment.
   ```bash
   .\moose-env\Scripts\activate
   ```

3. Go to the PyTorch website and install the appropriate PyTorch version for your system. **!DO NOT SKIP THIS!**

4. Finally, install MOOSE 3.0.
   ```bash
   pip install moosez
   ```

There you have it! You're ready to venture into the world of 3D medical image segmentation with MOOSE 3.0.

Happy exploring! 🚀🔬

## Usage Guide 📚

### Command-Line Tool for Batch Processing 🖥️🚀

Getting started with MOOSE 3.0 is as easy as slicing through butter 🧈🔪. Use the command-line tool to process multiple segmentation models in sequence or in parallel, making your workflow a breeze. 🌬️

#### *Running Single/Multiple Models in Sequence* 🏃‍♂️🎯

You can now run single or several models in sequence with a single command. Just provide the path to your subject images and list the segmentation models you wish to apply:

```bash
# For single model inference
moosez -d <path_to_image_dir> -m <model_name>

# For multiple model inference
moosez -d <path_to_image_dir> \
       -m <model_name1> \
          <model_name2> \
          <model_name3> \
```

For instance, to run clinical CT organ segmentation on a directory of images, you can use the following command:

```bash
moosez -d <path_to_image_dir> -m clin_ct_organs
```
Likewise, to run multiple models e.g. organs, ribs, and vertebrae, you can use the following command:

```bash
moosez -d <path_to_image_dir> \
       -m clin_ct_organs \
          clin_ct_ribs \
          clin_ct_vertebrae
 ```
MOOSE 3.0 will handle each model one after the other—no fuss, no hassle. 🙌✨

#### *Herd Mode: Running Parallel Instances* 🦌💨💻

Got a powerful server or HPC? Let the herd roam! 🦌🚀 Use **Herd Mode** to run multiple MOOSE instances in parallel. Just add the `-herd` flag with the number of instances you wish to run simultaneously:

```bash
moosez -d <path_to_image_dir> \
       -m clin_ct_organs \
          clin_ct_ribs \
          clin_ct_vertebrae \
       -herd 2
```
MOOSE will run two instances at the same time, utilizing your compute power like a true multitasking pro. 💪👨‍💻👩‍💻

And that's it! MOOSE 3.0 lets you process with ease and speed. ⚡✨



Need assistance along the way? Don't worry, we've got you covered. Simply type:

```bash
moosez -h
```

This command will provide you with all the help and the information about the available [models](https://github.com/QIMP-Team/MOOSE/blob/3fcfad710df790e29a4a1ea16f22e480f784f38e/moosez/resources.py#L29) and the [regions](https://github.com/QIMP-Team/MOOSE/blob/3fcfad710df790e29a4a1ea16f22e480f784f38e/moosez/constants.py#L64) it segments.

### Using MOOSE 3.0 as a Library 📦🐍

MOOSE 3.0 isn't just a command-line powerhouse; it’s also a flexible library for Python projects. Here’s how to make the most of it:

First, import the `moose` function from the `moosez` package in your Python script:

 ```python
from moosez import moose
 ```

#### *Calling the `moose` Function* 🦌

The `moose` function is versatile and accepts various input types. It takes four main arguments:

1. `input`: The data to process, which can be:
   - A path to an input file or directory (NIfTI, either `.nii` or `.nii.gz`).
   - A tuple containing a NumPy array and its spacing (e.g., `numpy_array`, `(spacing_x, spacing_y, spacing_z)`).
   - A `SimpleITK` image object.
2. `model_names`: A single model name or a list of model names for segmentation.
3. `output_dir`: The directory where the results will be saved.
4. `accelerator`: The type of accelerator to use (`"cpu"`, `"cuda"`, or `"mps"` for Mac).

#### Examples 📂✂️💻

Here are some examples to illustrate different ways to use the `moose` function:

1. **Using a file path and multiple models:**
    ```python
    moose('/path/to/input/file', ['clin_ct_organs', 'clin_ct_ribs'], '/path/to/save/output', 'cuda')
    ```

2. **Using a NumPy array with spacing:**
    ```python
    moose((numpy_array, (1.5, 1.5, 1.5)), 'clin_ct_organs', '/path/to/save/output', 'cuda')
    ```

3. **Using a SimpleITK image:**
    ```python
    moose(simple_itk_image, 'clin_ct_organs', '/path/to/save/output', 'cuda')
    ```

#### Ready, Set, Segment! 🚀

That's it! With these flexible inputs, you can use MOOSE 3.0 to fit your workflow perfectly—whether you’re processing a single image, a stack of files, or leveraging different data formats. 🖥️🎉

Happy segmenting with MOOSE 3.0! 🦌💫


## Directory Structure and Naming Conventions for MOOSE 📂🏷️

### Applicable only for batch mode ⚠️

Using MOOSE 3.0 optimally requires your data to be structured according to specific conventions. MOOSE 3.0 supports both DICOM and NIFTI formats. For DICOM files, MOOSE infers the modality from the DICOM tags and checks if the given modality is suitable for the chosen segmentation model. However, for NIFTI files, users need to ensure that the files are named with the correct modality as a suffix.

### Required Directory Structure 🌳
Please structure your dataset as follows:

```
MOOSEv2_data/ 📁
├── S1 📂
│   ├── AC-CT 📂
│   │   ├── WBACCTiDose2_2001_CT001.dcm 📄
│   │   ├── WBACCTiDose2_2001_CT002.dcm 📄
│   │   ├── ... 🗂️
│   │   └── WBACCTiDose2_2001_CT532.dcm 📄
│   └── AC-PT 📂
│       ├── DetailWB_CTACWBPT001_PT001.dcm 📄
│       ├── DetailWB_CTACWBPT001_PT002.dcm 📄
│       ├── ... 🗂️
│       └── DetailWB_CTACWBPT001_PT532.dcm 📄
├── S2 📂
│   └── CT_S2.nii 📄
├── S3 📂
│   └── CT_S3.nii 📄
├── S4 📂
│   └── S4_ULD_FDG_60m_Dynamic_Patlak_HeadNeckThoAbd_20211025075852_2.nii 📄
└── S5 📂
    └── CT_S5.nii 📄

```
**Note:** If the necessary naming conventions are not followed, MOOSE 3.0 will skip the subjects.

### Naming Conventions for NIFTI files 📝
When using NIFTI files, you should name the file with the appropriate modality as a suffix. 

For instance, if you have chosen the `model_name` as `clin_ct_organs`, the CT scan for subject 'S2' in NIFTI format, should have the modality tag 'CT_' attached to the file name, e.g. `CT_S2.nii`. In the directory shown above, every subject will be processed by `moosez` except S4.

**Remember:** Adhering to these file naming and directory structure conventions ensures smooth and efficient processing with MOOSE 3.0. Happy segmenting! 🚀

## :tada: Add and contribute Your Own nnUNetv2 Models to MooseZ :rocket:

Want to power-up your medical image segmentation tasks? :zap: Join the MooseZ community and contribute your own `nnUNetv2` models! 🥇:

By adding your custom models to MooseZ, you can enjoy:

- :fast_forward: **Increased Speed** - MooseZ is optimized for fast performance. Use it to get your results faster!
- :floppy_disk: **Reduced Memory** - MooseZ is designed to be efficient and lean, so it uses less memory!

So why wait? Make your models fly with MooseZ :airplane:

## How to Contribute Your Model :hammer_and_wrench:

1. **Prepare Your Model** :file_folder:

    Train your model using `nnUNetv2` and get it ready for the big leagues!

2. **Update AVAILABLE_MODELS List** :pencil2:

    Include your model's unique identifier to the `AVAILABLE_MODELS` list in the [resources.py](https://github.com/LalithShiyam/MOOSE/blob/d131a7c88b3d0defd43339c7d788f092a242f59d/moosez/resources.py#L29) file. The model name should follow a specific syntax: 'clin' or 'preclin' (indicating Clinical or Preclinical), modality tag (like 'ct', 'pt', 'mr'), and then the tissue of interest.

3. **Update MODELS Dictionary** :clipboard:

    Add a new entry to the `MODELS` dictionary in the [resources.py](https://github.com/LalithShiyam/MOOSE/blob/d131a7c88b3d0defd43339c7d788f092a242f59d/moosez/resources.py#L49) file. Fill in the corresponding details (like URL, filename, directory, trainer type, voxel spacing, and multilabel prefix). 

4. **Update expected_modality Function** :memo:

    Update the `expected_modality` function in the [resources.py](https://github.com/LalithShiyam/MOOSE/blob/d131a7c88b3d0defd43339c7d788f092a242f59d/moosez/resources.py#L100) file to return the imaging technique, modality, and tissue of interest for your model.

5. **Update map_model_name_to_task_number Function** :world_map:

    Modify the `map_model_name_to_task_number` function in the [resources.py](https://github.com/LalithShiyam/MOOSE/blob/d131a7c88b3d0defd43339c7d788f092a242f59d/moosez/resources.py#L130) file to return the task number associated with your model.

6. **Update `ORGAN_INDICES` in `constants.py`** 🧠

   Append the `ORGAN_INDICES` dictionary in the [constants.py](https://github.com/LalithShiyam/MOOSE/blob/3f5f9537365a41478060c96815c38c3824353bb9/moosez/constants.py#L66C1-L66C14) with your label intensity to region mapping. This is particularly important if you would like to have your stats from the PET images based on your CT masks.

That's it! You've successfully contributed your own model to the MooseZ community! :confetti_ball:

With your contribution 🙋, MooseZ becomes a stronger and more robust tool for medical image segmentation! :muscle:


## A Note on QIMP Python Packages: The 'Z' Factor 📚🚀

All of our Python packages here at QIMP carry a special signature – a distinctive 'Z' at the end of their names. The 'Z' is more than just a letter to us; it's a symbol of our forward-thinking approach and commitment to continuous innovation.

Our MOOSE package, for example, is named as 'moosez', pronounced "moose-see". So, why 'Z'?

Well, in the world of mathematics and science, 'Z' often represents the unknown, the variable that's yet to be discovered, or the final destination in a series. We at QIMP believe in always pushing boundaries, venturing into uncharted territories, and staying on the cutting edge of technology. The 'Z' embodies this philosophy. It represents our constant quest to uncover what lies beyond the known, to explore the undiscovered, and to bring you the future of medical imaging.

Each time you see a 'Z' in one of our package names, be reminded of the spirit of exploration and discovery that drives our work. With QIMP, you're not just installing a package; you're joining us on a journey to the frontiers of medical image processing. Here's to exploring the 'Z' dimension together! 🚀

## 🦌 MOOSE: A part of the [enhance.pet](https://enhance.pet) community

![Alt Text](https://github.com/QIMP-Team/MOOSE/blob/main/Images/Enhance.gif)

## 👥 Contributors

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tbody>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/LalithShiyam"><img src="https://github.com/LalithShiyam.png?s=100" width="100px;" alt="Lalith Kumar Shiyam Sundar"/><br /><sub><b>Lalith Kumar Shiyam Sundar</b></sub></a><br /><a href="https://github.com/ENHANCE-PET/MOOSE/commits?author=LalithShiyam" title="Code">💻</a> <a href="https://github.com/ENHANCE-PET/MOOSE/commits?author=LalithShiyam" title="Documentation">📖</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/Keyn34"><img src="https://github.com/Keyn34.png?s=100" width="100px;" alt="Sebastian Gutschmayer"/><br /><sub><b>Sebastian Gutschmayer</b></sub></a><br /><a href="https://github.com/ENHANCE-PET/MOOSE/commits?author=Keyn34" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/n7k-dobri"><img src="https://avatars.githubusercontent.com/u/114534264?v=4?s=100" width="100px;" alt="n7k-dobri"/><br /><sub><b>n7k-dobri</b></sub></a><br /><a href="https://github.com/ENHANCE-PET/MOOSE/commits?author=n7k-dobri" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/mprires"><img src="https://avatars.githubusercontent.com/u/48754309?v=4?s=100" width="100px;" alt="Manuel Pires"/><br /><sub><b>Manuel Pires</b></sub></a><br /><a href="https://github.com/ENHANCE-PET/MOOSE/commits?author=mprires" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://www.meduniwien.ac.at/web/forschung/researcher-profiles/researcher-profiles/index.php?id=688&res=zacharias_chalampalakis1"><img src="https://avatars.githubusercontent.com/u/62066397?v=4?s=100" width="100px;" alt="Zach Chalampalakis"/><br /><sub><b>Zach Chalampalakis</b></sub></a><br /><a href="https://github.com/ENHANCE-PET/MOOSE/commits?author=zax0s" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/dhaberl"><img src="https://avatars.githubusercontent.com/u/54232863?v=4?s=100" width="100px;" alt="David Haberl"/><br /><sub><b>David Haberl</b></sub></a><br /><a href="https://github.com/ENHANCE-PET/MOOSE/commits?author=dhaberl" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/W7ebere"><img src="https://avatars.githubusercontent.com/u/166598214?v=4?s=100" width="100px;" alt="W7ebere"/><br /><sub><b>W7ebere</b></sub></a><br /><a href="https://github.com/ENHANCE-PET/MOOSE/commits?author=W7ebere" title="Documentation">📖</a></td>
    </tr>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/Kazezaka"><img src="https://avatars.githubusercontent.com/u/29598301?v=4?s=100" width="100px;" alt="Kazezaka"/><br /><sub><b>Kazezaka</b></sub></a><br /><a href="https://github.com/ENHANCE-PET/MOOSE/commits?author=Kazezaka" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://ltetrel.github.io/"><img src="https://avatars.githubusercontent.com/u/37963074?v=4?s=100" width="100px;" alt="Loic Tetrel"/><br /><sub><b>Loic Tetrel @ Kitware</b></sub></a><br /><a href="https://github.com/ENHANCE-PET/MOOSE/commits?author=ltetrel" title="Code">💻</a> <a href="https://github.com/ENHANCE-PET/MOOSE/commits?author=ltetrel" title="Documentation">📖</a></td>
      
    </tr>
  </tbody>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->
