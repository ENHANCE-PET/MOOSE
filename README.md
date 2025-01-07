![Moose-logo](Images/moose.png)

## MOOSE 3.0 🦌- Furiously Fast. Brutally Efficient. Unmatched Precision. 💪


[![Documentation Status](https://img.shields.io/readthedocs/moosez/latest.svg?style=flat-square&logo=read-the-docs&color=CC00FF)](https://moosez.rtfd.io/en/latest/?badge=latest) [![PyPI version](https://img.shields.io/pypi/v/moosez?color=FF1493&style=flat-square&logo=pypi)](https://pypi.org/project/moosez/)
[![Code License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg?style=flat-square&logo=apache&color=blue)](https://www.apache.org/licenses/LICENSE-2.0) [![Model License: CC BY 4.0](https://img.shields.io/badge/Model%20License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/) [![Discord](https://img.shields.io/badge/Discord-Chat-blue.svg?style=flat-square&logo=discord&color=0000FF)](https://discord.gg/9uTHYhWCA5) [![Monthly Downloads](https://img.shields.io/pypi/dm/moosez?label=Downloads%20(Monthly)&color=9400D3&style=flat-square&logo=python)](https://pypi.org/project/moosez/) [![Daily Downloads](https://img.shields.io/pypi/dd/moosez?label=Downloads%20(Daily)&color=9400D3&style=flat-square&logo=python)](https://pypi.org/project/moosez/)

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

https://github.com/user-attachments/assets/b121a9f5-30b6-4a40-a451-6bad6570eb55

## Available Segmentation Models 🧬

MOOSE 3.0 offers a wide range of segmentation models catering to various clinical and preclinical needs. Here are the models currently available:

### Clinical 👫🏽

| **Model Name**        | **Intensities and Regions**                                                                                                                                                                                                                           |
|-----------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `clin_ct_body`        | 1:Legs, 2:Body, 3:Head, 4:Arms                                                                                                                                                                                                                        |
| `clin_ct_cardiac`     | 1: heart_myocardium, 2: heart_atrium_left, 3: heart_atrium_right, 4: heart_ventricle_left, 5: heart_ventricle_right, 6: aorta, 7: iliac_artery_left, 8: iliac_artery_right, 9: iliac_vena_left, 10: iliac_vena_right, 11: inferior_vena_cava, 12: portal_splenic_vein, 13: pulmonary_artery|
| `clin_ct_digestive`   | 1: colon, 2: duodenum, 3: esophagus, 4: small_bowel                                                                                                                                                                  |                                                                                      
| `clin_ct_lungs`       | 1:lung_upper_lobe_left, 2:lung_lower_lobe_left, 3:lung_upper_lobe_right, 4:lung_middle_lobe_right, 5:lung_lower_lobe_right                                                                                                                             |
| `clin_ct_muscles`     | 1: autochthon_left, 2: autochthon_right, 3: gluteus_maximus_left, 4: gluteus_maximus_right, 5: gluteus_medius_left, 6: gluteus_medius_right, 7: gluteus_minimus_left, 8: gluteus_minimus_right, 9: iliopsoas_left, 10: iliopsoas_right                          |
| `clin_ct_organs`      | 1: adrenal_gland_left, 2: adrenal_gland_right, 3: bladder, 4: brain, 5: gallbladder, 6: kidney_left, 7: kidney_right, 8: liver, 9: lung_lower_lobe_left, 10: lung_lower_lobe_right, 11: lung_middle_lobe_right, 12: lung_upper_lobe_left, 13: lung_upper_lobe_right, 14: pancreas, 15: spleen, 16: stomach, 17: thyroid_left, 18: thyroid_right, 19: trachea |
| `clin_ct_peripheral_bones` | 1: carpal_left, 2: carpal_right, 3: clavicle_left, 4: clavicle_right, 5: femur_left, 6: femur_right, 7: fibula_left, 8: fibula_right, 9: fingers_left, 10: fingers_right, 11: humerus_left, 12: humerus_right, 13: metacarpal_left, 14: metacarpal_right, 15: metatarsal_left, 16: metatarsal_right, 17: patella_left, 18: patella_right, 19: radius_left, 20: radius_right, 21: scapula_left, 22: scapula_right, 23: skull, 24: tarsal_left, 25: tarsal_right, 26: tibia_left, 27: tibia_right, 28: toes_left, 29: toes_right, 30: ulna_left, 31: ulna_right |
| `clin_ct_ribs`        | 1: rib_left_1, 2: rib_left_2, 3: rib_left_3, 4: rib_left_4, 5: rib_left_5, 6: rib_left_6, 7: rib_left_7, 8: rib_left_8, 9: rib_left_9, 10: rib_left_10, 11: rib_left_11, 12: rib_left_12, 13: rib_left_13, 14: rib_right_1, 15: rib_right_2, 16: rib_right_3, 17: rib_right_4, 18: rib_right_5, 19: rib_right_6, 20: rib_right_7, 21: rib_right_8, 22: rib_right_9, 23: rib_right_10, 24: rib_right_11, 25: rib_right_12, 26: rib_right_13, 27: sternum |
| `clin_ct_vertebrae`   | 1: vertebra_C1, 2: vertebra_C2, 3: vertebra_C3, 4: vertebra_C4, 5: vertebra_C5, 6: vertebra_C6, 7: vertebra_C7, 8: vertebra_T1, 9: vertebra_T2, 10: vertebra_T3, 11: vertebra_T4, 12: vertebra_T5, 13: vertebra_T6, 14: vertebra_T7, 15: vertebra_T8, 16: vertebra_T9, 17: vertebra_T10, 18: vertebra_T11, 19: vertebra_T12, 20: vertebra_L1, 21: vertebra_L2, 22: vertebra_L3, 23: vertebra_L4, 24: vertebra_L5, 25: vertebra_L6, 26: hip_left, 27: hip_right, 28: sacrum |
| `clin_ct_body_composition`   | 1: skeletal_muscle, 2: subcutaneous_fat, 3: visceral_fat |

### Preclinical 🐁
| **Model Name**        | **Intensities and Regions**                                                                                                                                                                                                                           |
|-----------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `preclin_ct_legs`     | 1:right_leg_muscle, 2:left_leg_muscle                                                                                                                                                                                                                |
| `preclin_mr_all`      | 1:Brain, 2:Liver, 3:Intestines, 4:Pancreas, 5:Thyroid, 6:Spleen, 7:Bladder, 8:OuterKidney, 9:InnerKidney, 10:HeartInside, 11:HeartOutside, 12:WAT Subcutaneous, 13:WAT Visceral, 14:BAT, 15:Muscle TF, 16:Muscle TB, 17:Muscle BB, 18:Muscle BF, 19:Aorta, 20:Lung, 21:Stomach |


Each model is designed to provide high-quality segmentation with MOOSE 3.0's optimized algorithms and data-centric AI principles.

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
    
#### Usage of `moose()` in your code
To use the `moose()` function, ensure that you wrap the function call within a main guard to prevent recursive process creation errors:
```python
from moosez import moose

if __name__ == '__main__':
    input_file = '/path/to/input/file'
    models = ['clin_ct_organs', 'clin_ct_ribs']
    output_directory = '/path/to/save/output'
    accelerator = 'cuda'
    moose(input_file, models, output_directory, accelerator)
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
      <td align="center" valign="top" width="14.28%"><a href="https://ltetrel.github.io/"><img src="https://avatars.githubusercontent.com/u/37963074?v=4?s=100" width="100px;" alt="Loic Tetrel"/><br /><sub><b>Loic Tetrel</b></sub></a><br /><a href="https://github.com/ENHANCE-PET/MOOSE/commits?author=ltetrel" title="Code">💻</a> <a href="https://github.com/ENHANCE-PET/MOOSE/commits?author=ltetrel" title="Documentation">📖</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://www.kitware.com"><img src="https://avatars.githubusercontent.com/u/87549?v=4?s=100" width="100px;" alt="Kitware, Inc."/><br /><sub><b>Kitware, Inc.</b></sub></a><br /><a href="https://github.com/ENHANCE-PET/MOOSE/commits?author=Kitware" title="Code">💻</a> <a href="https://github.com/ENHANCE-PET/MOOSE/commits?author=Kitware" title="Documentation">📖</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://khoanguyen.me"><img src="https://avatars.githubusercontent.com/u/3049054?v=4?s=100" width="100px;" alt="Khoa Nguyen"/><br /><sub><b>Khoa Nguyen</b></sub></a><br /><a href="https://github.com/ENHANCE-PET/MOOSE/commits?author=thangngoc89" title="Code">💻</a></td>
    </tr>
  </tbody>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->
