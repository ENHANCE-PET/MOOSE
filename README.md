![Moose-logo](Images/moose.png)
## MOOSE 2.0 🦌- Leaner. Meaner. Stronger 💪
[![Documentation Status](https://img.shields.io/readthedocs/moosez/latest.svg?style=flat-square&logo=read-the-docs&color=CC00FF)](https://moosez.rtfd.io/en/latest/?badge=latest) [![PyPI version](https://img.shields.io/pypi/v/moosez?color=FF1493&style=flat-square&logo=pypi)](https://pypi.org/project/moosez/) [![License: GPL v3](https://img.shields.io/badge/License-GPLv3-red.svg?style=flat-square&logo=gnu&color=FF0000)](https://www.gnu.org/licenses/gpl-3.0) [![Discord](https://img.shields.io/badge/Discord-Chat-blue.svg?style=flat-square&logo=discord&color=0000FF)](https://discord.gg/9uTHYhWCA5) [![Monthly Downloads](https://img.shields.io/pypi/dm/moosez?label=Downloads%20(Monthly)&color=9400D3&style=flat-square&logo=python)](https://pypi.org/project/moosez/) [![Daily Downloads](https://img.shields.io/pypi/dd/moosez?label=Downloads%20(Daily)&color=9400D3&style=flat-square&logo=python)](https://pypi.org/project/moosez/)

Unveiling a new dimension in 3D medical image segmentation: MOOSE 2.0 🚀

Crafted meticulously from the core principles of data-centric AI, MOOSE 2.0 is our response to the demands of both preclinical and clinical imaging.

:sparkles: **It's Leaner**: We've hacked away the fluff and made MOOSE 2.0 leaner than ever before. This bad boy doesn't need heavy-duty computing. With less than 32GB of RAM, compatibility across OS, and the flexibility to work with or without NVIDIA GPUs, MOOSE 2.0 fits right into any environment. :microscope:

:boom: **It's Meaner**: The QIMPies have poured their hearts and souls into building this beast from scratch. With the speed clocking 5x faster than its predecessor, MOOSE 2.0 cuts through the noise and gets down to business instantly. It serves up a range of segmentation models designed for both clinical and preclinical settings. No more waiting, no more compromises. It's Mean Machine time! :zap:

:fire: **It's Stronger**: MOOSE 2.0 is powered by the sheer strength of Data-centric AI principles. With a whopping 1.5k whole-body PET/CT datasets, that's ~40x times more data than our first model, we're packing a punch. MOOSE 2.0 comes with the strength and knowledge gained from an array of data that's simply unparalleled. The result? Better precision, improved outcomes, and a tool you can trust. :briefcase:

:bell: :loudspeaker: :boom: And now, it's even more **versatile**, with MOOSE 2.0, you now have the flexibility to use it as a powerful command-line tool for batch processing, or as a library package for individual processing in your Python projects. The choice is yours! :sunglasses:

Accommodating an array of modalities including PET, CT, and MRI, MOOSE 2.0 stands at the cusp of a paradigm shift. It’s not just an upgrade; it’s our commitment to making MOOSE 2.0 your go-to for segmentation tasks.

Join us as we embark on this journey.


<div align="center">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://github.com/QIMP-Team/MOOSE/blob/main/Images/MOOSE_storyboard_dark.gif">
  <source media="(prefers-color-scheme: light)" srcset="https://github.com/QIMP-Team/MOOSE/blob/main/Images/MOOSE_storyboard_light.gif">
  <img alt="Shows an illustrated MOOSE story board adopted to different themes" src="https://github.com/QIMP-Team/MOOSE/blob/main/Images/MOOSE_storyboard_light.gif">
</picture>
</div>

## Available Segmentation Models 🧬

MOOSE 2.0 offers a wide range of segmentation models catering to various clinical and preclinical needs. Here are the models currently available:

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

Each model is designed to provide high-quality segmentation with MOOSE 2.0's optimized algorithms and data-centric AI principles.

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

Before you dive into the incredible world of MOOSE 2.0, here are a few things you need to ensure for an optimal experience:

- **Operating System**: We've got you covered whether you're on Windows, Mac, or Linux. MOOSE 2.0 has been tested across these platforms to ensure seamless operation.

- **Memory**: MOOSE 2.0 has quite an appetite! Make sure you have at least 32GB of RAM for the smooth running of all tasks.

- **GPU**: If speed is your game, an NVIDIA GPU is the name! MOOSE 2.0 leverages GPU acceleration to deliver results fast. Don't worry if you don't have one, though - it will still work, just at a slower pace.

- **Python**: Ensure that you have Python 3.10 installed on your system. MOOSE 2.0 likes to keep up with the latest, after all!

So, that's it! Make sure you're geared up with these specifications, and you're all set to explore everything MOOSE 2.0 has to offer. 🚀🌐

## Installation Guide 🛠️

Available on Windows, Linux, and MacOS, the installation is as simple as it gets. Follow our step-by-step guide below and set sail on your journey with MOOSE 2.0.

## For Linux (and Intel x86 Mac)🐧

1. First, create a Python environment. You can name it to your liking; for example, 'moose-env'.
   ```bash
   python3 -m venv moose-env
   ```

2. Activate your newly created environment.
   ```bash
   source moose-env/bin/activate  # for Linux
   ```

3. Install MOOSE 2.0.
   ```bash
   pip install moosez
   ```

Voila! You're all set to explore with MOOSE 2.0.

## For Macs powered by Apple Silicon (M series chips with MPS) 🍏

1. First, create a Python environment. You can name it to your liking; for example, 'moose-env'.
   ```bash
   python3.10 -m venv moose-env
   ```

2. Activate your newly created environment.
   ```bash
   source moose-env/bin/activate 
   ```

3. Install MOOSE 2.0 and a special fork of PyTorch (MPS specific). You need to install the MPS specific branch for making MOOSE work with MPS
   ```bash
   pip install moosez
   pip install git+https://github.com/pytorch/pytorch.git@3c61c525694eca0f895bb01fc67c16793226051a
   ```
Now you are ready to use MOOSE on Apple Silicon 🏎⚡️.

## For Windows 🪟

1. Create a Python environment. You could name it 'moose-env', or as you wish.
   ```bash
   python -m venv moose-env
   ```

2. Activate your newly created environment.
   ```bash
   .\moose-env\Scripts\activate
   ```

3. Go to the PyTorch website and install the appropriate PyTorch version for your system. **!DO NOT SKIP THIS!**

4. Finally, install MOOSE 2.0.
   ```bash
   pip install moosez
   ```

There you have it! You're ready to venture into the world of 3D medical image segmentation with MOOSE 2.0.

Happy exploring! 🚀🔬

## Usage Guide 📚

### Command-line tool for batch processing :computer: 

Embarking on your journey with MOOSE 2.0 is straightforward and easy. Our command-line tool for batch processing requires only two arguments: the directory path where your subject images are stored, and the segmentation model name you wish to use. Here's how you can get started:

```bash
moosez -d <path_to_image_dir> -m <model_name>
```

Here `<path_to_image_dir>` refers to the directory containing your subject images and `<model_name>` is the name of the segmentation model you intend to utilize. 

For instance, to perform clinical CT organ segmentation, the command would be:

```bash
moosez -d <path_to_image_dir> -m clin_ct_organs
```

In this example, 'clin_ct_organs' is the segmentation model name for clinical CT organ segmentation.

And that's it! With just one command, you're all set to explore the new horizons of 3D medical image segmentation with MOOSE 2.0.

Need assistance along the way? Don't worry, we've got you covered. Simply type:

```bash
moosez -h
```

This command will provide you with all the help and the information about the available [models](https://github.com/QIMP-Team/MOOSE/blob/3fcfad710df790e29a4a1ea16f22e480f784f38e/moosez/resources.py#L29) and the [regions](https://github.com/QIMP-Team/MOOSE/blob/3fcfad710df790e29a4a1ea16f22e480f784f38e/moosez/constants.py#L64) it segments.

### Using MOOSE 2.0 as a Library :books:

MOOSE 2.0 can also be imported and used as a library in your own Python projects. Here's how you can do it:

First, import the `moose` function from the `moosez` package in your python script:

 ```python
 from moosez import moose
 ```

Then, call the `moose` function to run predictions. The `moose` function takes four arguments:

1. `model_name`: The name of the model to use for the predictions.
2. `input_dir`: The directory containing the images (in nifti, either .nii or .nii.gz) to process.
3. `output_dir`: The directory where the output will be saved.
4. `accelerator`: The type of accelerator to use (e.g., "cpu", "cuda").

Here's an example of how to call the `moose` function:

 ```python
 model_name = 'clin_ct_organs'
 input_dir = '/home/Documents/your_project/data/input'
 output_dir = '/home/Documents/your_project/data/output'
 accelerator = 'cuda'
 moose(model_name, input_dir, output_dir, accelerator)
 ```

Remember to replace `model_name`, `input_dir`, `output_dir`, and `accelerator` with the actual values you want to use.

That's it! MOOSE 2.0 will now process the images in the input directory and save the output in the output directory. Enjoy using MOOSE 2.0 as a library in your Python projects!

## Directory Structure and Naming Conventions for MOOSE 📂🏷️

### Applicable only for batch mode ⚠️

Using MOOSE 2.0 optimally requires your data to be structured according to specific conventions. MOOSE 2.0 supports both DICOM and NIFTI formats. For DICOM files, MOOSE infers the modality from the DICOM tags and checks if the given modality is suitable for the chosen segmentation model. However, for NIFTI files, users need to ensure that the files are named with the correct modality as a suffix.

### Required Directory Structure 🌳
Please structure your dataset as follows:

```
MOOSEv2_data/
├── S1
│   ├── AC-CT
│   │   ├── WBACCTiDose2_2001_CT001.dcm
│   │   ├── WBACCTiDose2_2001_CT002.dcm
│   │   ├── .
│   │   ├── .
│   │   ├── .
│   │   └── WBACCTiDose2_2001_CT532.dcm
│   └── AC-PT
│       ├── DetailWB_CTACWBPT001_PT001.dcm
│       ├── DetailWB_CTACWBPT001_PT002.dcm
│       ├── .
│       ├── .
│       ├── .
│       └── DetailWB_CTACWBPT001_PT532.dcm
├── S2
│   └── CT_S2.nii
├── S3
│   └── CT_S3.nii
├── S4
│   └── S4_ULD_FDG_60m_Dynamic_Patlak_HeadNeckThoAbd_20211025075852_2.nii
└── S5
    └── CT_S5.nii
```
**Note:** If the necessary naming conventions are not followed, MOOSE 2.0 will skip the subjects.

### Naming Conventions for NIFTI files 📝
When using NIFTI files, you should name the file with the appropriate modality as a suffix. 

For instance, if you have chosen the `model_name` as `clin_ct_organs`, the CT scan for subject 'S2' in NIFTI format, should have the modality tag 'CT_' attached to the file name, e.g. `CT_S2.nii`. In the directory shown above, every subject will be processed by `moosez` except S4.

**Remember:** Adhering to these file naming and directory structure conventions ensures smooth and efficient processing with MOOSE 2.0. Happy segmenting! 🚀

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
