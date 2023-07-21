![Moose-logo](Images/Moose-logo-new-2.png)

![](https://komarev.com/ghpvc/?username=QIMP-Team&color=blueviolet&style=for-the-badge)[![image](https://img.shields.io/badge/YouTube-FF0000?style=for-the-badge&logo=youtube&logoColor=white)](https://www.youtube.com/playlist?list=PLZQERorVWrbcG4AMkDQ9KrL_Rr77D1-6k) [![image](https://img.shields.io/badge/Discord-5865F2?style=for-the-badge&logo=discord&logoColor=white)](https://discord.gg/9uTHYhWCA5) [![image](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/company/qimp/) [![Share on Twitter](https://img.shields.io/badge/Twitter-share%20on%20twitter-blue?logo=twitter&style=for-the-badge)](https://twitter.com/intent/tweet?text=Check%20out%20MOOSE%20(Multi-organ%20objective%20segmentation%20:https://github.com/QIMP-Team/MOOSE)%20a%20data-centric%20AI%20solution%20that%20generates%20multilabel%20organ%20segmentations%20to%20facilitate%20systemic%20TB%20whole-person%20research.) 


## MOOSE 2.0 🫎 - Leaner. Meaner. Stronger 💪

Unveiling a new dimension in 3D medical image segmentation: MOOSE 2.0 🚀

Crafted meticulously from the core principles of data-centric AI, MOOSE 2.0 is our response to the demands of both preclinical and clinical imaging. 

✨ It's Leaner: We've hacked away the fluff and made MOOSE 2.0 leaner than ever before. This bad boy doesn't need heavy-duty computing. With less than 32GB of RAM, compatibility across OS, and the flexibility to work with or without NVIDIA GPUs, MOOSE 2.0 fits right into any environment. 🔬

💥 It's Meaner: The QIMPies have poured their hearts and souls into building this beast from scratch. With the speed clocking 5x faster than its predecessor, MOOSE 2.0 cuts through the noise and gets down to business instantly. It serves up a range of segmentation models designed for both clinical and preclinical settings. No more waiting, no more compromises. It's Mean Machine time! ⚡

🔥 It's Stronger: MOOSE 2.0 is powered by the sheer strength of Data-centric AI principles. With a whopping 2.5k datasets, that's ~60x times more data than our first model, we're packing a punch. MOOSE 2.0 comes with the strength and knowledge gained from an array of data that's simply unparalleled. The result? Better precision, improved outcomes, and a tool you can trust. 💼

Accommodating an array of modalities including PET, CT, and MRI, MOOSE 2.0 stands at the cusp of a paradigm shift. It’s not just an upgrade; it’s our commitment to making MOOSE 2.0 your go-to for segmentation tasks.

Join us as we embark on this journey.

## Requirements ✅

Before you dive into the incredible world of MOOSE 2.0, here are a few things you need to ensure for an optimal experience:

- **Operating System**: We've got you covered whether you're on Windows, Mac, or Linux. MOOSE 2.0 has been tested across these platforms to ensure seamless operation.

- **Memory**: MOOSE 2.0 has quite an appetite! Make sure you have at least 32GB of RAM for the smooth running of all tasks.

- **GPU**: If speed is your game, an NVIDIA GPU is the name! MOOSE 2.0 leverages GPU acceleration to deliver results fast. Don't worry if you don't have one, though - it will still work, just at a slower pace.

- **Python**: Ensure that you have Python 3.9 or above installed on your system. MOOSE 2.0 likes to keep up with the latest, after all!

So, that's it! Make sure you're geared up with these specifications, and you're all set to explore everything MOOSE 2.0 has to offer. 🚀🌐

## Installation Guide 🛠️

Available on Windows, Linux, and MacOS, the installation is as simple as it gets. Follow our step-by-step guide below and set sail on your journey with MOOSE 2.0.

## For Linux and MacOS 🐧🍏

1. First, create a Python environment. You can name it to your liking; for example, 'moose-env'.
   ```bash
   python3 -m venv moose-env
   ```

2. Activate your newly created environment.
   ```bash
   source moose-env/bin/activate  # for Linux
   source moose-env/bin/activate  # for MacOS
   ```

3. Install MOOSE 2.0.
   ```bash
   pip install moosez
   ```

Voila! You're all set to explore with MOOSE 2.0.

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

Embarking on your journey with MOOSE 2.0 is straightforward and easy. Our command-line tool requires only two arguments: the directory path where your subject images are stored, and the segmentation model name you wish to use. Here's how you can get started:

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

This command will provide you with all the help and additional information you might need.

## Directory Structure and Naming Conventions for MOOSE 📂🏷️

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

## A Note on QIMP Python Packages: The 'Z' Factor 📚🚀

All of our Python packages here at QIMP carry a special signature – a distinctive 'Z' at the end of their names. The 'Z' is more than just a letter to us; it's a symbol of our forward-thinking approach and commitment to continuous innovation.

Our MOOSE package, for example, is named as 'moosez', pronounced "moose-see". So, why 'Z'?

Well, in the world of mathematics and science, 'Z' often represents the unknown, the variable that's yet to be discovered, or the final destination in a series. We at QIMP believe in always pushing boundaries, venturing into uncharted territories, and staying on the cutting edge of technology. The 'Z' embodies this philosophy. It represents our constant quest to uncover what lies beyond the known, to explore the undiscovered, and to bring you the future of medical imaging.

Each time you see a 'Z' in one of our package names, be reminded of the spirit of exploration and discovery that drives our work. With QIMP, you're not just installing a package; you're joining us on a journey to the frontiers of medical image processing. Here's to exploring the 'Z' dimension together! 🚀

## 🦌 MOOSE: An ENHANCE-PET Project

![Alt Text](https://github.com/QIMP-Team/MOOSE/blob/main/Images/DALL·E%202022-11-01%2018.13.35%20-%20a%20moose%20with%20majestic%20horns.png)
<p align="right">Above image generated by dall-e</p>
