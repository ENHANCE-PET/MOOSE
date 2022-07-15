![Moose-logo](Images/Moose-logo.png)

[![image](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/company/qimp/) [![image](https://img.shields.io/badge/Discord-5865F2?style=for-the-badge&logo=discord&logoColor=white)](https://discord.gg/hnrWbvPc) [![Share on Twitter](https://img.shields.io/badge/Twitter-share%20on%20twitter-blue?logo=twitter&style=for-the-badge)](https://twitter.com/intent/tweet?text=Check%20out%20MOOSE%20(Multi-organ%20objective%20segmentation%20:https://github.com/QIMP-Team/MOOSE)%20a%20data-centric%20AI%20solution%20that%20generates%20multilabel%20organ%20segmentations%20to%20facilitate%20systemic%20TB%20whole-person%20research.) 

## 🦌 About MOOSE 

<p align="center">
  <source media="(prefers-color-scheme: dark)" srcset="https://github.com/LalithShiyam/MOOSE-V.1.0/blob/main/Images/MOOSE_storyboard_dark.gif" width="500" height="500">
  <img alt="Shows an illustrated storyboard in light color mode and an inverted storyboard in dark color mode." src="https://github.com/LalithShiyam/MOOSE-V.1.0/blob/main/Images/MOOSE_storyboard_dark.gif" width="500" height="500">
</p>

MOOSE (Multi-organ objective segmentation) a data-centric AI solution that generates multilabel organ segmentations to facilitate systemic TB whole-person research.The pipeline is based on nn-UNet and has the capability to segment 120 unique tissue classes from a whole-body 18F-FDG PET/CT image. The input can be an 18F-FDG PET/CT image or a CT only image (but never a PET only image) and the segmentation of the tissues is done automatically based on the provided input.


![Alt Text](https://github.com/QIMP-Team/MOOSE-v0.1.0/blob/main/Images/MOOSE-results.gif)


## ⛔️ Hard requirements 

`MOOSE` has been *ONLY* tested on **Ubuntu linux OS**, with the following hardware capabilities:
- Intel(R) Xeon(R) Silver 4216 CPU @ 2.10GHz 
- 256 GB of RAM (Very important for total-body datasets)
- 1 x Nvidia GeForce RTX 3090 Ti (or similar)
We are testing different configurations now, but the RAM (256 GB) seems to be a hard requirement if you are using whole-body CT datasets with high resolution. 

## ⚙️ Installation

### ‼️ ⚠️ NOTE: For people who already have the alpha version of moose in their servers 

If you have already installed `moose` before. You need to uninstall `moose` before installing the current version. This can be easily done by using the command below

#### Step: 1

```bash
sudo git config --global url."https://".insteadOf git://
git clone https://github.com/QIMP-Team/MOOSE-v0.1.0.git
cd MOOSE-v0.1.0
sudo bash moose_uninstaller.sh
```
Once these steps are done, follow the steps below to do a fresh install of `moose`.

#### Step: 2

```bash
sudo bash moose_installer.sh
```
#### Step: 3

‼️ Source the .bashrc file again**

```bash
source ~/.bashrc
```

### 📀 Fresh install 

Kindly copy the code below and paste it on your ubuntu terminal, the installer should ideally take care of the rest. A fresh install would approximately take 10 minutes.

#### Step: 1

```bash
sudo git config --global url."https://".insteadOf git://
git clone https://github.com/QIMP-Team/MOOSE-v0.1.0.git
cd MOOSE-v0.1.0
sudo bash moose_installer.sh
```

#### Step: 2

‼️ Source the .bashrc file again**

```bash
source ~/.bashrc
```

## 🗂 Required folder structure 

`moose` inherently performs batchwise analysis. Once you have all the patients to be analysed in a main directory, MOOSE performs the analysis sequentially. The output folders that will be created by the script itself are highlighted with the tag "Auto-generated" (refer results section). Organising the folder structure is the sole responsibility of the user. Also closely monitor the moose.log file for finding out more about the workflow of MOOSE. All the labels are stored under the 'labels' folder of each subject. 

```bash

main_folder/                         # The mother folder that holds all the patient folders (folder name can be anything)
├── SUB01                            # Individual patient folder (folder name can be anything)  
│   ├── AC_CT                        # Required: The CT folder name can be named anything as long as the files inside this folder is DICOM 
│   └── PET_WB                       # Required: The PT folder name can be named anything as long as the files inside this folder is DICOM          
└── SUB02
│   ├── AC_CT_1.2.752.37.47.345051852996.20220311.1441.5.430761
│   └── PET_WB_CORRECTED_1.2.752.37.47.345051852996.20220311.1441.5.430763
└── .
└── .
└── .
└── SUB0N
    ├── AC_CT_1.2.752.37.47.345051852996.20220311.1441.5.430761
    └── PET_WB_CORRECTED_1.2.752.37.47.345051852996.20220311.1441.5.430763    
```

## 🖥 Usage

- For running the moose directly from the command-line terminal using the default options - please use the following command. In general, MOOSE performs the error analysis (refer paper) in similarity space and assumes that the given (if given) PET image is static.

```bash

#syntax:
moose -f path_to_main_folder 

#example: 
moose -f '/home/kyloren/Documents/main_folder'

```
## 📈 Results

After the analysis the following folders would be created.

```bash

main_folder/                         # The mother folder that holds all the patient folders (folder name can be anything)
├── SUB01                            # Individual patient folder (folder name can be anything)  
│   ├── AC_CT                        # Required: The CT folder name can be named anything as long as the files inside this folder is DICOM 
│   ├── MOOSE-SUB01                  # Auto-generated: All the files generated by MOOSE will be stored here
│   │   ├── CT                       # Auto-generated: The NIFTI CT file derived from the DICOM images will be stored here 
│   │   ├── labels                   # Auto-generated: All the generated labels will be stored here
│   │   │   └── sim_space            
│   │   │       └── similarity-space # Auto-generated: All the files generated during the error analysis  will be stored here
│   │   ├── PT                       # Auto-generated: The NIFTI PT file dereived from DICOM images will be stored here
│   │   └── temp                     # Auto-generated: Temporary folder for house-keeping                 
│   └── PET_WB                       # Required: The PT folder name can be named anything as long as the files inside this folder is DICOM          
└── SUB02
    ├── AC_CT_1.2.752.37.47.345051852996.20220311.1441.5.430761
    ├── MOOSE-SUB02
    │   ├── CT
    │   ├── labels
    │   │   └── sim_space
    │   │       └── similarity-space
    │   ├── PT
    │   └── temp
    └── PET_WB_CORRECTED_1.2.752.37.47.345051852996.20220311.1441.5.430763
```
The generated labels are currently in `nifti` format and for each subject `SUB0XX`, the labels will be stored in `SUB0XX/MOOSE-SUB0XX/labels`.

**🤔 Legends for the generated nifti labels** 

The label to region correspondence can be found [here](https://github.com/QIMP-Team/MOOSE/blob/main/labels-to-regions.md)!

[1] Unified labels: 
- `MOOSE-Non-cerebral-tissues-CT-SUB0XX-XXXX.nii.gz:` Multilabel non-cerebral CT segmentations from the subject's CT.
- `MOOSE-unified-PET-CT-atlas.nii.gz:` Multilabel MOOSE atlas which should ideally contain all the cerebral (from PT) and non-cerebral tissues (from CT)

[2] Compartmental labels:
- `Bones_XXXXX.nii.gz:` Multilabel bone segmentations from the subject's CT.
- `Fat-Muscle_XXXXX.nii.gz:` Multilabel fat-muscle (skeletal muscle, subcutaneous and visceral fat) segmentations from the subject's CT.
- `Organs_XXXXX.nii.gz:` Multilabel abdominal organ segmentations from the subject's CT.
- `Psoas_XXXXX.nii.gz:` Psoas muscle segmentations from the subject's CT.
- `Brain_XXXXX.nii.gz:` Multilabel hammersmith atlas segmentations from the subject's PT.

[3] Individual labels: 
All individual labels can be found inside `SUB0XX/MOOSE-SUB0XX/labels/sim_space/similarity-space/`. Each label is a binary mask with their actual region name. E.g. 'Aorta' would be named as `Aorta.nii.gz`.

**🤔 Statistical measures**

`MOOSE` also derives statistical measures (Mean, Median, Standard-deviation, Maximum and Minimum) based on the segmentations and they are stored in `MOOSE-SUB0XX/stats` as `.csv` files.

- `XXXXX-ct-hu-values.csv:` Hounsfield values of the regions derived from CT 
- `XXXXX-ct-volume-stats.csv:` Volume of the regions derived from CT
- `XXXXX-SUV-values.csv:` SUV values of the regions derived from PT

**🤔 Segmentation accuracy report**

- `XXXXX-Risk-of-Segmentation-error.csv:` An automatically generated report, which highlights the risk of the segmentation errors for each region with the tag `high` or `low`. High indicates that the chance of the segmentation being erroenous is high and low indicates vice-versa.

## 📖 Citations

`MOOSE` is built on top of some amazing open-source libraries, please consider citing them as well as a token of appreciation.

`MOOSE`
```
Shiyam Sundar LK, Yu J, Muzik O, et al. Fully-automated, semantic segmentation of whole-body 18F-FDG PET/CT images based on data-centric artificial intelligence. J Nucl Med. June 2022.
```
`nnUNet`
```
Isensee, F., Jaeger, P. F., Kohl, S. A., Petersen, J., & Maier-Hein, K. H. (2020). nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation. Nature Methods, 1-9.
```
`dcm2niix`
```
Li X, Morgan PS, Ashburner J, Smith J, Rorden C. (2016) The first step for neuroimaging data analysis: DICOM to NIfTI conversion. J Neurosci Methods. 264:47-56.
```
`SimpleITK`
```
Z. Yaniv, B. C. Lowekamp, H. J. Johnson, R. Beare, “SimpleITK Image-Analysis Notebooks: a Collaborative Environment for Education and Reproducible Research”, J Digit Imaging., doi: 10.1007/s10278-017-0037-8, 31(3): 290-303, 2018.
```

## 🙋 FAQ

**[1]** Will MOOSE only work on whole-body 18F-FDG PET/CT datasets?

  *MOOSE ideally works on whole-body (head to toe) PET/CT datasets, but also works on semi whole-body PET/CT datasets (head to pelvis). Unfortunately, we haven't tested other field-of-views. We will post the evaluations soon.*


**[2]** Will MOOSE only work on multimodal 18F-FDG PET/CT datasets or can it also be applied to CT only? or PET only?

 *MOOSE automatically infers the modality type using the DICOM header tags. MOOSE builds the entire atlas with 120 tissues if the user provides multimodal 18F-FDG PET/CT datasets. The user can also provide CT only DICOM folder, MOOSE will infer the modality type and segment only the non-cerebral tissues (36/120 tissues) and will not segment the 83 subregions of the brain. MOOSE will definitely not work if only provided with 18F-FDG PET images.*


**[3]** Will MOOSE work on non-DICOM formats?

 *Unfortunately the current version accepts only DICOM formats. In the future, we will try to enable non-DICOM formats for processing as well.*


## 🙏 Acknowledgement

This research is supported through an [IBM University Cloud Award](https://www.research.ibm.com/university/)


## 🛠 To do 

**MOOSEv0.2.0: October release candidate**

- [x] Create a working `moose_uninstaller.sh `
- [ ] Create a docker image for the current version of moose v0.1.0
- [ ] Enable `moose` to accept non-dicom inputs (e.g. nifti/analyze/mha)
- [ ] Allow users to select the choose segmentation compartments (Organs, Bones, Fat-muscle, Brain, Psoas)
- [ ] Prune/Compress the models for faster inference. (PRs welcome)[@davidiommi] 
- [ ] Choose a faster resampling scheme for faster inference. (PRs welcome) [@davidiommi]
- [ ] Reduce memory requirement for MOOSE during inference. (PRs welcome) [@davidiommi]


