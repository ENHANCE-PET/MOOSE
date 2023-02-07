![Moose-logo](Images/Moose-logo.png)

![](https://komarev.com/ghpvc/?username=QIMP-Team&color=blueviolet&style=for-the-badge)[![image](https://img.shields.io/badge/YouTube-FF0000?style=for-the-badge&logo=youtube&logoColor=white)](https://www.youtube.com/playlist?list=PLZQERorVWrbcG4AMkDQ9KrL_Rr77D1-6k) [![image](https://img.shields.io/badge/Discord-5865F2?style=for-the-badge&logo=discord&logoColor=white)](https://discord.gg/9uTHYhWCA5) [![image](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/company/qimp/) [![Share on Twitter](https://img.shields.io/badge/Twitter-share%20on%20twitter-blue?logo=twitter&style=for-the-badge)](https://twitter.com/intent/tweet?text=Check%20out%20MOOSE%20(Multi-organ%20objective%20segmentation%20:https://github.com/QIMP-Team/MOOSE)%20a%20data-centric%20AI%20solution%20that%20generates%20multilabel%20organ%20segmentations%20to%20facilitate%20systemic%20TB%20whole-person%20research.) 


## 🚀 News

### December 20, 2022:

Dear MOOSE users,

We sincerely apologize for the delay in the release of MOOSE v.02. We understand that many of you have been eagerly anticipating this update and we apologize for any inconvenience or frustration this delay may have caused.

The reason for the delay is that we have recently been focused on the release of FALCON, which required a significant allocation of resources and time. As a result, we have not been able to devote as much attention to MOOSE as we would have liked.

We want to assure you that we are working hard to get MOOSE v.02 released as soon as possible and are committed to providing our users with the best possible experience. In the meantime, if you have any questions or concerns, please do not hesitate to reach out to us.

Thank you for your understanding and patience.

Sincerely, 

The MOOSE-dev team

### October 22, 2022:

[MOOSE version 0.1.4](https://github.com/QIMP-Team/MOOSE/releases/tag/moose-v0.1.4): If your previous MOOSE installation suddenly stopped working or if you downloaded MOOSE recently (Oct 1-22, 2022) and it doesn't work, reinstall MOOSE with the latest version (MOOSE v0.1.4). Please reach us out on discord (click the discord label link above), if the error persists, we will be happy to help.

## 🦌 About MOOSE 

<div align="center">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://github.com/QIMP-Team/MOOSE/blob/main/Images/MOOSE_storyboard_dark.gif">
  <source media="(prefers-color-scheme: light)" srcset="https://github.com/QIMP-Team/MOOSE/blob/main/Images/MOOSE_storyboard_light.gif">
  <img alt="Shows an illustrated MOOSE story board adopted to different themes" src="https://github.com/QIMP-Team/MOOSE/blob/main/Images/MOOSE_storyboard_light.gif">
</picture>
</div>


MOOSE (Multi-organ objective segmentation) a [data-centric AI solution](https://datacentricai.org) that generates multilabel organ segmentations to facilitate systemic TB whole-person research. The pipeline is based on [nn-UNet](https://github.com/MIC-DKFZ/nnUNet) and has the capability to segment 120 unique tissue classes from a whole-body 18F-FDG PET/CT image. The input can be an 18F-FDG PET/CT image or a CT only image (but never a PET only image) and the segmentation of the tissues is done automatically based on the provided `DICOM` input. 

As mentioned earlier, MOOSE is built on [data-centric AI principles](https://snorkel.ai/principles-of-data-centric-ai-development/) where the state-of-the-art architecture (`nnUNet` in our case) is fixed and the training data is selectively augmented to ensure peak segmentation performance. The segmentation performance is continously monitored in a systemic manner using the concept of similarity space (refer [manuscript](https://jnm.snmjournals.org/content/early/2022/06/30/jnumed.122.264063.abstract)). Data that causes a decrease in the performance is automatically identified and included to the initial training dataset for maintaining peak performance. ⭐️ us if you like our work!


![Alt Text](https://github.com/QIMP-Team/MOOSE-v0.1.0/blob/main/Images/MOOSE-results.gif)


## ⛔️ Hard requirements 

`MOOSE` has been *ONLY* tested on **Ubuntu linux OS**, with the following hardware capabilities:
- Intel(R) Xeon(R) Silver 4216 CPU @ 2.10GHz 
- 256 GB of RAM (Very important for total-body datasets)
- 1 x Nvidia GeForce RTX 3090 Ti (or similar)
We are testing different configurations now, but the RAM (256 GB) seems to be a hard requirement if you are using whole-body CT datasets with high resolution. 

## ⚙️ Installation

We offer two flavors of installation: 
- Installing in your local machine and 
- Installing in your shared server (e.g. DGX) via docker. 

I personally recommend the docker as you have a fully functional, ready-to-go solution. But honestly, depends on what you like!

### 🖥️ Local machine 

*Include `sudo` as shown below, in case you don't have write access. If you use `sudo`, make sure you type `sudo su` before you run `moose`, check [usage](https://github.com/QIMP-Team/MOOSE#-usage)! If you do have write access, meaning its really your personal server, feel free to ignore the `sudo` part*

#### 📀 Fresh install 

Kindly copy the code below and paste it on your ubuntu terminal, the installer should ideally take care of the rest. A fresh install would approximately take 5-10 minutes. We have also made a [video](https://youtu.be/L448q47Psfc) of how to perform a fresh install of `moose`

##### Step: 1

```bash
sudo git config --global url."https://".insteadOf git://
sudo git clone https://github.com/QIMP-Team/MOOSE.git
cd MOOSE
sudo bash moose_installer.sh
```

##### Step: 2

‼️ Source the .bashrc file again**

```bash
source ~/.bashrc
```

#### 📀 Uninstalling MOOSE

You can uninstall MOOSE by following the steps [here](https://github.com/QIMP-Team/MOOSE#step-1-1) and also there is a [video](https://youtu.be/zYiIWhDHabs) which shows how to perform the uninstallation!

#### ⚠️ NOTE: For people who already have the alpha version of moose in their machines 

If you have already installed `moose` before. You need to uninstall `moose` before installing the current version. This can be easily done by using the command below. 

##### Step: 1

```bash
sudo git config --global url."https://".insteadOf git://
sudo git clone https://github.com/QIMP-Team/MOOSE.git
cd MOOSE
sudo bash moose_uninstaller.sh
```
Once these steps are done, follow the steps below to do a fresh install of `moose`.

##### Step: 2

```bash
sudo bash moose_installer.sh
```
##### Step: 3

‼️ Source the .bashrc file again**

```bash
source ~/.bashrc
```


### 🐋 Installing using Docker 

We have already created the `docker` image for you, all you need to do is load it. We assume that you have already installed docker in your system (solutions to common 'installation and image-loading' issues in docker can be found [here](https://github.com/NVIDIA/nvidia-docker/issues/1243#issuecomment-694981577)). Make sure you replace the `path_to_mount_without_the_quotes` in the last command with your own local path (e.g. `/home/Documents/data-to-moose`) , which needs to be mounted to the container (now your mounted data will be in the container at `/data`). We have made a [video](https://youtu.be/DUg3RLcP25U) regarding the moose docker installation, to give an overview of how it is done.

##### Step: 1

```bash
mkdir moose_dckr
cd moose_dckr
wget "https://moose-files.s3.eu-de.cloud-object-storage.appdomain.cloud/moose_16072022.tar"
docker load < moose_16072022.tar
docker run --gpus all --name moose -it --ipc=host -v 'path_to_mount_without_the_quotes':/data moose:latest /bin/bash
```
After this step, a docker container with the name 'moose' will be created. In case you exited the ```moose``` container, you can start and run the container using the following commands mentioned in step 2!

##### Step: 2

```bash
docker start moose
docker attach moose
```
You will now be inside the moose container after the execution of the ```docker attach moose``` command. Kindly refer the [usage](https://github.com/QIMP-Team/MOOSE#-usage) section for running ```moose``` on your datasets. You don't need to `sudo su` before you run `moose`, if you are using it via `docker`!

If you have troubles with the installation, you can reach us via [discord](https://discord.gg/m3pjREWQ)!

## 🗂 Required folder structure 

`MOOSE` inherently performs batchwise analysis. Once you have all the patients to be analysed in a main directory, the analysis is performed sequentially. The output folders that will be created by the script itself are highlighted with the tag "Auto-generated" (refer results section). Organising the folder structure is the sole responsibility of the user. Also closely monitor the moose.log file for finding out more about the workflow of MOOSE. All the labels are stored under the 'labels' folder of each subject. 

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

For running the moose directly from the command-line terminal using the default options - please use the following command. In general, MOOSE performs the error analysis (refer paper) in similarity space and assumes that the given (if given) PET image is static.

```bash

#syntax:
moose -f path_to_main_folder 

```
#### Local machine

```bash

sudo su # In your terminal before you run moose, if you installed earlier with sudo in your local machine.

#example: 
moose -f /home/kyloren/Documents/main_folder # input can be absolute path
                    or 
moose -f /main_folder # or relative path

```
#### Usage via docker

After you start the `moose` docker container, you can use the command below.

```bash

#example:
moose -f '/data/main_folder' # always absolute path

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

#### 🤔 Legends for the generated nifti labels

The label to region correspondence can be found [here](https://github.com/QIMP-Team/MOOSE/blob/main/labels-to-regions.md)!

- Unified labels: 
  - `MOOSE-Non-cerebral-tissues-CT-SUB0XX-XXXX.nii.gz:` Multilabel non-cerebral CT segmentations from the subject's CT.
  - `MOOSE-unified-PET-CT-atlas.nii.gz:` Multilabel MOOSE atlas which should ideally contain all the cerebral (from PT) and non-cerebral tissues (from CT)

- Compartmental labels:
  - `Bones_XXXXX.nii.gz:` Multilabel bone segmentations from the subject's CT.
  - `Fat-Muscle_XXXXX.nii.gz:` Multilabel fat-muscle (skeletal muscle, subcutaneous and visceral fat) segmentations from the subject's CT.
  - `Organs_XXXXX.nii.gz:` Multilabel abdominal organ segmentations from the subject's CT.
  - `Psoas_XXXXX.nii.gz:` Psoas muscle segmentations from the subject's CT.
  - `Brain_XXXXX.nii.gz:` Multilabel hammersmith atlas segmentations from the subject's PT.

- Individual labels: 
  All individual labels can be found inside `SUB0XX/MOOSE-SUB0XX/labels/sim_space/similarity-space/`. Each label is a binary mask with their actual region    name. E.g. 'Aorta' would be named as `Aorta.nii.gz`.

#### 🤔 Statistical measures

`MOOSE` also derives statistical measures (Mean, Median, Standard-deviation, Maximum and Minimum) based on the segmentations and they are stored in `MOOSE-SUB0XX/stats` as `.csv` files.

- `XXXXX-ct-hu-values.csv:` Hounsfield values of the regions derived from CT 
- `XXXXX-ct-volume-stats.csv:` Volume of the regions derived from CT
- `XXXXX-SUV-values.csv:` SUV values of the regions derived from PT

#### 🤔 Segmentation accuracy report

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
## 🎦 Videos

**MOOSE installation/uninstallation:**

- [Installation in your local workstation](https://youtu.be/L448q47Psfc)
- [Uninstallation in your local workstation](https://youtu.be/zYiIWhDHabs)
- [Installation via docker image](https://youtu.be/DUg3RLcP25U)


## 🙏 Acknowledgement

This research was supported by:
- [IBM University Cloud Award](https://www.research.ibm.com/university/)
- [National Center for High-performance Computing, Taiwan](https://www.nchc.org.tw/Page?itemid=58&mid=109)

## <img src="https://github.com/QIMP-Team/MOOSE/blob/main/Images/github.png" width="35"> GitHub Sponsors

### [Hermes Medical Solutions](https://github.com/HermesMedicalSolutions)

<picture>
  <source media="(prefers-color-scheme: dark)" srcset=https://github.com/QIMP-Team/MOOSE/blob/main/Images/HMS_RGB_White.svg width="300">
  <source media="(prefers-color-scheme: light)" srcset=https://github.com/QIMP-Team/MOOSE/blob/main/Images/HMS_RGB_Blue.svg width="300">
  <img alt="Shows an HMS logo according to gh theme." src=https://github.com/QIMP-Team/MOOSE/blob/main/Images/HMS_RGB_Blue.svg width="300">
</picture>

## 🙋 FAQ

**[1]** Will MOOSE only work on whole-body 18F-FDG PET/CT datasets?

  *MOOSE ideally works on whole-body (head to toe) PET/CT datasets, but also works on semi whole-body PET/CT datasets (head to pelvis). We have also tested it on abdominal CT as well as thorax/chest CT. MOOSE will ideally work, if the test datasets have sufficient quality and not too different from the training datasets.


**[2]** Will MOOSE only work on multimodal 18F-FDG PET/CT datasets or can it also be applied to CT only? or PET only?

 *MOOSE automatically infers the modality type using the DICOM header tags. MOOSE builds the entire atlas with 120 tissues if the user provides multimodal 18F-FDG PET/CT datasets. The user can also provide CT only DICOM folder, MOOSE will infer the modality type and segment only the non-cerebral tissues (36/120 tissues) and will not segment the 83 subregions of the brain. MOOSE will definitely not work if only provided with 18F-FDG PET images.*


**[3]** Will MOOSE work on non-DICOM formats?

 *Unfortunately the current version accepts only DICOM formats. In the future, we will try to enable non-DICOM formats for processing as well.*


## 🛠 To do 

**MOOSEv0.1.0: July release candidate**

- [x] Create a working `moose_uninstaller.sh` [@LalithShiyam](https://github.com/LalithShiyam)
- [x] Create a docker image (`moose_16072022.tar`) for the current version of moose v0.1.0 [@LalithShiyam](https://github.com/LalithShiyam)

**MOOSEv0.2.0: Feb 25, 2023 release candidate** 

- [ ] Enable `moose` to accept non-dicom inputs (e.g. nifti/analyze/mha)[@LalithShiyam](https://github.com/LalithShiyam)
- [ ] Allow users to select the choose segmentation compartments (Organs, Bones, Fat-muscle, Brain, Psoas)[@LalithShiyam](https://github.com/LalithShiyam)
- [ ] Prune/Compress the models for faster inference: (PRs welcome)[@davidiommi](https://github.com/davidiommi) 
- [ ] Reduce memory requirement (No more 256 GB, ideally 32 GB) for MOOSE during inference: (PRs welcome) [@dhaberl](https://github.com/dhaberl)[@Keyn34](https://github.com/Keyn34)

## 🦌 MOOSE: An ENHANCE-PET Project

![Alt Text](https://github.com/QIMP-Team/MOOSE/blob/main/Images/DALL·E%202022-11-01%2018.13.35%20-%20a%20moose%20with%20majestic%20horns.png)
<p align="right">Above image generated by dall-e</p>
