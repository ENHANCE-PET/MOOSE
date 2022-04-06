![Moose-logo](Images/Moose-logo.png)

## 🦌 About MOOSE 

MOOSE (Multi-organ objective segmentation) a data-centric AI solution that generates multilabel organ segmentations to facilitate systemic TB whole-person research.The pipeline is based on nn-UNet and has the capability to segment 120 unique tissue classes from a whole-body 18F-FDG PET/CT image.

## 🗂 Required folder structure 

MOOSE inherently performs batchwise analysis. Once you have all the patients to be analysed in a main directory, MOOSE performs the analysis sequentially. The output folders that will be created by the script itself are highlighted using CAPS. Organising the folder structure is the sole responsibility of the user. 

```bash
├── main_folder                     # The mother folder that holds all the patient folders (folder name can be anything)
│   ├── patient_folder_1            # Individual patient folder (folder name can be anything)
│       ├── fdgpet                  # The PET folder name can be named anything as long as the files inside this folder is DICOM and has a modality tag.
│       ├── ct                      # The CT folder name can be named anything as long as the files inside this folder is DICOM and has a modality tag.
│       ├── INFERENCE               # Auto-generated 
│       ├── MOOSE-TEMP              # Auto-generated
│       ├── LABELS                  # Auto-generated: contains all the generated labels.
│       ├── CT-NIFTI                # Auto-generated 
│       ├── PT-NIFTI                # Auto-generated
│       ├── RISK-ANALYSIS-XXX.xlsx  # Auto-generated: contains the risk-of-error analysis.
├── patient_folder_2    
│       ├── fdgpet                  # The PET folder name can be named anything as long as the files inside this folder is DICOM and has a modality tag.
│       ├── ct                      # The CT folder name can be named anything as long as the files inside this folder is DICOM and has a modality tag.
│       ├── INFERENCE               # Auto-generated 
│       ├── MOOSE-TEMP              # Auto-generated
│       ├── LABELS                  # Auto-generated: contains all the generated labels.
│       ├── CT-NIFTI                # Auto-generated 
│       ├── PT-NIFTI                # Auto-generated
│       ├── RISK-ANALYSIS-XXX.xlsx  # Auto-generated: contains the risk-of-error analysis.
│   .
│   .
│   .
├── patient_folder_n
│       ├── fdgpet                  # The PET folder name can be named anything as long as the files inside this folder is DICOM and has a modality tag.
│       ├── ct                      # The CT folder name can be named anything as long as the files inside this folder is DICOM and has a modality tag.
│       ├── INFERENCE               # Auto-generated 
│       ├── MOOSE-TEMP              # Auto-generated
│       ├── LABELS                  # Auto-generated: contains all the generated labels.
│       ├── CT-NIFTI                # Auto-generated 
│       ├── PT-NIFTI                # Auto-generated
│       ├── RISK-ANALYSIS-XXX.xlsx  # Auto-generated: contains the risk-of-error analysis.
```

## ⛔️ Hard requirements 

The entire script has been *ONLY* tested on **Ubuntu linux OS**, with the following hardware capabilities:
- Intel(R) Xeon(R) Silver 4216 CPU @ 2.10GHz 
- 256 GB of RAM (Very important for total-body datasets)
- 1 x Nvidia GeForce RTX 3090 Ti (or similar)
We are testing different configurations now, but the RAM (256 GB) seems to be a hard requirement. 

## ⚙️ Installation

Kindly copy the code below and paste it on your ubuntu terminal, the installer should ideally take care of the rest. Also pay attention to the installation process as the FSL installation requires you to answer some questions. A fresh install would approximately take 30 minutes.

```bash

git clone https://github.com/LalithShiyam/MOOSE.git
cd MOOSE
source ./moose_installer.sh
```
**NOTE: Do not forget to source the .bashrc file again**

```bash
source ~/.bashrc
```
## 🖥 Usage

- For running the moose directly from the command-line terminal using the default options - please use the following command. In general, MOOSE performs the error analysis (refer paper) in similarity space and assumes that the given (if given) PET image is static.

```bash

#syntax:
moose -f path_to_main_folder 

#example: 
moose -f '/home/kyloren/Documents/main_folder'

```
- For notifying the program if the given 18F-FDG PET is static (-dp False) or dynamic (-dp True) and for switching on (-ea True) or off (-ea False) the error analysis error analysis in 'similarity space', use the following command with appropriate syntax.

```bash
#syntax:
moose -f path_to_main_folder -ea False -dp True 

#example for performing error analysis for a static PET/CT image: 
moose -f '/home/kyloren/Documents/main_folder' -ea True -dp False

#example for performing error analysis for a dynamic PET/CT image:
moose -f '/home/kyloren/Documents/main_folder' -ea True -dp True

#example for not performing error analysis:
moose -f '/home/kyloren/Documents/main_folder' -ea False -dp False

```

## 📈 Results

- The multi-label atlas for each subject will be stored in the auto-generated ```labels``` folder under the subject's respective directory (refer folder structure). The label-index to region correspondence is stored in the excel sheet: ```MOOSE-Label-Index-Correspondene-Dual-organs-without-split.xlsx```, which can be found inside the ```~/MOOSE/MOOSE-files/similarity-space``` folder.
- In addition, an auto-generated ```Segmentation-Risk-of-error-analysis-XXXX.xlsx``` file will be created in the individual subject-directory ('XXXX'). The excel file highlights segmentations that might be erroneously segmented. The excel sheet is supposed to serve as an quality control measure.

## 📖 Citations

- Software citation: Shiyam Sundar, L. K. (2022). MOOSE-120 (Version 0.9.0) [Computer software]. https://doi.org/10.5281/zenodo.5829597
- *Manuscript in submission*

## 🙏 Acknowledgement

This research is supported through an IBM University Cloud Award (https://www.research.ibm.com/university/)

## 🙋 FAQ

**[1]** Will MOOSE only work on whole-body 18F-FDG PET/CT datasets?

  *MOOSE ideally works on whole-body (head to toe) PET/CT datasets, but also works on semi whole-body PET/CT datasets (head to pelvis). Unfortunately, we haven't tested other field-of-views. We will post the evaluations soon.*


**[2]** Will MOOSE only work on multimodal 18F-FDG PET/CT datasets or can it also be applied to CT only? or PET only?

 *MOOSE automatically infers the modality type using the DICOM header tags. MOOSE builds the entire atlas with 120 tissues if the user provides multimodal 18F-FDG PET/CT datasets. The user can also provide CT only DICOM folder, MOOSE will infer the modality type and segment only the non-cerebral tissues (36/120 tissues) and will not segment the 83 subregions of the brain. MOOSE will definitely not work if only provided with 18F-FDG PET images.*


**[3]** Will MOOSE work on non-DICOM formats?

 *Unfortunately the current version accepts only DICOM formats. In the future, we will try to enable non-DICOM formats for processing as well.*


