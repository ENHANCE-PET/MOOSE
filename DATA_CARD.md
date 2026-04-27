# 🦌ENHANCE.PET MOOSE 1.6k — Dataset Organization & Access 


----
### **Scope**  

- ENHANCE.PET 1.6k contains **1,683 whole-/total-body [18F]FDG-PET/CT studies** with **130 CT-derived, expert-verified segmentations** per scan.  
- All imaging is provided as **anonymized NIfTI** volumes, plus spreadsheets covering CT/PET acquisition parameters and participant demographics.  
- The dataset is intended for segmentation benchmarking, multi-organ analysis, radiomics, and PET/CT AI research.

| Estimated size   | Primary access method                                           | Support contact                   |
|------------------|-----------------------------------------------------------------|-----------------------------------|
| ~266 GB          | Science Data Bank (https://doi.org/10.57760/sciencedb.34150)    | Lalith.shiyam@med.uni-muenchen.de |
|                  | see “Access on Science DB via MOOSE CLI” below)                 |                                   |

---

## 1) Storage layout (Science Data Bank, https://doi.org/10.57760/sciencedb.34150)

Top-level prefix: `ENHANCE-PET_1.6k/`

```
ENHANCE-PET_1.6k/
  CT-details.xlsx
  PT-details.xlsx
  labels.json
  imaging-data/
    images/
      CT_0001-1000/
        0001.nii.gz
        ...
        1000.nii.gz
      CT_1001-1683/
        1001.nii.gz
        ...
        1683.nii.gz
      PT_0001-1683/
        0001.nii.gz
        ...
        1683.nii.gz
    ground-truth/
      Body-Composition/
        0001.nii.gz … 1683.nii.gz
      Cardiac/
        0001.nii.gz … 1683.nii.gz
      Muscles/
        0001.nii.gz … 1683.nii.gz
      Organs/
        0001.nii.gz … 1683.nii.gz
      Peripheral-Bones/
        0001.nii.gz … 1683.nii.gz
      Ribs/
        0001.nii.gz … 1683.nii.gz
      Vertebrae/
        0001.nii.gz … 1683.nii.gz
```

**Notes**
- File IDs are **sequential** (`0001.nii.gz` … `1683.nii.gz`) and correspond across CT, PT, and each segmentation group.
- Segmentations are **grouped** into seven model-specific folders under `ground-truth/`.

---

## 2) File types

- **Images:** NIfTI `.nii.gz`  
  - `imaging-data/images/CT/` — CT volumes (anonymized)  
  - `imaging-data/images/PT/` — PET volumes (anonymized)
- **Segmentations:** NIfTI `.nii.gz`, one file per case for each **segmentation group** (see §3) under `imaging-data/ground-truth/…`
- **Spreadsheets:**  
  - `CT-details.xlsx` — CT acquisition parameters  
  - `PT-details.xlsx` — PET acquisition parameters and demographics  
- **Class mapping:**  
  - `labels.json` — group-specific **class → intensity** mapping

---

## 3) Segmentation groups and class mapping (`labels.json`)

Seven segmentation groups are provided. Below are **representative excerpts** from the `labels.json` mapping (full mapping in the file).

### 3.1 Cardiac (`clin_ct_cardiac`)
```json
{
  "1": "heart_myocardium",
  "2": "heart_atrium_left",
  "3": "heart_atrium_right",
  "4": "heart_ventricle_left",
  "5": "heart_ventricle_right",
  "6": "aorta",
  "7": "iliac_artery_left",
  "8": "iliac_artery_right",
  "9": "iliac_vena_left",
  "10": "iliac_vena_right",
  "11": "inferior_vena_cava",
  "12": "portal_splenic_vein",
  "13": "pulmonary_artery"
}
```

### 3.2 Muscles (`clin_ct_muscles`)
```json
{
  "1": "autochthon_left",
  "2": "autochthon_right",
  "3": "gluteus_maximus_left",
  "4": "gluteus_maximus_right",
  "5": "gluteus_medius_left",
  "6": "gluteus_medius_right",
  "7": "gluteus_minimus_left",
  "8": "gluteus_minimus_right",
  "9": "iliopsoas_left",
  "10": "iliopsoas_right"
}
```

### 3.3 Organs (`clin_ct_organs`)
```json
{
  "1": "adrenal_gland_left",
  "2": "adrenal_gland_right",
  "3": "bladder",
  "5": "gallbladder",
  "6": "kidney_left",
  "7": "kidney_right",
  "8": "liver",
  "9": "lung_lower_lobe_left",
  "10": "lung_lower_lobe_right",
  "11": "lung_middle_lobe_right",
  "12": "lung_upper_lobe_left",
  "13": "lung_upper_lobe_right",
  "14": "pancreas",
  "15": "spleen",
  "16": "stomach",
  "17": "thyroid_left",
  "18": "thyroid_right"
}
```

### 3.4 Peripheral Bones (`clin_ct_peripheral_bones`)
```json
  {
    "1": "carpal_left",
    "2": "carpal_right",
    "3": "clavicle_left",
    "4": "clavicle_right",
    "5": "femur_left",
    "6": "femur_right",
    "7": "fibula_left",
    "8": "fibula_right",
    "9": "fingers_left",
    "10": "fingers_right",
    "11": "humerus_left",
    "12": "humerus_right",
    "13": "metacarpal_left",
    "14": "metacarpal_right",
    "15": "metatarsal_left",
    "16": "metatarsal_right",
    "17": "patella_left",
    "18": "patella_right",
    "19": "radius_left",
    "20": "radius_right",
    "21": "scapula_left",
    "22": "scapula_right",
    "24": "tarsal_left",
    "25": "tarsal_right",
    "26": "tibia_left",
    "27": "tibia_right",
    "28": "toes_left",
    "29": "toes_right",
    "30": "ulna_left",
    "31": "ulna_right"
  }
```

### 3.5 Ribs (`clin_ct_ribs`)
```json
{
    "1": "rib_left_1",
    "2": "rib_left_2",
    "3": "rib_left_3",
    "4": "rib_left_4",
    "5": "rib_left_5",
    "6": "rib_left_6",
    "7": "rib_left_7",
    "8": "rib_left_8",
    "9": "rib_left_9",
    "10": "rib_left_10",
    "11": "rib_left_11",
    "12": "rib_left_12",
    "13": "rib_left_13",
    "14": "rib_right_1",
    "15": "rib_right_2",
    "16": "rib_right_3",
    "17": "rib_right_4",
    "18": "rib_right_5",
    "19": "rib_right_6",
    "20": "rib_right_7",
    "21": "rib_right_8",
    "22": "rib_right_9",
    "23": "rib_right_10",
    "24": "rib_right_11",
    "25": "rib_right_12",
    "26": "rib_right_13",
    "27": "sternum"
  }
```

### 3.6 Vertabrae (`clin_ct_vertebrae`)
```json
{
    "1": "vertebra_C1",
    "2": "vertebra_C2",
    "3": "vertebra_C3",
    "4": "vertebra_C4",
    "5": "vertebra_C5",
    "6": "vertebra_C6",
    "7": "vertebra_C7",
    "8": "vertebra_T1",
    "9": "vertebra_T2",
    "10": "vertebra_T3",
    "11": "vertebra_T4",
    "12": "vertebra_T5",
    "13": "vertebra_T6",
    "14": "vertebra_T7",
    "15": "vertebra_T8",
    "16": "vertebra_T9",
    "17": "vertebra_T10",
    "18": "vertebra_T11",
    "19": "vertebra_T12",
    "20": "vertebra_L1",
    "21": "vertebra_L2",
    "22": "vertebra_L3",
    "23": "vertebra_L4",
    "24": "vertebra_L5",
    "25": "vertebra_L6",
    "26": "hip_left",
    "27": "hip_right",
    "28": "sacrum"
  }
```

### 3.7 Body Composition (`clin_ct_body_composition`)
```json
{
    "1": "skeletal_muscle",
    "2": "subcutaneous_fat",
    "3": "visceral_fat"
  }
```

---

## 4) Naming conventions

- **Imaging files:** zero‑padded numeric IDs (`0001.nii.gz` … `1683.nii.gz`)  
- **One‑to‑one mapping:** the same ID indexes a **single case** across CT, PT, and each segmentation group  
- **labels.json:** provides the **canonical intensity mapping** for each segmentation group

---

## 5) Anonymization

- All PET/CT contributions are defaced (cranial region handled in CT and PET to prevent re‑identification).    
- All shared data are **anonymized** prior to release.

---

## 6) License and provenance

Licensing is **per originating site**, recorded in the `Data-Source` column of both `CT-details.xlsx` and `PT-details.xlsx`:

| Data-Source                                    | License     | Notes                                                        |
|------------------------------------------------|-------------|--------------------------------------------------------------|
| AutoPET Challenge                              | CC BY-NC 4.0| Open challenge dataset; non-commercial use                   |
| University Hospital Leipzig (DE).              | CC BY 4.0   | LUCAPET Consortium                                           |
| Azienda Ospedaliero Universitaria Careggi (IT) | CC BY 4.0   | LUCAPET Consortium                                           |
| University of California Davis (US) 		 | CC BY 4.0   | External contribution, provided directly by the institution  |


---

## 7) Access to Science DB via MOOSE CLI

This dataset is **distributed via the MOOSE CLI** to simplify discovery and download.

1. **Install MOOSE** (see the MOOSE repository for setup instructions).  
2. **Download** to a local folder of your choice:
   ```bash
   moosez -dtd -dd /path/to/download/
   ```
3. **Explore** the data locally using a viewer such as **3D Slicer**, or load NIfTI with Python tooling.

---

## 8) Expected counts & integrity checks

- **CT volumes:** 1,683 files  
- **PT volumes:** 1,683 files  
- **Segmentations:** for each of the seven groups, **1,683 files**  
- **Metadata files:** `CT-details.xlsx`, `PT-details.xlsx`, `labels.json`

---

## 9) Known caveats

- Segmentations are **derived from CT**; in cases with notable **patient motion**, PET↔CT misalignment may be present.  
- Some very small or thin structures can be more challenging (e.g., small vessels, digits), which should be considered during downstream QA.

---

## 10) Contact

For questions, issues, or requests: **Lalith Kumar Shiyam Sundar** — `Lalith.shiyam@med.uni-muenchen.de`
