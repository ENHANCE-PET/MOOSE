# 🦌ENHANCE.PET MOOSE 1.6k — Dataset Organization & Access (AWS)

### **Scope**  

- ENHANCE.PET 1.6k contains **1,597 whole-/total-body [^18F]FDG-PET/CT studies** with **130 CT-derived, expert-verified segmentations** per scan.  
- All imaging is provided as **anonymized NIfTI** volumes, plus spreadsheets covering CT/PET acquisition parameters and participant demographics.  
- The dataset is intended for segmentation benchmarking, multi-organ analysis, radiomics, and PET/CT AI research.

| Estimated size   | Primary access method | Support contact                      |
|------------------|-----------------------|---------------------------------------|
| ~250 GB          | MOOSE CLI (see “Access on AWS via MOOSE CLI” below) | Lalith.shiyam@med.uni-muenchen.de |

---

## 1) Storage layout (AWS Open Data bucket)

Top-level prefix : `enhance-pet-1-6k/`

```
s3://<aws-open-data-bucket>/enhance-pet-1-6k/
  CT-details.xlsx
  PT-details.xlsx
  labels.json
  imaging-data/
    images/
      CT/
        0001.nii.gz
        ...
        1597.nii.gz
      PT/
        0001.nii.gz
        ...
        1597.nii.gz
    ground-truth/
      Body-Composition/
        0001.nii.gz … 1597.nii.gz
      Cardiac/
        0001.nii.gz … 1597.nii.gz
      Muscles/
        0001.nii.gz … 1597.nii.gz
      Organs/
        0001.nii.gz … 1597.nii.gz
      Peripheral-Bones/
        0001.nii.gz … 1597.nii.gz
      Ribs/
        0001.nii.gz … 1597.nii.gz
      Vertebrae/
        0001.nii.gz … 1597.nii.gz
```

**Notes**
- File IDs are **sequential** (`0001.nii.gz` … `1597.nii.gz`) and correspond across CT, PT, and each segmentation group.
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
  "4": "brain",
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

---

## 4) Naming conventions

- **Imaging files:** zero‑padded numeric IDs (`0001.nii.gz` … `1597.nii.gz`)  
- **One‑to‑one mapping:** the same ID indexes a **single case** across CT, PT, and each segmentation group  
- **labels.json:** provides the **canonical intensity mapping** for each segmentation group

---

## 5) Anonymization

- Cohorts from Careggi and Leipzig are defaced (cranial region handled in CT and PET to prevent re‑identification).  
- PET/CT contributions from AutoPET retain upstream anonymization + facial removal.  
- All shared data are **anonymized** prior to release.

---

## 6) License and provenance

Licensing is **per originating site**, recorded in the `Data-Source` column of both `CT-details.xlsx` and `PT-details.xlsx`:

| Data-Source                                     | License     | Notes                                      |
|------------------------------------------------|-------------|--------------------------------------------|
| AutoPET Challenge                              | CC BY-NC 4.0| Open challenge dataset; non-commercial use |
| University Hospital Leipzig (Germany)          | CC BY 4.0   | LUCAPET Consortium                         |
| Azienda Ospedaliero Universitaria Careggi (IT) | CC BY 4.0   | LUCAPET Consortium                         |

---

## 7) Access on AWS via MOOSE CLI

This dataset is **distributed via the MOOSE CLI** to simplify discovery and download.

1. **Install MOOSE** (see the MOOSE repository for setup instructions).  
2. **Download** to a local folder of your choice:
   ```bash
   moosez -dtd -dd /path/to/download/
   ```
3. **Explore** the data locally using a viewer such as **3D Slicer**, or load NIfTI with Python tooling.

---

## 8) Expected counts & integrity checks

- **CT volumes:** 1,597 files  
- **PT volumes:** 1,597 files  
- **Segmentations:** for each of the seven groups, **1,597 files**  
- **Metadata files:** `CT-details.xlsx`, `PT-details.xlsx`, `labels.json`

---

## 9) Known caveats

- Segmentations are **derived from CT**; in cases with notable **patient motion**, PET↔CT misalignment may be present.  
- Some very small or thin structures can be more challenging (e.g., small vessels, digits), which should be considered during downstream QA.

---

## 10) Contact

For questions, issues, or requests: **Lalith Kumar Shiyam Sundar** — `Lalith.shiyam@med.uni-muenchen.de`
