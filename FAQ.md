
# 🧐 Frequently Asked Questions (FAQ)

## Q1: How to fix "IndexError: list index out of range"?

**Problem Description:**

Running into trouble with Moose?  These error messages might be the culprit:

**Error Message 1:**
```
moose_ct_atlas = ie.segment_ct(ct_file[0], out_dir)
File "/export/moose/moose-0.1.0/src/inferenceEngine.py", line 78, in segment_ct
out_label = fop.get_files(out_dir, pathlib.Path(nifti_img).stem + '*')[0]
IndexError: list index out of range
```

**Error Message 2:**
```
[1/1] Running prediction for PETCT_0225325b91 using clin_ct_muscles...Traceback (most recent call last):
...
IndexError: list index out_of_range
```

**Solutions:**

1. **Double-Check Requirements:**  ✅
    - Make sure you're using Python version 3.10. You can check this by running `python --version` in your terminal. 
    - Ensure you have enough RAM available for your tasks. Moose might require more RAM for complex datasets.
    - Refer to the official Moose repository for the complete list of requirements: [https://github.com/ENHANCE-PET/MOOSE](https://github.com/ENHANCE-PET/MOOSE)

2. **Watch Out for Spaces!**  
    - Moose might have trouble with spaces in your directory paths. Try renaming directories to replace spaces with underscores (_). 

3. **Set Environment Variables (if applicable):**  ⚙️
    - Some systems might require specific environment variables to be set for Moose to function correctly. Refer to the Moose documentation for details on any necessary environment variables.

4. **Consider Specific Versions (for Stability):**  ✨
    - If you're facing persistent issues, try using a specific version of PyTorch (2.1.1) and CUDA (11.8) known to be compatible with Moose.

## Q2: Why doesn't Moose work on MRI?**

Moose is currently designed to work specifically with CT scans. If you're looking to analyze MRI data, you'll need a different tool.  For more information on Moose's functionalities, refer to the README section: [https://github.com/ENHANCE-PET/MOOSE](https://github.com/ENHANCE-PET/MOOSE)

**Q3: Using Moose as a Package?**

**Important Notes:**

- To use Moose as a package, you'll need to convert your DICOM files to the .nifti format (.nii or .nii.gz).
- Moose currently processes data for one subject at a time. Batch processing is not yet supported.
