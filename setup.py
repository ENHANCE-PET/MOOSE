from setuptools import setup, find_packages

setup(
    name='moosez',
    version='2.3.1',
    author='Lalith Kumar Shiyam Sundar | Sebastian Gutschmayer',
    author_email='Lalith.shiyamsundar@meduniwien.ac.at',
    description='An AI-inference engine for 3D clinical and preclinical whole-body segmentation tasks',
    python_requires='>=3.9.2',
    long_description='mooseZ is an AI-inference engine based on nnUNet, designed for 3D clinical and preclinical'
                     ' whole-body segmentation tasks. It serves models tailored towards different modalities such'
                     ' as PET, CT, and MR. mooseZ provides fast and accurate segmentation results, making it a '
                     'reliable tool for medical imaging applications.',
    url='https://github.com/QIMP-Team/mooseZ',
    license='Apache 2.0',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Healthcare Industry',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3 :: Only',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Topic :: Scientific/Engineering :: Medical Science Apps.',
    ],

    keywords='moosez model-zoo nnUNet medical-imaging tumor-segmentation organ-segmentation bone-segmentation'
             ' lung-segmentation muscle-segmentation fat-segmentation vessel-segmentation'
             ' vertebral-segmentation rib-segmentation'
             ' preclinical-segmentation clinical-segmentation',
    packages=find_packages(),
    install_requires=[
        'nnunetv2',
        'nibabel~=3.2.2',
        'halo~=0.0.31',
        'pandas~=1.4.1',
        'SimpleITK~=2.2.1',
        'pydicom~=2.2.2',
        'argparse~=1.4.0',
        'imageio~=2.16.1',
        'numpy',
        'mpire~=2.3.3',
        'openpyxl~=3.0.9',
        'matplotlib',
        'pyfiglet~=0.8.post1',
        'natsort~=8.1.0',
        'pillow>=9.2.0',
        'colorama~=0.4.6',
        'dask',
        'rich',
        'pandas',
        'dicom2nifti~=2.4.8',
        'emoji',
        'dask[distributed]',
        'opencv-python-headless',
    ],
    entry_points={
        'console_scripts': [
            'moosez=moosez.moosez:main',
        ],
    },
)
