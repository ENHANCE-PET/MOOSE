from setuptools import setup, find_packages


setup(
    name='moosez',
    version="3.0.20",
    author='Lalith Kumar Shiyam Sundar | Sebastian Gutschmayer | Manuel Pires',
    author_email='Lalith.shiyamsundar@meduniwien.ac.at',
    description='An AI-inference engine for 3D clinical and preclinical whole-body segmentation tasks',
    python_requires='>=3.10',
    long_description='mooseZ is an AI-inference engine based on nnUNet, designed for 3D clinical and preclinical'
                     ' whole-body segmentation tasks. It serves models tailored towards different modalities such'
                     ' as PET, CT, and MR. mooseZ provides fast and accurate segmentation results, making it a '
                     'reliable tool for medical imaging applications.',
    url='https://github.com/ENHANCE-PET/MOOSE',
    license='GPLv3',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Healthcare Industry',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Programming Language :: Python :: 3 :: Only',
        'Programming Language :: Python :: 3.10',
        'Topic :: Scientific/Engineering :: Medical Science Apps.',
    ],

    keywords='moosez model-zoo nnUNet medical-imaging tumor-segmentation organ-segmentation bone-segmentation'
             ' lung-segmentation muscle-segmentation fat-segmentation vessel-segmentation'
             ' vertebral-segmentation rib-segmentation'
             ' preclinical-segmentation clinical-segmentation',
    packages=find_packages(),
    install_requires=[
        'torch',
        'dynamic-network-architectures==0.3.1',
        'SimpleITK',
        'nnunetv2>=2.6.0',
        'halo~=0.0.31',
        'pydicom~=2.2.2',
        'argparse',
        'numpy<2.0',
        'pyfiglet~=0.8.post1',
        'natsort',
        'colorama~=0.4.6',
        'dask',
        'rich',
        'pandas',
        'dicom2nifti~=2.4.8',
        'emoji',
        'matplotlib',
        'psutil',
        'nibabel'
    ],
    entry_points={
        'console_scripts': [
            'moosez=moosez.moosez:main',
        ],
    },
)
