import os
import subprocess
import sys

from setuptools import setup, find_packages
from setuptools.command.install import install

# check if requests is installed, if not, install it
try:
    import requests
except ImportError:
    subprocess.run(["pip", "install", "requests"])
    import requests

# check if multiprocessing is installed, if not, install it
try:
    import multiprocessing

    n_cores = multiprocessing.cpu_count()
except ImportError:
    subprocess.run(["pip", "install", "multiprocessing"])
    import multiprocessing

    n_cores = multiprocessing.cpu_count()


def get_virtual_env_root():
    python_exe = sys.executable
    virtual_env_root = os.path.dirname(os.path.dirname(python_exe))
    return virtual_env_root


def install_dcm2niix():
    project_root = get_virtual_env_root()
    os.makedirs(project_root, exist_ok=True)
    os.chdir(project_root)
    subprocess.run(["git", "clone", "https://github.com/rordenlab/dcm2niix.git"])
    os.chdir("dcm2niix")
    subprocess.run(["mkdir", "build"])
    os.chdir("build")
    subprocess.run(["cmake", "-DZLIB_IMPLEMENTATION=Cloudflare", "-DUSE_JPEGLS=ON", "-DUSE_OPENJPEG=ON", ".."])
    subprocess.run(["make", "install"])
    os.chdir("..")
    os.chdir("..")


class CustomInstallCommand(install):
    def run(self):
        install.run(self)
        install_dcm2niix()


setup(
    name='moosez',
    version='2.0.0',
    author='Lalith Kumar Shiyam Sundar',
    author_email='Lalith.shiyamsundar@meduniwien.ac.at',
    description='An AI-inference engine for 3D clinical and preclinical whole-body segmentation tasks',
    long_description='mooseZ is an AI-inference engine based on nnUNet, designed for 3D clinical and preclinical'
                     ' whole-body segmentation tasks. It serves models tailored towards different modalities such'
                     ' as PET, CT, and MR. mooseZ provides fast and accurate segmentation results, making it a '
                     'reliable tool for medical imaging applications.',
    url='https://github.com/QIMP-Team/mooseZ',
    license='Apache 2.0',
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Researchers',
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
        'nibabel~=3.2.2',
        'halo~=0.0.31',
        'pandas~=1.4.1',
        'SimpleITK~=2.1.1',
        'pydicom~=2.2.2',
        'argparse~=1.4.0',
        'numpy~=1.22.3',
        'mpire~=2.3.3',
        'openpyxl~=3.0.9',
        'matplotlib~=3.1.3',
        'pyfiglet~=0.8.post1',
        'natsort~=8.1.0',
        'pillow>=9.2.0',
    ],
    cmdclass={'install': CustomInstallCommand},
    entry_points={
        'console_scripts': [
            'moosez=moosez.moosez:main',
        ],
    },
)
