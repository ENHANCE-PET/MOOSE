from setuptools import setup, find_packages
import sys
import os
import subprocess
from setuptools.command.install import install
import tarfile

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


def install_library(library_name, library_url, library_version):
    project_root = get_virtual_env_root()
    os.chdir(project_root)
    r = requests.get(library_url, allow_redirects=True)
    open(f"{library_name}.tar.gz", "wb").write(r.content)
    tar = tarfile.open(f"{library_name}.tar.gz")
    tar.extractall()
    tar.close()
    os.remove(f"{library_name}.tar.gz")
    os.chdir(f"{library_name}-{library_version}")
    subprocess.run(["mkdir", "build"])
    os.chdir("build")
    subprocess.run(["cmake", "BUILD_SHARED_LIBS=ON", "CMAKE_BUILD_TYPE=Release", ".."])
    subprocess.run(["make", "install"])
    subprocess.run(["make", "-j", str(n_cores / 2)])
    os.chdir("..")
    os.chdir("..")


def install_itk():
    install_library("InsightToolkit",
                    "https://github.com/InsightSoftwareConsortium/ITK/releases/download/v5.1.2/InsightToolkit-5.1.2"
                    ".tar.gz",
                    "5.1.2")


def install_vtk():
    install_library("vtk", "https://gitlab.kitware.com/vtk/vtk/-/archive/v9.1.0/vtk-v9.1.0.tar.gz", "v9.1.0")


def get_itk_binaries_path():
    project_root = get_virtual_env_root()
    return os.path.join(project_root, "InsightToolkit-5.1.2", "build", "bin")


def get_vtk_binaries_path():
    project_root = get_virtual_env_root()
    return os.path.join(project_root, "vtk-v9.1.0", "build", "bin")


def install_greedy():
    project_root = get_virtual_env_root()
    os.chdir(project_root)
    subprocess.run(["git", "clone", "https://github.com/pyushkevich/greedy.git"])
    os.chdir("greedy")
    subprocess.run(["mkdir", "build"])
    os.chdir("build")
    subprocess.run(["cmake", f"ITK_DIR={get_itk_binaries_path()}", f"VTK_DIR={get_vtk_binaries_path()}",
                    "CMAKE_BUILD_TYPE=Release", ".."])
    subprocess.run(["make", "-j", str(n_cores / 2)])


class CustomInstallCommand(install):
    def run(self):
        install.run(self)
        install_dcm2niix()
        install_itk()
        install_vtk()
        install_greedy()


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
