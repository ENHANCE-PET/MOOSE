#!/usr/bin/env bash
echo '[1] Installing python packages for running moose...'
pip install -r requirements.txt
echo '[2] Downloading required files IBM cloud storage...'
wget "https://moose-files.s3.eu-de.cloud-object-storage.appdomain.cloud/MOOSE-files-24062022.zip"
echo '[3] Unzipping moose files...'
unzip MOOSE-files-24062022.zip
echo '[4] Removing moose zip files...'
rm MOOSE-files-24062022.zip

# shellcheck disable=SC2006
main_dir=`pwd`
# shellcheck disable=SC2006
moose_dir=`pwd`/'MOOSE-files-24062022'
moose_bin=$moose_dir/'bin'
root_path='/usr/local/bin'
nnUNet_dir=$moose_dir/'moose-models'
sim_space_dir=$moose_dir/'similarity-space/Norm_DB.xlsx'
moose_src=$main_dir/'src'/'run_moose.py'
brain_detector_dir=$nnUNet_dir/'brain_detector.pkl'

echo '[5] Setting up nnUNet environment variables in the .bashrc'
# shellcheck disable=SC2129
# shellcheck disable=SC2027
echo "export nnUNet_raw_data_base=""${nnUNet_dir}""/nnUNet_raw" >> ~/.bashrc
# shellcheck disable=SC2027
echo "export nnUNet_preprocessed=""${nnUNet_dir}""/nnUNet_preprocessed" >> ~/.bashrc
# shellcheck disable=SC2027
# shellcheck disable=SC2086
echo "export RESULTS_FOLDER="${nnUNet_dir}"/nnUNet_trained_models" >> ~/.bashrc
# shellcheck disable=SC2027
# shellcheck disable=SC2086
echo "export SIM_SPACE_DIR="${sim_space_dir}"" >> ~/.bashrc
# shellcheck disable=SC2027
# shellcheck disable=SC2086
echo "export BRAIN_DETECTOR_DIR="${brain_detector_dir}"" >> ~/.bashrc

echo '[6] Check later if environment variables for nnUNet are set in .bashrc'
echo '[7] Setting up symlinks for dependencies...'
sudo ln -s "$moose_bin"/'c3d' $root_path/'c3d'
# shellcheck disable=SC2086
sudo ln -s $moose_bin/'greedy' $root_path/'greedy'
echo '[8] Building dcm2niix from source...'
sudo apt-get install cmake pkg-config
sudo apt install git
git clone https://github.com/rordenlab/dcm2niix.git
# shellcheck disable=SC2164
cd dcm2niix
# shellcheck disable=SC2164
mkdir build && cd build
cmake -DZLIB_IMPLEMENTATION=Cloudflare -DUSE_JPEGLS=ON -DUSE_OPENJPEG=ON ..
sudo make install
echo '[09] Installing pytorch...'
pip install torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio==0.10.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
echo '[10] Installing pigz parallel compression library...'
sudo apt-get update -y
sudo apt-get install -y pigz
sudo chmod +x "$moose_src"
sudo ln -s "$moose_src" $root_path/'moose'
