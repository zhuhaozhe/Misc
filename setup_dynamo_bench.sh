cd ${WORK_DIR}
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p ${WORK_DIR}/miniconda3
source ${WORK_DIR}/miniconda3/bin/activate
conda create --name hz python=3.9 -y
conda activate hz

cd ${WORK_DIR}
git clone https://github.com/pytorch/pytorch.git
cd pytorch
git checkout gh/zhuhaozhe/4/orig
git submodule update --init --recursive
conda install cmake ninja -y
conda install intel::mkl-static intel::mkl-include
pip install -r requirements.txt
CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"} python setup.py develop

cd ../
git clone https://github.com/pytorch/vision.git
cd vision
git submodule update --init --recursive
python setup.py develop

cd ../
git clone https://github.com/pytorch/audio.git
cd audio
git submodule update --init --recursive
pip install -r requirements.txt
python setup.py develop

cd ../
git clone https://github.com/pytorch/benchmark.git
cd benchmark
python install.py --continue_on_fail

conda install jemalloc intel-openmp -y
