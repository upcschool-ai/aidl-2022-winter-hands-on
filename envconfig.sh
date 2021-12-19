echo Initializing...

mkdir -p /workspace/conda
mkdir -p /workspace/data

conda create --prefix /workspace/conda/aidl python=3.8.12 &&
echo "conda activate /workspace/conda/aidl" >> ~/.bashrc &&
export PATH=/workspace/conda/arcw/bin:$PATH &&
source ~/.bashrc
export SHELL=/bin/bash

conda install -y pytorch torchvision torchaudio cpuonly -c pytorch

echo Done...