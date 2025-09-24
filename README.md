## Installation

Prerequisites:
- CUDA version >=11.8 (this is required if you want to perform a full installation of this repo and perform RT-1 or Octo inference)
- An NVIDIA GPU (ideally RTX; for non-RTX GPUs, such as 1080Ti and A100, environments that involve ray tracing will be slow). Currently TPU is not supported as SAPIEN requires a GPU to run.

Install SimplerEnv dependencies:
```
conda create -n simpler_env python=3.10 (any version above 3.10 should be fine)
conda activate simpler_env
git clone https://github.com/szhaovas/SimplerEnv-OpenVLA.git
cd SimplerEnv-OpenVLA/ManiSkill2_real2sim
pip install -e .
cd ..
pip install -e .
pip install --upgrade numpy==1.24.2
```

Install OpenVLA-OFT dependencies:
```
sudo apt install ffmpeg
pip install tensorflow==2.15.0
pip install -r requirements_full_install.txt
pip install tensorflow[and-cuda]==2.15.1 # tensorflow gpu support
pip install git+https://github.com/nathanrooy/simulated-annealing
pip install torch==2.2.0 torchvision==0.17.0 timm==0.9.10 tokenizers==0.19.1 accelerate==0.32.1
pip install flash-attn==2.6.1 --no-build-isolation
pip install --upgrade tensorflow-graphics json-numpy==2.1.1 draccus==0.11.5 jsonlines==4.0.0 diffusers==0.35.1 transformers==4.40.1 gymnasium==0.28.1
```
