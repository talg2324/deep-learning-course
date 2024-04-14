# Steps to setup VM

## Clone the repo
1. git clone https://github.com/talg2324/deep-learning-course.git
2. cd deep-learning-course/Project
3. git submodule update --init --recursive

## Setup conda env
4. conda env create -f env.yaml
5. conda activate ldm
6. python -m pip install -e ./taming-transformers
7. python -m pip install git+https://github.com/openai/CLIP.git

## Setup jupyter
8. python -m pip install notebook
9. chmod +x ./setup/setup_jupyter.sh
10. ./setup/setup_jupyter.sh

## Get the pre-trained model
11. mkdir -p latent-diffusion/models/ldm/cin256-v2/
12. wget -O latent-diffusion/models/ldm/cin256-v2/model.ckpt https://ommer-lab.com/files/latent-diffusion/nitro/cin/model.ckpt