# Steps to setup VM

## Clone the repo
1. git clone https://github.com/talg2324/deep-learning-course.git
2. cd deep-learning-course/Project
3. git submodule update --init --recursive

## Setup conda env
4. conda env create -f env.yaml
5. conda activate ldm
6. python -m pip install -e ./taming-transformers

## Get the pre-trained model
7. wget -O latent-diffusion/models/ldm/cin256/model.zip https://ommer-lab.com/files/latent-diffusion/cin.zip
8. cd latent-diffusion/models/ldm/cin256
9. unzip -o model.zip