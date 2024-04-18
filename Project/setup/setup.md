# Steps to setup VM

## git publickey setup (ssh-keygen)
the below should allow you to pull a repo using SSH clone, and then push updates back easily
(see https://stackoverflow.com/a/2643584)

1. cd ~/.ssh && ssh-keygen
2. cat id_rsa.pub
3. copy the key to clipboard
4. go to your github account settings page > SSH and GPG keys > New SSH key.
   paste the key.
6. make sure to setup git username and email (not sure if necessary)
   - git config --global user.name "your.name"
   - git config --global user.email your@mail.com
8. eval $(ssh-agent -s)
9. ssh-add ~/.ssh/id_rsa

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

## GCSFuse Instructions
13. https://cloud.google.com/storage/docs/gcsfuse-quickstart-mount-bucket