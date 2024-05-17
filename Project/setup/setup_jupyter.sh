#!/usr/bin/env bash

echo Jupyter setup script ...
jupyter notebook --generate-config
cp -f ./setup/jupyter_notebook_config.py ~/.jupyter/
echo Please set a password for your Jupyter notebook
python -m jupyter_server.auth password
sudo chmod -R 777 /opt
