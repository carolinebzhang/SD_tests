conda create --name my_env
#!/bin/bash

source activate my_env
pip install pytorch
pip install PIL
pip install diffusers
# Your commands here

conda deactivate
#bash my_script.sh