#!/bin/bash
username=$(whoami)
#echo $username

condastr=`conda info | grep 'active environment' | awk '{print $4}'`

cudarootstr=/home/${username}/miniconda3/envs/${condastr}/
echo ${cudarootstr}


# export CUDA_ROOT=$(./activate_cuda.sh)

