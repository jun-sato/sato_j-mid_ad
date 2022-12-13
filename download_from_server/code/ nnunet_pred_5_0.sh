#!/bin/bash
#PBS -q SQUID
#PBS --group=K22A11
#PBS -m be
#PBS -M j-sato@radiol.med.osaka-u.ac.jp
#PBS -l gpunum_job=8
#PBS -l elapstim_req=120:00:00
cd nnUNet/nnunet
nvidia-smi
source activate nnunet
python -c 'import torch;print(torch.cuda.device_count())'
python -c 'import torch;print(torch.distributed.is_mpi_available())'
python -c 'import torch;print(torch.distributed.is_nccl_available())'
python -c 'import torch;print(torch.distributed.is_torchelastic_launched())'

CUDA_VISIBLE_DEVICES=0 nnUNet_predict -i /sqfs/work/K22A11/u6b588/jmid/data/remain_5_0/ -o /sqfs/work/K22A11/u6b588/jmid/data/pred_5_0/ -t 601 -m 3d_fullres -p nnUNetPlansv2.1_ps288_bs2  --part_id=0 --num_parts=4 & CUDA_VISIBLE_DEVICES=1 nnUNet_predict -i /sqfs/work/K22A11/u6b588/jmid/data/remain_5_0/ -o /sqfs/work/K22A11/u6b588/jmid/data/pred_5_0/ -t 601 -m 3d_fullres -p nnUNetPlansv2.1_ps288_bs2  --part_id=1 --num_parts=4 & CUDA_VISIBLE_DEVICES=2 nnUNet_predict -i /sqfs/work/K22A11/u6b588/jmid/data/remain_5_0/ -o /sqfs/work/K22A11/u6b588/jmid/data/pred_5_0/ -t 601 -m 3d_fullres -p nnUNetPlansv2.1_ps288_bs2  --part_id=2 --num_parts=4 & CUDA_VISIBLE_DEVICES=3 nnUNet_predict -i /sqfs/work/K22A11/u6b588/jmid/data/remain_5_0/ -o /sqfs/work/K22A11/u6b588/jmid/data/pred_5_0/ -t 601 -m 3d_fullres -p nnUNetPlansv2.1_ps288_bs2  --part_id=3 --num_parts=4 
