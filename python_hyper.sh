#!/bin/bash
#SBATCH -J python       # job name, optional
#SBATCH -p defq       # partition name (should always be defq)
#SBATCH -c 8
#SBATCH --ntasks=1    # maximum number of parallel tasks (processes)
#SBATCH --gres=gpu:01  # number of gpus allocated on each node


   
python hypergcn.py --depth 4 --data coauthorship --dataset cora
python hypergcn.py --depth 8 --data coauthorship --dataset cora
python hypergcn.py --depth 16 --data coauthorship --dataset cora
python hypergcn.py --depth 32 --data coauthorship --dataset cora
python hypergcn.py --depth 64 --data coauthorship --dataset cora


python hypergcn.py --depth 4 --data cocitation --dataset cora
python hypergcn.py --depth 8 --data cocitation --dataset cora
python hypergcn.py --depth 16 --data cocitation --dataset cora
python hypergcn.py --depth 32 --data cocitation --dataset cora
python hypergcn.py --depth 64 --data cocitation --dataset cora