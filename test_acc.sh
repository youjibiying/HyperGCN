#!/bin/bash
#BATCH -J python       # job name, optional
#SBATCH -p defq       # partition name (should always be defq)
#SBATCH -N 1          # number of computing node
#SBATCH --ntasks=1    # maximum number of parallel tasks (processes)
#SBATCH --gres=gpu:01  # number of gpus allocated on each node


#exec 1>test_acc:.csv

layers=(2 4 8 16 32 64 128)

dset=(cocitation/pubmed cocitation/citeseer coauthorship/dblp)

data=(cocitation coauthorship)
dataset1=(cora pubmed citeseer)
dataset2=(cora dblp)

for ((t=3;t<7;t++)){
   for ((i=0;i<2;i++)){
      python hypergcn.py --mediators True --split 1 --depth ${layers[t]} --data coauthorship --dataset ${dataset2[i]}
}
}

for ((t=3;t<7;t++)){
   for ((i=0;i<3;i++)){
      python hypergcn.py --mediators True --split 1 --depth ${layers[t]} --data cocitation --dataset ${dataset1[i]}
}
}







