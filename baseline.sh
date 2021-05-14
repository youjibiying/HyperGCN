#!/bin/bash


#python hypergcn.py --mediators True --split 1 --depth 16 --data cocitation --dataset citeseer
#python hypergcn.py --mediators True --split 1 --depth 32 --data cocitation --dataset citeseer
#python hypergcn.py --mediators True --split 1 --depth 64 --data cocitation --dataset citeseer --gpu 0 &
#python hypergcn.py --mediators True --split 1 --depth 64 --data cocitation --dataset cora --gpu 0 &
#python hypergcn.py --mediators True --split 1 --depth 64 --data cocitation --dataset pubmed --gpu 1 &
#python hypergcn.py --mediators True --split 1 --depth 64 --data coauthorship --dataset cora --gpu 3 &
python hypergcn.py --mediators True --split 1 --gpu 2 --depth 64 --data coauthorship --dataset dblp > depth_64.log 2>&1 &



#python hypergcn.py --mediators True --split 1 --depth 32 --data cocitation --dataset citeseer --gpu 0 &
#python hypergcn.py --mediators True --split 1 --depth 32 --data cocitation --dataset cora --gpu 0 &
#python hypergcn.py --mediators True --split 1 --depth 32 --data cocitation --dataset pubmed --gpu 1 &
#python hypergcn.py --mediators True --split 1 --depth 32 --data coauthorship --dataset cora --gpu 1 &
#python hypergcn.py --mediators True --split 1 --depth 32 --data coauthorship --dataset dblp --gpu 0 &


#python hypergcn.py --mediators True --split 1 --depth 16 --data cocitation --dataset citeseer --gpu 2 &
#python hypergcn.py --mediators True --split 1 --depth 16 --data cocitation --dataset cora --gpu 1 &
#python hypergcn.py --mediators True --split 1 --depth 16 --data cocitation --dataset pubmed --gpu 1 &
#python hypergcn.py --mediators True --split 1 --depth 16 --data coauthorship --dataset cora --gpu 0
#python hypergcn.py --mediators True --split 1 --depth 16 --data coauthorship --dataset dblp --gpu 0
#
#python hypergcn.py --mediators True --split 1 --depth 8 --data cocitation --dataset citeseer --gpu 2 &
#python hypergcn.py --mediators True --split 1 --depth 8 --data cocitation --dataset cora --gpu 1 &
#python hypergcn.py --mediators True --split 1 --depth 8 --data coauthorship --dataset cora --gpu 0 &
#python hypergcn.py --mediators True --split 1 --depth 8 --data coauthorship --dataset dblp --gpu 0
#python hypergcn.py --mediators True --split 1 --depth 8 --data cocitation --dataset pubmed --gpu 1
#
#python hypergcn.py --mediators True --split 1 --depth 4 --data cocitation --dataset citeseer --gpu 2 &
#python hypergcn.py --mediators True --split 1 --depth 4 --data cocitation --dataset cora --gpu 2 &
#python hypergcn.py --mediators True --split 1 --depth 4 --data coauthorship --dataset cora --gpu 1 &
#python hypergcn.py --mediators True --split 1 --depth 4 --data cocitation --dataset pubmed --gpu 1 &
#python hypergcn.py --mediators True --split 1 --depth 4 --data coauthorship --dataset dblp --gpu 0 &

#python hypergcn.py --mediators True --split 1 --depth 2 --data cocitation --dataset citeseer --gpu 3 &
#python hypergcn.py --mediators True --split 1 --depth 2 --data cocitation --dataset cora --gpu 3 &
#python hypergcn.py --mediators True --split 1 --depth 2 --data coauthorship --dataset cora --gpu 3 &
#python hypergcn.py --mediators True --split 1 --depth 2 --data cocitation --dataset pubmed --gpu 3 &
#python hypergcn.py --mediators True --split 1 --depth 2 --data coauthorship --dataset dblp --gpu 3 &