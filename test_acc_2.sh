#!/bin/bash


python hypergcn.py --mediators True --split 1 --depth 16 --data cocitation --dataset citeseer
python hypergcn.py --mediators True --split 1 --depth 32 --data cocitation --dataset citeseer
python hypergcn.py --mediators True --split 1 --depth 64 --data cocitation --dataset citeseer
python hypergcn.py --mediators True --split 1 --depth 128 --data cocitation --dataset citeseer

