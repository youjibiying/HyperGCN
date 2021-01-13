python train.py --activate_dataset coauthorship/cora --data_path ../data/data/hgnn/hypergcn
--save_dir ../model/hgnn --print_freq 10 --nbaseblocklayer 32 --epochs 1000 --inputlayer gcn
--outputlayer gcn --gpu 7 --add_self_loop --hidden 256 --gpu 0 --type gcnii --debug

python train.py --activate_dataset coauthorship/cora --data_path ../data/data/hgnn/hypergcn
--save_dir ../model/hgnn --print_freq 10 --nbaseblocklayer 2 --epochs 1000 --inputlayer dense
--outputlayer dense --add_self_loop --hidden 256 --gpu 0 --type gcnii --tensorboard --lr 0.001


--activate_dataset
coauthorship/cora
--data_path
../data/data/hgnn/hypergcn
--save_dir
../model/hgnn
--print_freq
10
--nbaseblocklayer
32
--epochs
1000
--inputlayer
gcn
--outputlayer
gcn
--gpu
7
--add_self_loop
--hidden
256
--gpu
0
--type
gcnii
--debug
--tensorboard
True
--lr
0.001