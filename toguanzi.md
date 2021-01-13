# hypergraph laplacian emsemble learning(mix learning)
环境： pytorch 1.5+
## 代码normalization laplacian 来自readme.md 中的AAAI HGNN
## 跑19年 NIPS中的dataset
### 1.debug
A-19-NIPS-HyperGCN-A New Method of Training GraphConvolutional Networks on Hypergraphs

python train.py --activate_dataset coauthorship/cora --data_path ..data/data/hgnn/hypergcn --save_dir ../model/hgnn --print_freq 1 --nbaseblocklayer 1 --epochs 2 --inputlayer gcn --outputlayer gcn --gpu 7 --add_self_loop --hidden 32 --gpu 3 --type mutigcn --debug

注： --activate_dataset 可以换成  cocitation/cora  cocitation/citeseer cocitation/pubmed coauthorship/dblp; 后面两个可能会out of menory

### 2. train
python train.py --activate_dataset coauthorship/cora --data_path ..data/data/hgnn/hypergcn --save_dir ../model/hgnn --print_freq 1 --nbaseblocklayer 1 --epochs 200 --inputlayer gcn --outputlayer gcn --gpu 7 --add_self_loop --hidden 32 --gpu 3 --type mutigcn --debug 

注：即把---debug 去掉，会跑多个random_seed 然后取平均
## 跑19年AAAI dataset

python train.py --data_path ../data/hgnn --on_dataset NTU2012 --activate_dataset cora --gvcnn_feature_structure --use_gvcnn_feature --save_dir ../model/hgnn/gvcnn --is_probH --m_prob 1.22329227718 --add_self_loop --epochs 800 --seed 1000 --K_neigs 11 --type mutigcn --hidden 128 --mlp_hidden 128 --print_freq 30 --nhiddenlayer 1 --nbaseblocklayer 0 --lr 0.01 --dropout 0.204278 --weight_decay 0.000822278913618 --gpu 3 --inputlayer gcn --outputlayer gcn --debug

注：同样的去掉--debug 就是训练，可跑数据集： --activate_dataset citeseer --activate_dataset pubmed    

# todo
1. 仿照util/hypergraph_utils.py 中的def _generate_G_from_H()， 加入另外两种Laplacian的方式，计算Laplacian 矩阵，用于超图的Y=GXW 中的G。
其中G为laplacian matrix,X为n*m维的顶点特征，每行为一个node,W为可学习权重。
最后representation_all=(Y_1+Y_2+Y_3), y_i=G_iXW
2. 在 19NIPS数据集上先调调看
