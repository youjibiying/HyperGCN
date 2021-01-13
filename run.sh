#!bin/bash
#CUDA_VISIBLE_DEVICES=4 python train.py --type gcnii --learnable_adj --tensorboard --print_freq 20 --nbaseblocklayer 2 \
#--epochs 2000 --save_dir ../model/hgnn/NTU_mvcnn  --mvcnn_feature_structure\
##CUDA_VISIBLE_DEVICES=1 python train.py --type gcnii --nbaseblocklayer 2 --tensorboard --print_freq 20 --epochs 2000
# # gcnii

#python -u train.py --data_path ../data/hgnn \
#                  --on_dataset NTU2012 \
#                  --activate_dataset mvgnn \
#                  --mvcnn_feature_structure \
#                  --use_mvcnn_feature \
#                  --save_dir results \
#                  --is_probH \
#                  --m_prob 7.171234 \
#                  --debug \
#                  --epochs 1000 \
#                  --seed 500 \
#                  --K_neigs 10 \
#                  --type gcnii \
#                  --hidden 128 \
#                  --print_freq 20 \
#                  --sampling_percent 1 \
#                  --nhiddenlayer 1 \
#                  --nbaseblocklayer 2 \
#                  --lr 0.01 \
#                  --lamda 0.498782016389 \
#                  --dropout 0.329750074268 \
#                  --wd_adj 0.09421833 \
#                  --weight_decay 5e-4 \
#                  --gpu 1 \
#                  --inputlayer linear \
#                  --outputlayer linear \
##                  --learnable_adj
## citeseer
#python -u train.py --data_path ../data/hgnn \
#                  --on_dataset NTU2012  \
#                  --activate_dataset citeseer \
#                  --gvcnn_feature_structure \
#                  --use_gvcnn_feature \
#                  --save_dir ../model/hgnn/gvcnn \
#                  --is_probH \
#                  --m_prob 1.22329227718 \
#                  --add_self_loop \
#                  --epochs 250 \
#                  --seed 1000 \
#                  --K_neigs 11 \
#                  --type mutigcn \
#                  --hidden 64 \
#                  --mlp_hidden 64 \
#                  --print_freq 10 \
#                  --sampling_percent 1 \
#                  --nhiddenlayer 1 \
#                  --nbaseblocklayer 0 \
#                  --lr 0.01 \
#                  --lamda 0.0 \
#                  --dropout 0.461281552516 \
#                  --wd_adj 0.562556216748 \
#                  --weight_decay 0.00121068530475 \
#                  --gpu 4 \
#                  --lr_adj 0.0444225729618 \
#                  --adj_loss_coef 0.0190482023752 \
#                  --theta 1.0 \
#                  --inputlayer gcn \
#                  --outputlayer gcn \
##                  --learnable_adj \
##                  --debug \
##                  --learnable_adj \
##                  --attn_adj \


## cora
#python -u train.py --data_path ../data/hgnn \
#                  --on_dataset NTU2012  \
#                  --activate_dataset cora \
#                  --gvcnn_feature_structure \
#                  --use_gvcnn_feature \
#                  --save_dir ../model/hgnn/gvcnn \
#                  --is_probH \
#                  --m_prob 1.22329227718 \
#                  --add_self_loop \
#                  --epochs 500 \
#                  --seed 1000 \
#                  --K_neigs 11 \
#                  --type mutigcn \
#                  --hidden 128 \
#                  --mlp_hidden 128 \
#                  --print_freq 40 \
#                  --sampling_percent 1 \
#                  --nhiddenlayer 1 \
#                  --nbaseblocklayer 0 \
#                  --lr 0.01 \
#                  --lamda 0.0 \
#                  --dropout 0.204278 \
#                  --wd_adj 0.534579914794 \
#                  --weight_decay 0.000822278913618 \
#                  --gpu 3 \
#                  --theta 0.1 \
#                  --lr_adj 0.0281404388 \
#                  --adj_loss_coef 0.00800880392026 \
#                  --inputlayer gcn \
#                  --outputlayer gcn \
#                  --learnable_adj \
### cora of co authorship
#python -u train.py --data_path ../data/hgnn/hypergcn \
#                  --on_dataset NTU2012  \
#                  --activate_dataset coauthorship/cora \
#                  --gvcnn_feature_structure \
#                  --use_gvcnn_feature \
#                  --save_dir ../model/hgnn/gvcnn \
#                  --is_probH \
#                  --m_prob 1 \
#                  --add_self_loop \
#                  --epochs 300 \
#                  --seed 1000 \
#                  --K_neigs 11 \
#                  --type mutigcn \
#                  --hidden 128 \
#                  --mlp_hidden 128 \
#                  --print_freq 40 \
#                  --sampling_percent 1 \
#                  --nhiddenlayer 1 \
#                  --nbaseblocklayer 1 \
#                  --lr 0.01 \
#                  --lamda 0.0 \
#                  --dropout 0.204278 \
#                  --wd_adj 0.534579914794 \
#                  --weight_decay 0.0147796770816 \
#                  --gpu 4 \
#                  --theta 0.1 \
#                  --lr_adj 0.0281404388 \
#                  --adj_loss_coef 0.00800880392026 \
#                  --inputlayer gcn \
#                  --outputlayer gcn \
#                  --attn_adj \
#                  --learnable_adj \
##                  --debug \
## cora of co citation
#python -u train.py --data_path ../data/hgnn/hypergcn \
#                  --on_dataset NTU2012  \
#                  --activate_dataset cocitation/cora \
#                  --gvcnn_feature_structure \
#                  --use_gvcnn_feature \
#                  --save_dir ../model/hgnn/gvcnn \
#                  --is_probH \
#                  --m_prob 1 \
#                  --add_self_loop \
#                  --epochs 300 \
#                  --seed 1000 \
#                  --K_neigs 11 \
#                  --type mutigcn \
#                  --hidden 128 \
#                  --mlp_hidden 128 \
#                  --print_freq 40 \
#                  --sampling_percent 1 \
#                  --nhiddenlayer 1 \
#                  --nbaseblocklayer 1 \
#                  --lr 0.01 \
#                  --lamda 0.0 \
#                  --dropout 0.5 \
#                  --wd_adj 0.534579914794 \
#                  --weight_decay 0.0005 \
#                  --gpu 5 \
#                  --theta 0.1 \
#                  --lr_adj 0.0281404388 \
#                  --adj_loss_coef 0.00800880392026 \
#                  --inputlayer gcn \
#                  --outputlayer gcn \
#                  --attn_adj \
#                  --learnable_adj \
##                  --debug \

## citeseer of co-citation
#python -u train.py --data_path ../data/hgnn/hypergcn \
#                  --on_dataset NTU2012  \
#                  --activate_dataset cocitation/citeseer \
#                  --gvcnn_feature_structure \
#                  --use_gvcnn_feature \
#                  --save_dir ../model/hgnn/citeseer \
#                  --is_probH \
#                  --m_prob 1 \
#                  --add_self_loop \
#                  --epochs 300 \
#                  --seed 1000 \
#                  --K_neigs 11 \
#                  --type mutigcn \
#                  --hidden 64 \
#                  --mlp_hidden 64 \
#                  --print_freq 40 \
#                  --sampling_percent 1 \
#                  --nhiddenlayer 1 \
#                  --nbaseblocklayer 1 \
#                  --lr 0.01 \
#                  --lamda 0.0 \
#                  --dropout 0.496314457945 \
#                  --wd_adj 0.534579914794 \
#                  --weight_decay 0.029612134041 \
#                  --gpu 3 \
#                  --theta 0.1 \
#                  --lr_adj 0.0281404388 \
#                  --adj_loss_coef 0.00800880392026 \
#                  --inputlayer gcn \
#                  --outputlayer gcn \
#                  --learnable_adj \
#                  --attn_adj \
#                  --learnable_adj \
#                  --attn_adj \
##                  --debug \

##
## NTU2012
#python -u train.py --data_path ../data/hgnn \
#                  --on_dataset NTU2012 \
#                  --activate_dataset mvcnn \
#                  --mvcnn_feature_structure \
#                  --use_mvcnn_feature \
#                  --save_dir ../model/hgnn/mvcnn \
#                  --is_probH \
#                  --m_prob 1.0 \
#                  --add_self_loop \
#                  --epochs 600 \
#                  --seed 1000 \
#                  --K_neigs 10 \
#                  --type mutigcn \
#                  --hidden 128 \
#                  --print_freq 30 \
#                  --sampling_percent 1 \
#                  --nhiddenlayer 1 \
#                  --nbaseblocklayer 0 \
#                  --lr 0.001 \
#                  --lamda 0.0 \
#                  --dropout 0.5 \
#                  --theta 0.2 \
#                  --early_stopping 300 \
#                  --wd_adj 0.2 \
#                  --weight_decay 0.005 \
#                  --gpu 7 \
#                  --lr_adj 0.001 \
#                  --adj_loss_coef 0.1 \
#                  --inputlayer gcn \
#                  --outputlayer gcn \
##                  --learnable_adj
python -u train.py --data_path ../data/hgnn \
                  --on_dataset NTU2012 \
                  --activate_dataset gvcnn \
                  --mvcnn_feature_structure \
                  --use_gvcnn_feature \
                  --save_dir ../model/hgnn/gvcnn \
                  --is_probH \
                  --m_prob 1.0 \
                  --add_self_loop \
                  --epochs 800 \
                  --seed 1000 \
                  --K_neigs 10 \
                  --type mutigcn \
                  --hidden 128 \
                  --mlp_hidden 128 \
                  --print_freq 50 \
                  --sampling_percent 1 \
                  --nhiddenlayer 1 \
                  --nbaseblocklayer 0 \
                  --lr 0.001 \
                  --lamda 0.0 \
                  --dropout 0.5 \
                  --theta 0.05 \
                  --early_stopping 300 \
                  --wd_adj 0.2 \
                  --weight_decay 0.005 \
                  --gpu 6 \
                  --lr_adj 0.001 \
                  --adj_loss_coef 0.1 \
                  --inputlayer gcn \
                  --outputlayer gcn \
                  --learnable_adj \
#                  --use_mvcnn_feature \

                  #                  --gvcnn_feature_structure \
