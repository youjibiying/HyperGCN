#!bin/bash
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
#                  --epochs 230 \
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
#                  --learnable_adj \
#                  --debug \
#                  --attn_adj \g
#                  --learnable_adj \

## cora
python -u train.py --data_path ../data/hgnn \
                  --on_dataset NTU2012  \
                  --activate_dataset cora \
                  --gvcnn_feature_structure \
                  --use_gvcnn_feature \
                  --save_dir ../model/hgnn/gvcnn \
                  --is_probH \
                  --m_prob 1.22329227718 \
                  --add_self_loop \
                  --epochs 1000 \
                  --seed 1000 \
                  --K_neigs 11 \
                  --type mutigcn \
                  --hidden 128 \
                  --mlp_hidden 128 \
                  --print_freq 30 \
                  --nhiddenlayer 1 \
                  --nbaseblocklayer 0 \
                  --lr 0.01 \
                  --dropout 0.204278 \
                  --weight_decay 0.000822278913618 \
                  --gpu 3 \
                  --inputlayer gcn \
                  --outputlayer gcn \

#python -u train.py --data_path ../data/hgnn \
#                  --on_dataset NTU2012 \
#                  --activate_dataset mvcnn \
#                  --mvcnn_feature_structure \
#                  --use_mvcnn_feature \
#                  --save_dir ../model/hgnn/mvcnn \
#                  --m_prob 0.441049784585 \
#                  --add_self_loop \
#                  --debug \
#                  --epochs 1000 \
#                  --seed 1000 \
#                  --K_neigs 10 \
#                  --type mutigcn \
#                  --hidden 64 \
#                  --print_freq 30 \
#                  --sampling_percent 1 \
#                  --nhiddenlayer 1 \
#                  --nbaseblocklayer 0 \
#                  --lr 0.01 \
#                  --lamda 0.0 \
#                  --dropout 0.155774164268 \
#                  --wd_adj 0.175308183519 \
#                  --weight_decay 0.0646163893685 \
#                  --gpu 7 \
#                  --lr_adj 0.001 \
#                  --adj_loss_coef 0.1 \
#                  --theta 0.05 \
#                  --inputlayer gcn \
#                  --outputlayer gcn \
#                  --debug \
#                  --learnable_adj