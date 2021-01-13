import os
import time
import copy
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import pprint as pp
import utils.hypergraph_utils as hgut
from models.model import GCNModel, MultiLayerHGNN, HGNN
# from config import get_config
from datasets import load_feature_construct_H, randomedge_sample_H
from datasets import source_select
from parsing import train_args
from torch.nn.utils import clip_grad_norm_
from utils import hypergraph_utils as hgut
import scipy.sparse as sp
from datasets import data

import csv
# try:
#     from apex import amp
#     import torch.distributed as dist
#     from apex.parallel import DistributedDataParallel
#
#     dist.init_process_group(backend='nccl')
#     mutigpu=True
# except:
#     mutigpu=False
#     pass
import scipy

args = train_args()
# cfg = get_config('config/config.yaml', args)
# os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
device = torch.device('cuda', args.gpu) if torch.cuda.is_available() else torch.device('cpu')


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    # sparse_mx = sp.coo_matrix(sparse_mx)
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape).to(device)


def normalization(dist):
    if args.is_probH:
        dist = np.array(dist)
        dist = np.exp(-dist.T ** 2 / (
                args.m_prob * dist.mean(1)) ** 2).T  # 对每一行进行标准化，一行表示一条超边 A_ij=exp( -D_ij*2 / m_pro*mean(D[i,:])*2 )
    else:
        dist = -dist / args.m_prob
    return dist


# initialize data
data_dir = args.modelnet40_ft if args.on_dataset == 'ModelNet40' \
    else args.ntu2012_ft
adj = None
dist = None

if args.activate_dataset in ['cora', 'pubmed', 'citeseer']:
    source = source_select(args)
    print(f'Using {args.activate_dataset} dataset')
    fts, lbls, idx_train, idx_val, idx_test, n_category, adj, edge_dict = source(args)
    if args.add_self_loop:
        adj = adj.todense() + np.eye(adj.shape[0])
    else:
        adj = adj.todense()
    args.K_neigs = list(np.array(adj, dtype=np.int8).sum(0))

    adj = sp.coo_matrix(adj)
    dist = -adj.todense() + 1

    # dist = hgut.cal_distance_map(fts)
elif args.activate_dataset.startswith('coauthorship') or args.activate_dataset.startswith('cocitation'):
    dataset, idx_train, idx_test = data.load(args)
    idx_val = idx_test
    hypergraph, fts, lbls = dataset['hypergraph'], dataset['features'], dataset['labels']
    lbls = np.argmax(lbls, axis=1)
    H = np.zeros((dataset['n'], len(hypergraph)))
    for i, (a, p) in enumerate(hypergraph.items()):
        H[list(p), i] = 1
    adj = H
    # args.K_neigs = list(np.array(adj, dtype=np.int8).sum(0))

else:
    fts, lbls, idx_train, idx_test, mvcnn_dist, gvcnn_dist = \
        load_feature_construct_H(data_dir,
                                 m_prob=args.m_prob,
                                 K_neigs=args.K_neigs,
                                 is_probH=args.is_probH,
                                 use_mvcnn_feature=args.use_mvcnn_feature,
                                 use_gvcnn_feature=args.use_gvcnn_feature,
                                 use_mvcnn_feature_for_structure=args.mvcnn_feature_structure,
                                 use_gvcnn_feature_for_structure=args.gvcnn_feature_structure)
    idx_val = idx_test
    if args.mvcnn_feature_structure:
        dist = mvcnn_dist
    else:
        dist = gvcnn_dist
# dist = normalization(dist)


# G = hgut.generate_G_from_H(H) # D_v^1/2 H W D_e^-1 H.T D_v^-1/2 :

n_class = int(lbls.max()) + 1

# transform data to device
fts = torch.Tensor(fts).to(device)  # features -> fts
lbls = torch.Tensor(lbls).squeeze().long().to(device)
# G = torch.Tensor(G).to(device)
idx_train = torch.Tensor(idx_train).long().to(device)
idx_test = torch.Tensor(idx_test).long().to(device)
idx_val = torch.Tensor(idx_val).long().to(device)
# dist = sp.coo_matrix(dist)
# dist= normalization(dist)
# dist = torch.Tensor(dist).to(device)  # features -> fts


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def train_model(model, criterion, optimizer, scheduler, num_epochs=25, print_freq=500, dist=None, args=None):
    time_start = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
    if args.tensorboard:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(
            os.path.join(args.save_dir, f'{time_start}'))
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc, best_epoch, best_val_loss = -1, 0, -1.0

    if args.activate_dataset in ['cora', 'pubmed', 'citeseer']:  # H，G 均为稀疏矩阵
        H = adj.todense()
        G = hgut.generate_G_from_H(H)
        # G = sparse_mx_to_torch_sparse_tensor(G)
    elif args.activate_dataset.startswith('coauthorship') or args.activate_dataset.startswith('cocitation'):
        H = adj
        G = hgut.generate_G_from_H(H)

    else:

        H = randomedge_sample_H(mvcnn_dist=mvcnn_dist,
                                gvcnn_dist=gvcnn_dist,
                                split_diff_scale=False,
                                m_prob=args.m_prob,
                                K_neigs=args.K_neigs,
                                is_probH=args.is_probH,
                                percent=args.sampling_percent,
                                use_mvcnn_feature_for_structure=args.mvcnn_feature_structure,
                                use_gvcnn_feature_for_structure=args.gvcnn_feature_structure)

        # H_target = torch.tensor(H, dtype=torch.float32).to(device)
        # G = hgut.generate_G_from_H(H)  # D_v^1/2 H W D_e^-1 H.T D_v^-1/2 :
        G = hgut.generate_G_from_H(H)  # D_v^1/2 H W D_e^-1 H.T D_v^-1/2 :


    # H = torch.Tensor(H).half().to(device)
    # G = torch.Tensor(G).half().to(device)
    if args.type == 'gcn':
        G = hgut.aug_normalized_adjacency_from_H(H)

    H = torch.Tensor(H).to(device)
    G = torch.Tensor(G).to(device)
    H_target = G  # torch.tensor(G_target, dtype=torch.half).to(device)
    # dist = sparse_mx_to_torch_sparse_tensor(dist)
    # loss_adj_f = torch.nn.L1Loss(reduction='mean')
    loss_adj_f = torch.nn.MSELoss(reduction='sum')
    for epoch in range(num_epochs):

        if epoch % print_freq == 0:
            print('-' * 10)
            print(f'Epoch {epoch}/{num_epochs - 1}')

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                # scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            idx = idx_train if phase == 'train' else idx_val  # idx_test

            # Iterate over data.
            optimizer.zero_grad()
            with torch.set_grad_enabled(phase == 'train'):

                outputs = model(fts, G=G, adj=H)

                loss = criterion(outputs[idx], lbls[idx])

                _, preds = torch.max(outputs, 1)
                # backward + optimize only if in training phase
                if phase == 'train':
                    # with amp.scale_loss(loss, optimizer) as scaled_loss:
                    #     scaled_loss.backward()
                    loss.backward()
                    clip_grad_norm_(model.parameters(), 40)
                    optimizer.step()
                    scheduler.step(loss)

            # statistics
            running_loss += loss.item() * fts.size(0)
            running_corrects += torch.sum(preds[idx] == lbls.data[idx])

            epoch_loss = running_loss / len(idx)
            epoch_acc = running_corrects.double() / len(idx)

            if epoch % print_freq == 0:

                print(
                    f'{phase} avgLoss: {epoch_loss:.4f} pre_loss: {loss:.4f}'
                    f'lr: {optimizer.param_groups[0]["lr"]} Acc: {epoch_acc:.4f}')

            # deep copy the model
            if phase == 'val':
                if epoch_acc > best_acc:
                    early_stop = 0
                    best_acc = epoch_acc
                    best_epoch = epoch
                    best_val_loss = loss
                    best_model_wts = copy.deepcopy(model.state_dict())
                else:
                    early_stop += 1

            if args.tensorboard:
                if phase == 'train':
                    writer.add_scalar('loss/train', loss, epoch)
                    writer.add_scalar('acc/train', epoch_acc, epoch)
                else:
                    writer.add_scalar('loss/val', loss, epoch)
                    writer.add_scalar('acc/val', epoch_acc, epoch)
                    writer.add_scalar('best_acc/val', best_acc, epoch)

        #data_write_csv('../data/data/acc-cora-m-4.csv',[epoch, loss.cpu().detach().numpy(), epoch_acc.cpu().detach().numpy(), best_acc.cpu().detach().numpy()])

        if epoch % print_freq == 0:
            print(f'Best val Acc: {best_acc:4f} at epoch: {best_epoch}')
            print('-' * 20)


        if early_stop > args.early_stopping:
            print('early stop at epoch', epoch)
            break

    time_elapsed = time.time() - since
    print(f'\nTraining complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # load best model weights
    model.load_state_dict(best_model_wts)
    model.eval()
    # test
    outputs = model(fts, adj=H, G=G)
    try:
        print(f'The last layers adj distance:{(H - model.adj.data).norm()}')
    except:
        pass
    # loss = criterion(outputs[idx_test], lbls[idx_test])
    _, preds = torch.max(outputs, 1)
    test_acc = torch.sum(preds[idx_test] == lbls.data[idx_test]).double() / len(idx_test)
    print(args)
    print(f"test_acc={test_acc}\n"
          f"best_val_acc={best_acc}\n"
          f"best_val_epoch={best_epoch}\n"
          f"best_val_loss={best_val_loss}")


    if args.tensorboard:
        writer.add_histogram('best_acc', test_acc)
    return model, float(test_acc)


def data_write_csv(filename,data):
    with open(filename, "a", encoding="utf-8", newline='') as w:
        writer = csv.writer(w)
        writer.writerow(data)

def _main():
    print(f"Classification on {args.on_dataset} dataset!!! class number: {n_class}")
    print(f"use MVCNN feature: {args.use_mvcnn_feature}")
    print(f"use GVCNN feature: {args.use_gvcnn_feature}")
    print(f"use MVCNN feature for structure: {args.mvcnn_feature_structure}")
    print(f"use GVCNN feature for structure: {args.gvcnn_feature_structure}")
    print(args)

    if args.type == 'HGCN':

        model = HGNN(in_ch=fts.shape[1],
                     n_class=n_class,
                     n_hid=args.hidden,
                     dropout=args.dropout,
                     )  # 两层卷积
    # model = MultiLayerHGNN(in_features=fts.shape[1], hidden_features=args.hidden, out_features=n_class,
    #                           nbaselayer=cfg['nbaseblocklayer'],
    #                           withbn=False, withloop=False, dropout=args.dropout,
    #                           aggrmethod="nores", dense=False)
    else:
        model = GCNModel(nfeat=fts.shape[1],
                         nhid=args.hidden,
                         nclass=n_class,
                         nhidlayer=args.nhiddenlayer,
                         dropout=args.dropout,
                         baseblock=args.type,
                         inputlayer=args.inputlayer,
                         outputlayer=args.outputlayer,
                         nbaselayer=args.nbaseblocklayer,
                         activation=F.relu,
                         withbn=args.withbn,
                         withloop=args.withloop,
                         aggrmethod=args.aggrmethod,
                         mixmode=args.mixmode,
                         args=args)
        # init_dist=scipy.special.softmax(adj.todense()))

    print(model)
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr,
                           weight_decay=args.weight_decay)
    # p1, p2 = [], [] # 用于设置不同的层使用不同的学习率
    # for n, p in model.named_parameters():
    #     # print(n, p.shape)
    #     if '.dynamic_adj.' in n:
    #         p2.append(p)
    #     else:
    #         p1.append(p)
    # optimizer = optim.Adam([
    #     {'params': p1, 'weight_decay': args.weight_decay, 'lr': args.lr},
    #     {'params': p2, 'weight_decay': args.wd_adj, 'lr': args.lr_adj}])

    # model, optimizer = amp.initialize(model, optimizer, opt_level="O0")  # 使用apex 混合精度训练，这里O2表示以float16为主
    # model = DistributedDataParallel(model)
    # optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=cfg['weight_decay)
    # schedular = optim.lr_scheduler.MultiStepLR(optimizer,
    #                                            milestones=args.milestones,
    #                                            gamma=args.gamma)
    schedular = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=100,
                                                           verbose=False,
                                                           threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0,
                                                           eps=1e-08)
    criterion = torch.nn.CrossEntropyLoss()

    model, test_acc = train_model(model, criterion, optimizer, schedular, args.epochs, print_freq=args.print_freq,
                                  dist=dist, args=args)
    return test_acc


if __name__ == '__main__':
    if args.activate_dataset.startswith('coauthor') or args.activate_dataset.startswith('cocitation'):
        setup_seed(args.seed)
        if args.debug:
            splits = [args.split]
        else:
            splits = [args.split + i for i in range(10)]  # 1000
        results = []
        for split in splits:
            print(f"split: {split}/{splits}")
            args.split = split
            results.append(_main())
            print(results)
    else:
        if args.debug:
            seed_nums = [args.seed]  # 1000
        else:
            seed_nums = [args.seed + i for i in range(10)]  # 1000
        results = []
        for seed_num in seed_nums:
            print(f"seed:{seed_num}/{seed_nums}")
            setup_seed(seed_num)
            results.append(_main())
            print(results)
    results = np.array(results)
    print(f"avg_test_acc={results.mean()} \n"
          f"std={results.std()}")
    #data_write_csv('../data/data/result.csv',[results.mean(), results.std(),args.type,args.nbaseblocklayer,args.hidden,args.lr,args.inputlayer])
    #data_write_csv('../data/data/result_v2.csv',[results.mean(), results.std(), args.type, args.nbaseblocklayer, args.hidden, args.lr,
                    #args.alpha, args.lamda,args.gvcnn_feature_structure,args.mvcnn_feature_structure,args.use_gvcnn_feature,args.use_mvcnn_feature])

    # #data_write_csv('../data/data/result_v3.csv',
    #                [results.mean(), results.std(), args.type, args.nbaseblocklayer, args.hidden, args.lr,
    #                 args.inputlayer, args.alpha,args.lamda])

    # data_write_csv('../data/data/result_v4.csv',[results.mean(), results.std(), args.type, args.nbaseblocklayer, args.hidden, args.lr,
    #                 args.alpha, args.lamda])
