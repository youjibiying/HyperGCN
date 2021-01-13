# --------------------------------------------------------
import numpy as np
import scipy.sparse as sp

import time

def Eu_dis(x):
    """
    Calculate the distance among each raw of x
    :param x: N X D
                N: the object number
                D: Dimension of the feature
    :return: N X N distance matrix
    """
    # np.multiply(np.mat(A), np.mat(B))  # 矩阵对应元素位置相乘，利用np.mat()将数组转换为矩阵
    x = np.mat(x)
    aa = np.sum(np.multiply(x, x), 1)
    ab = x * x.T
    dist_mat = aa + aa.T - 2 * ab
    dist_mat[dist_mat < 0] = 0
    dist_mat = np.sqrt(dist_mat)
    dist_mat = np.maximum(dist_mat, dist_mat.T)
    return dist_mat


def Cosine_dis(x):
    from numpy.linalg import norm
    # from sklearn.preprocessing import normalize as norm
    features1 = features2 = x
    norm1 = norm(features1, axis=-1).reshape(features1.shape[0], 1)
    norm2 = norm(features2, axis=-1).reshape(1, features2.shape[0])
    end_norm = np.dot(norm1, norm2)
    cos = np.dot(features1, features2.T) / end_norm
    similarity = 0.5 * cos + 0.5
    return np.mat(similarity)


def feature_concat(*F_list, normal_col=False):
    """
    Concatenate multiple modality feature. If the dimension of a feature matrix is more than two,
    the function will reduce it into two dimension(using the last dimension as the feature dimension,
    the other dimension will be fused as the object dimension)
    :param F_list: Feature matrix list
    :param normal_col: normalize each column of the feature
    :return: Fused feature matrix
    """
    features = None
    for f in F_list:
        if f is not None and f != []:
            # deal with the dimension that more than two
            if len(f.shape) > 2:
                f = f.reshape(-1, f.shape[-1])
            # normal each column
            if normal_col:
                f_max = np.max(np.abs(f), axis=0)
                f = f / f_max
            # facing the first feature matrix appended to fused feature matrix
            if features is None:
                features = f
            else:
                features = np.hstack((features, f))
    if normal_col:
        features_max = np.max(np.abs(features), axis=0)
        features = features / features_max
    return features


def hyperedge_concat(*H_list):
    """
    Concatenate hyperedge group in H_list
    :param H_list: Hyperedge groups which contain two or more hypergraph incidence matrix
    :return: Fused hypergraph incidence matrix
    """
    H = None
    for h in H_list:
        if h is not None and h != []:
            # for the first H appended to fused hypergraph incidence matrix
            if H is None:
                H = h
            else:
                if type(h) != list:
                    H = np.hstack((H, h))
                else:
                    tmp = []
                    for a, b in zip(H, h):
                        tmp.append(np.hstack((a, b)))
                    H = tmp
    return H


def generate_G_from_H(H, variable_weight=False):
    """
    calculate G from hypgraph incidence matrix H
    :param H: hypergraph incidence matrix H
    :param variable_weight: whether the weight of hyperedge is variable
    :return: G
    """
    if type(H) != list:
        return _generate_G_from_H(H, variable_weight)  # D_v^1/2 H W D_e^-1 H.T D_v^-1/2

        #return _generate_G_from_H_4(H, variable_weight)  # D_v^-1-H D_e^-1 HT

        #return _generate_G_from_H_5(H, variable_weight)  # D_v^-1-H D_e^-1 HT

        #return _generate_G_from_H(H, variable_weight)+ _generate_G_from_H_4(H, variable_weight)+ _generate_G_from_H_5(H, variable_weight)+_generate_G_from_H_7(H, variable_weight)# D_v^-1-H D_e^-1 HT
    else:
        G = []
        for sub_H in H:
            G.append(generate_G_from_H(sub_H, variable_weight))
        return G


def _generate_G_from_H(H, variable_weight=False):
    """
    calculate G from hypgraph incidence matrix H
    :param H: hypergraph incidence matrix H
    :param variable_weight: whether the weight of hyperedge is variable
    :return: G
    """
    H = np.array(H)
    n_edge = H.shape[1]  # 4024
    # the weight of the hyperedge
    W = np.ones(n_edge)  # 使用权重为1
    # the degree of the node
    DV = np.sum(H * W, axis=1)  # [2012]数组*是对应位置相乘，矩阵则是矩阵乘法 https://blog.csdn.net/TeFuirnever/article/details/88915383
    # the degree of the hyperedge
    DE = np.sum(H, axis=0)  # [4024]

    invDE = np.mat(np.diag(np.power(DE, -1)))
    invDE[np.isinf(invDE)] = 0  # D_e ^-1
    invDV = np.power(DV, -0.5)
    invDV[np.isinf(invDV)] = 0
    DV2 = np.mat(np.diag(invDV))  # D_v^-1/2

    W = np.mat(np.diag(W))
    H = np.mat(H)
    HT = H.T

    if variable_weight:
        DV2_H = DV2 * H
        invDE_HT_DV2 = invDE * HT * DV2
        return DV2_H, W, invDE_HT_DV2
    else:
        G = DV2 * H * W * invDE * HT * DV2  # D_v^1/2 H W D_e^-1 H.T D_v^-1/2
        return G


def _generate_G_from_H_4(H, variable_weight=False):
    """
    calculate G from hypgraph incidence matrix H
    :param H: hypergraph incidence matrix H
    :param variable_weight: whether the weight of hyperedge is variable
    :return: G
    """
    H = np.array(H)
    DV = np.sum(H, axis=1)
    DV=np.mat(DV)
    DE = np.sum(H, axis=0)
    invDE = np.mat(np.diag(np.power(DE, -1)))
    invDE[np.isinf(invDE)] = 0  # D_e ^-1

    H = np.mat(H)
    HT = H.T

    G = DV - H * invDE * HT
    return G


def _generate_G_from_H_5(H, variable_weight=False):
    """
    calculate G from hypgraph incidence matrix H
    :param H: hypergraph incidence matrix H
    :param variable_weight: whether the weight of hyperedge is variable
    :return: G
    """
    H = np.array(H)
    H = H.astype(int)
    # start1=time.time()
    # W1 = np.zeros([H.shape[0],H.shape[0]])
    # for i in range(H.shape[0]):
    #     for j in range(H.shape[0]):
    #         for k in range(H.shape[1]):
    #             if H[i][k]==1 and H[j][k]==1:
    #                 W[i][j]+=1
    # end1=time.time()
    #
    # print(f'iterative time: {start1-end1}')

    start2 = time.time()
    HC = np.array([H]*H.shape[0])
    HCT = HC.swapaxes(0,1).copy()

    HC= HC.reshape(H.shape[0]**2,H.shape[1])
    HCT = HCT.reshape(H.shape[0] ** 2, H.shape[1])

    HAND=HC & HCT


    W = np.sum(HAND,axis=1).reshape(H.shape[0],H.shape[0])

    end2 = time.time()

    print(f'iterative time: {start2 - end2}')

    # if W1==W:
    #     print('W equal')
    DV = np.sum(W, axis=1)
    DV = np.mat(DV)
    H = np.mat(H)
    HT = H.T

    G = DV-H*HT
    return G


def _generate_G_from_H_6(H, variable_weight=False):
    """
    calculate G from hypgraph incidence matrix H
    :param H: hypergraph incidence matrix H
    :param variable_weight: whether the weight of hyperedge is variable
    :return: G
    """
    H = np.array(H)
    n_edge = H.shape[1]  # 4024
    # the weight of the hyperedge
    W = np.ones(n_edge)  # 使用权重为1
    # the degree of the node
    DV = np.sum(H * W, axis=1)  # [2012]数组*是对应位置相乘，矩阵则是矩阵乘法 https://blog.csdn.net/TeFuirnever/article/details/88915383
    # the degree of the hyperedge
    DE = np.sum(H, axis=0)  # [4024]

    invDE = np.mat(np.diag(np.power(DE, -1)))
    invDE[np.isinf(invDE)] = 0  # D_e ^-1
    invDV = np.power(DV, -0.5)
    invDV[np.isinf(invDV)] = 0
    DV2 = np.mat(np.diag(invDV))  # D_v^-1/2

    W = np.mat(np.diag(W))
    H = np.mat(H)
    HT = H.T

    if variable_weight:
        DV2_H = DV2 * H
        invDE_HT_DV2 = invDE * HT * DV2
        return DV2_H, W, invDE_HT_DV2
    else:
        G = DV2 * H * W * invDE * HT * DV2  # D_v^1/2 H W D_e^-1 H.T D_v^-1/2
        return G


def _generate_G_from_H_7(H, variable_weight=False):
    """
    calculate G from hypgraph incidence matrix H
    :param H: hypergraph incidence matrix H
    :param variable_weight: whether the weight of hyperedge is variable
    :return: G
    """
    H = np.array(H)
    HT = H.T

    G = np.dot(H,HT)
    G = np.mat(G)
    return G


def construct_H_with_KNN_from_distance(dis_mat, k_neig, is_probH=True, m_prob=1, percent=1):
    """
    construct hypregraph incidence matrix from hypergraph node distance matrix
    :param dis_mat: node distance matrix
    :param k_neig: K nearest neighbor
    :param is_probH: prob Vertex-Edge matrix or binary
    :param m_prob: prob
    :return: N_object X N_hyperedge
    """
    # if percent >= 1.0:  # percent=0.05
    #     return self.stub_sampler(normalization, cuda)

    n_obj = dis_mat.shape[0]
    # construct hyperedge from the central feature space of each node
    n_edge = n_obj
    H = np.zeros((n_obj, n_edge))
    nnz = n_edge  # 2012条边
    if percent <= 1:
        perm = np.random.permutation(nnz)  # 随机排列
        preserve_nnz = int(nnz * percent)  # 取出比例为100
        perm = perm[:preserve_nnz]
    else:
        raise ValueError('percent should not larger than 1')

    for center_idx in range(n_obj):
        if center_idx not in perm:  # drop edge
            continue
        # dis_mat[center_idx, center_idx] = 0
        dis_vec = dis_mat[center_idx]
        nearest_idx = np.array(np.argsort(-dis_vec)).squeeze()  # argsort函数返回的是数组值从小到大的索引值
        avg_dis = np.average(dis_vec)
        if not np.any(nearest_idx[:k_neig] == center_idx):  # any()查看两矩阵是否有一个对应元素相等
            nearest_idx[k_neig - 1] = center_idx  # 如果k领域里没有当前的核心点，则令最后一个最后当前核心点

        for node_idx in nearest_idx[:k_neig]:
            # if node_idx in perm: # drop node
            #     continue
            if is_probH:
                H[node_idx, center_idx] = np.exp(-dis_vec[0, node_idx] ** 2 / (m_prob * avg_dis) ** 2)  # 产生概率阵
            else:
                H[node_idx, center_idx] = 1.0
    H = H[:, perm]
    return H


def cal_distance_map(X):
    """
    init multi-scale hypergraph Vertex-Edge matrix from original node feature matrix
    :param X: N_object x feature_number
    :param K_neigs: the number of neighbor expansion
    :param split_diff_scale: whether split hyperedge group at different neighbor scale
    :param is_probH: prob Vertex-Edge matrix or binary
    :param m_prob: prob
    :return: N_object x N_hyperedge
    """
    if len(X.shape) != 2:
        X = X.reshape(-1, X.shape[-1])
    # dis_mat = Eu_dis(X)  # 计算欧式距离
    dis_mat = Cosine_dis(X)
    return dis_mat


def construct_H_with_KNN(dis_mat, K_neigs, split_diff_scale=False, is_probH=True, m_prob=1, percent=1):
    if type(K_neigs) == int:
        K_neigs = [K_neigs]
    H = []
    for k_neig in K_neigs:  # 如果使用多个领域构成的图，则下面会拼接在一起
        H_tmp = construct_H_with_KNN_from_distance(dis_mat, k_neig, is_probH, m_prob, percent=percent)
        if not split_diff_scale:
            H = hyperedge_concat(H, H_tmp)  # H 拼接
        else:
            H.append(H_tmp)
    return H


def _edge_dict_to_H(edge_dict, percent=1):
    """
    calculate H from edge_list
    :param edge_dict: edge_list[i] = adjacent indices of index i
    :return: H, (n_nodes, n_nodes) numpy ndarray
    """
    n_nodes = len(edge_dict)
    H = np.zeros(shape=(n_nodes, n_nodes))
    nnz = n_nodes  # 10556条边
    perm = np.random.permutation(nnz)  # 随机排列
    preserve_nnz = int(nnz * percent)  # 取出比例为527条
    perm = perm[:preserve_nnz]  # 对随机排列的边取出 前527条 array([1950, 8129, 8315, 1360,..., 7676, 9025, 5400])
    for center_id, adj_list in enumerate(edge_dict):
        if center_id not in perm:
            continue
        H[center_id, center_id] = 1.0
        for adj_id in adj_list:
            H[adj_id, center_id] = 1.0
    H = H[:, perm]
    return H


def adj_to_H(adj, percent=1):
    adj = sp.coo_matrix(adj)
    if percent >= 1.0:  # percent=0.05
        raise ValueError(f"wrong")

    nnz = adj.nnz  # 10556条边
    perm = np.random.permutation(nnz)  # 随机排列
    preserve_nnz = int(nnz * percent)  # 取出比例为527条
    perm = perm[:preserve_nnz]  # 对随机排列的边取出 前527条 array([1950, 8129, 8315, 1360,..., 7676, 9025, 5400])
    r_adj = sp.coo_matrix((adj.data[perm],  # coo矩阵中[row ,col. data] 这三个向量分别存储，维度 train_adj.data [1,10556]
                           (adj.row[perm],  # 维度 train_adj.row [1,10556]
                            adj.col[perm])),  # train_adj.col [1,10556]
                          shape=adj.shape)  # 构建只有preserve_nzz条边的sparse adj
    H = r_adj.todense()  # 标准化 在normalization.py中 # [2708,2708]
    H = H.T
    return np.array(H, dtype=np.float32)  # 返回进行处理后的


def aug_normalized_adjacency(adj):  # A' = (D + I)^-1/2 * ( A + I ) * (D + I)^-1/2
    adj = adj + sp.eye(adj.shape[0])
    adj = sp.coo_matrix(adj)
    row_sum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(row_sum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()


def aug_normalized_adjacency_from_H(H):
    DV = np.sum(H, 1)
    adj = H.dot(H.T) - np.diag(DV)
    adj = np.sign(adj)
    G = aug_normalized_adjacency(adj)
    return G.todense()
