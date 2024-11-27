import numpy as np
import scipy.sparse as sp
import torch
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

n_users = 0
n_items = 0
dataset = ''
train_user_set = defaultdict(list)
test_user_set = defaultdict(list)
valid_user_set = defaultdict(list)


def read_cf_amazon(file_name):
    return np.loadtxt(file_name, dtype=np.int32)  # [u_id, i_id]


def read_cf_yelp2018(file_name):
    inter_mat = list()
    lines = open(file_name, "r").readlines()
    for l in lines:
        tmps = l.strip()
        inters = [int(i) for i in tmps.split(" ")]
        u_id, pos_ids = inters[0], inters[1:]
        pos_ids = list(set(pos_ids))
        for i_id in pos_ids:
            inter_mat.append([u_id, i_id])
    return np.array(inter_mat)


def load_pretrained_embeddings(pretrain_path):
    """加载预训练嵌入"""
    try:
        data = torch.load(pretrain_path)
        # 加载drug和target的嵌入及其ID映射
        drug_embeddings = torch.FloatTensor(data['drug']['embeddings'])
        target_embeddings = torch.FloatTensor(data['target']['embeddings'])
        drug_ids = data['drug']['ids']
        target_ids = data['target']['ids']
        
        print("Detailed embedding information:")
        print(f"Number of unique drug IDs: {len(set(drug_ids))}")
        print(f"Number of unique target IDs: {len(set(target_ids))}")
        print(f"Drug embeddings shape: {drug_embeddings.shape}")
        print(f"Target embeddings shape: {target_embeddings.shape}")
        print(f"Sample drug IDs: {drug_ids[:5]}")
        print(f"Sample target IDs: {target_ids[:5]}")
        
        # 创建ID到索引的映射字典
        drug_id_to_idx = {id_: idx for idx, id_ in enumerate(drug_ids)}
        target_id_to_idx = {id_: idx for idx, id_ in enumerate(target_ids)}
        
        print("\nDetailed mapping information:")
        print("Drug ID to Index mapping (first 5 entries):")
        first_5_drug_items = list(drug_id_to_idx.items())[:5]
        for orig_id, mapped_idx in first_5_drug_items:
            print(f"Original Drug ID: {orig_id} -> Index: {mapped_idx}")
            
        print("\nTarget ID to Index mapping (first 5 entries):")
        first_5_target_items = list(target_id_to_idx.items())[:5]
        for orig_id, mapped_idx in first_5_target_items:
            print(f"Original Target ID: {orig_id} -> Index: {mapped_idx}")
            
        print("\nMapping Statistics:")
        print(f"Number of drug mappings: {len(drug_id_to_idx)}")
        print(f"Number of target mappings: {len(target_id_to_idx)}")
        print(f"Drug ID range: {min(drug_ids)} - {max(drug_ids)}")
        print(f"Target ID range: {min(target_ids)} - {max(target_ids)}")
        print(f"Drug embeddings shape: {drug_embeddings.shape}")
        print(f"Target embeddings shape: {target_embeddings.shape}")
        
        return drug_embeddings, target_embeddings, drug_id_to_idx, target_id_to_idx
    except Exception as e:
        print(f"Error loading pretrained embeddings: {e}")
        return None, None, {}, {}


def build_sparse_graph(train_cf, drug_id_to_idx=None, target_id_to_idx=None):
    def _bi_norm_lap(adj):
        # D^{-1/2}AD^{-1/2}
        rowsum = np.array(adj.sum(1))

        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

        bi_lap = d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)
        return bi_lap.tocoo()

    def _si_norm_lap(adj):
        # D^{-1}A
        rowsum = np.array(adj.sum(1))

        d_inv = np.power(rowsum, -1).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)

        norm_adj = d_mat_inv.dot(adj)
        return norm_adj.tocoo()

    # 1. 构建用户-物品交互矩阵
    cf = train_cf.copy()
    cf[:, 1] = cf[:, 1] + n_users  # [0, n_items) -> [n_users, n_users+n_items) # 将item的id映射到[n_users, n_users+n_items)范围
    
    # 2. 构建双向图(用户->物品, 物品->用户)
    cf_ = cf.copy()
    cf_[:, 0], cf_[:, 1] = cf[:, 1], cf[:, 0]  # user->item, item->user

    # diag = np.array([[i, i] for i in range(n_users+n_items)])
    # cf_ = np.concatenate([cf, cf_, diag], axis=0)  # [[0, R], [R^T, 0]] + I
    cf_ = np.concatenate([cf, cf_], axis=0)  # [[0, R], [R^T, 0]]

    # 3. 创建稀疏矩阵
    vals = [1.] * len(cf_)
    mat = sp.coo_matrix((vals, (cf_[:, 0], cf_[:, 1])), shape=(n_users+n_items, n_users+n_items))
    
    # 4. 计算归一化的拉普拉斯矩阵
    return _bi_norm_lap(mat)


def load_data(model_args):
    global args, dataset, n_users, n_items
    args = model_args
    dataset = args.dataset
    directory = args.data_path + dataset + '/'

    print('reading train and test user-item set ...')
    train_cf = read_cf_amazon(directory + 'train.txt')
    test_cf = read_cf_amazon(directory + 'test.txt')
    valid_cf = read_cf_amazon(directory + 'valid.txt')
    
    # 加载预训练嵌入
    pretrain_drug_emb, pretrain_target_emb = None, None
    drug_id_to_idx, target_id_to_idx = {}, {}
    if args.use_pretrain:
        print('\nLoading pretrained embeddings...')
        pretrain_drug_emb, pretrain_target_emb, drug_id_to_idx, target_id_to_idx = \
            load_pretrained_embeddings(args.pretrain_path)
        if pretrain_drug_emb is not None:
            print(f"Loaded pretrained embeddings with shapes: {pretrain_drug_emb.shape}, {pretrain_target_emb.shape}")
    
    # 统计用户和物品数量
    n_users = max(train_cf[:, 0].max() + 1, len(drug_id_to_idx))
    n_items = max(train_cf[:, 1].max() + 1, len(target_id_to_idx))

    # 构建用户-物品集合
    for u, i in train_cf:
        train_user_set[u].append(i)
    for u, i in test_cf:
        test_user_set[u].append(i)
    for u, i in valid_cf:
        valid_user_set[u].append(i)
    
    norm_mat = build_sparse_graph(train_cf)
    
    n_params = {
        'n_users': int(n_users),
        'n_items': int(n_items),
        'pretrain_drug_emb': pretrain_drug_emb,
        'pretrain_target_emb': pretrain_target_emb,
        'drug_id_to_idx': drug_id_to_idx,
        'target_id_to_idx': target_id_to_idx
    }
    
    user_dict = {
        'train_user_set': train_user_set,
        'valid_user_set': valid_user_set,
        'test_user_set': test_user_set,
    }

    return train_cf, user_dict, n_params, norm_mat

