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

    # 直接读取原始数据，不进行ID映射
    raw_data = np.loadtxt(file_name, dtype=np.int32)  # [u_id, i_id]
    return raw_data


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

    try:
        data = torch.load(pretrain_path)
        drug_embeddings = torch.FloatTensor(data['drug']['embeddings'])
        target_embeddings = torch.FloatTensor(data['target']['embeddings'])
        drug_ids = data['drug']['ids']
        target_ids = data['target']['ids']
        
        print("pt-------------Detailed embedding information:")
        print(f"Number of unique drug IDs: {len(set(drug_ids))}")
        print(f"Number of unique target IDs: {len(set(target_ids))}")
        print(f"Drug embeddings shape: {drug_embeddings.shape}")
        print(f"Target embeddings shape: {target_embeddings.shape}")
        print(f"Sample drug IDs: {drug_ids[:5]}")
        print(f"Sample target IDs: {target_ids[:5]}")
        
        # 保持原始ID作为索引
        drug_id_to_idx = {id_: id_ for id_ in drug_ids}
        target_id_to_idx = {id_: id_ for id_ in target_ids}
        
        print("\npt-------------Mapping Statistics:")
        print(f"Drug ID range: 0 - {max(drug_ids)}")
        print(f"Target ID range: 0 - {max(target_ids)}")
        print(f"Number of drug embeddings: {len(drug_embeddings)}")
        print(f"Number of target embeddings: {len(target_embeddings)}")
        
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

    # 1. 构建户-物品交互矩阵
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
    
    # # 检查train_cf, test_cf, valid_cf的ID范围
    # print("\nChecking ID ranges in datasets:")
    # print(f"Train set - Drug ID range: [{train_cf[:, 0].min()}, {train_cf[:, 0].max()}], Target ID range: [{train_cf[:, 1].min()}, {train_cf[:, 1].max()}]")
    # print(f"Test set - Drug ID range: [{test_cf[:, 0].min()}, {test_cf[:, 0].max()}], Target ID range: [{test_cf[:, 1].min()}, {test_cf[:, 1].max()}]")
    # print(f"Valid set - Drug ID range: [{valid_cf[:, 0].min()}, {valid_cf[:, 0].max()}], Target ID range: [{valid_cf[:, 1].min()}, {valid_cf[:, 1].max()}]")
    
    # # 检查不重复的ID数量
    # print("\nNumber of unique IDs:")
    # print(f"Train set - Unique drugs: {len(set(train_cf[:, 0]))}, Unique targets: {len(set(train_cf[:, 1]))}")
    # print(f"Test set - Unique drugs: {len(set(test_cf[:, 0]))}, Unique targets: {len(set(test_cf[:, 1]))}")
    # print(f"Valid set - Unique drugs: {len(set(valid_cf[:, 0]))}, Unique targets: {len(set(valid_cf[:, 1]))}")
    
    # 这是打印：正确的映射，唯一ID，对应txt文件中的ID
    # Checking ID ranges in datasets:
    # Train set - Drug ID range: [0, 16579], Target ID range: [0, 4734]
    # Test set - Drug ID range: [23, 13746], Target ID range: [8, 4688]
    # Valid set - Drug ID range: [1, 14945], Target ID range: [0, 4725]
    # Number of unique IDs:
    # Train set - Unique drugs: 6673, Unique targets: 3890
    # Test set - Unique drugs: 260, Unique targets: 845
    # Valid set - Unique drugs: 2437, Unique targets: 1774
    
    # train_cf与txt中的ID范围一致，在txt中，drug的ID范围是0-16580，target的ID范围是0-4735！！
    # print(f"train_cf: {train_cf}") 
    # train_cf: [[  302   959]
    # [ 4786  2719]
    # [ 2959   303]
    # ...
    # [ 6754  3641]
    # [  236   875]
    # [12499  1669]]
    
    
    # 加载预训练嵌入，注意load_pretrained_embeddings里面的映射只是按照pt文件中的ID来，也就是与txt中一致, 所以并没有新的映射，drug_id_to_idx中是：drug_id_to_idx: {302: 302, 4786: 4786, 2959: 2959, 10002: 10002, .....}
    pretrain_drug_emb, pretrain_target_emb = None, None
    drug_id_to_idx, target_id_to_idx = {}, {}
    if args.use_pretrain:
        print('\nLoading pretrained embeddings...')
        pretrain_drug_emb, pretrain_target_emb, drug_id_to_idx, target_id_to_idx = \
            load_pretrained_embeddings(args.pretrain_path)
        if pretrain_drug_emb is not None:
            print(f"Loaded pretrained embeddings with shapes: {pretrain_drug_emb.shape}, {pretrain_target_emb.shape}")
    # 这里是映射表
    # print(f"drug_id_to_idx: {drug_id_to_idx}")  
    # drug_id_to_idx: {302: 302, 4786: 4786, 2959: 2959, 10002: 10002, .....}
    
    
    # (train_cf[:, 0].max() + 1) is 16580
    # len(drug_id_to_idx) is 6673
    # print("--------------------------------")
    # print(train_cf[:, 0].max() + 1)
    # print(len(drug_id_to_idx))
    # 统计用户和物品数量
    n_users = max(train_cf[:, 0].max() + 1, len(drug_id_to_idx))  # 应该是16580，保持完整的drug ID空间，也与txt一致
    n_items = max(train_cf[:, 1].max() + 1, len(target_id_to_idx))  # 应该是4735，保持完整的target ID空间，也与txt一致
    print(f"n_users: {n_users}")
    print(f"n_items: {n_items}")
    # n_users: 16580
    # n_items: 4735

    # 构建用户-物品集
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

