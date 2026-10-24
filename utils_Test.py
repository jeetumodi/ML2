import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from evaluation import compute_roc, compute_aupr, compute_mcc, acc_score,micro_score,compute_performance
import os
import time
from torch.nn.parameter import Parameter
import pickle

from torch_geometric.utils import dense_to_sparse

THREADHOLD = 0.4

# These values are observed in the training sets in our paper
Max_pssm = np.array([8, 9, 9, 9, 12, 9, 8, 8, 12, 9, 7, 9, 11, 10, 9, 8, 8, 13, 10, 8])
Min_pssm = np.array([-11,-12,-13,-13,-12,-13,-13,-12,-13,-13,-13,-13,-12,-12,-13,-12,-12,-13,-13,-12])
Max_hhm = np.array([10655,12141,12162,11354,11802,11835,11457,11686,11806,11262,11571,11979,12234,11884,11732,11508,11207,11388,12201,11743])
Min_hhm = np.zeros(20)


def getID(file_path):
    IDs=[]
    labels=[]
    sequences=[]

    with open(file_path, 'r') as file:
        lines = file.readlines()

    for i in range(0, len(lines), 3):
        protein_id = lines[i].strip()[1:]
        sequence = lines[i + 1].strip()
        label = int(lines[i + 2].strip())
        IDs.append(protein_id)
        labels.append(label)
        sequences.append(sequence)
    return IDs,labels,sequences


def protein_to_one_hot(protein_sequence):
    # 定义蛋白质序列的字母表
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'

    # 创建空的one-hot编码矩阵
    sequence_length = len(protein_sequence)
    alphabet_size = len(amino_acids)
    one_hot_matrix = np.zeros((sequence_length, alphabet_size))

    # 对每个氨基酸进行编码
    for i, amino_acid in enumerate(protein_sequence):
        if amino_acid in amino_acids:
            # 将当前氨基酸的位置索引设为1
            index = amino_acids.index(amino_acid)
            one_hot_matrix[i, index] = 1

    return one_hot_matrix



class ProDataset(Dataset):
    def __init__(self,  data_path):
        self.IDs,self.labels,self.seqs = getID(data_path)
        self.data_path = data_path


    def __getitem__(self, index):
        protein_id = self.IDs[index]
        label = self.labels[index]
        seq = self.seqs[index]




        seq = protein_to_one_hot(seq)       #   # seq之one-hot
        # 从特征文件加载节点特征和边距离
        dssp = np.load(f'Feature/dssp/{protein_id}.npy')    #   torch.from_numpy(
        pssm = np.load(f'Feature/pssm/{protein_id}.npy')
        hmm = np.load(f'Feature/hmm/{protein_id}.npy')
        # gip_sim=np.load(f'Feature/edge_similarity/{protein_id}.npy')

        resAF = np.load(f'Feature/resAF/{protein_id}.npy')
        # resAF1 = np.load(f'../Feature/Bio_fea/af_7/{protein_id}.npy')




        if protein_id == '2j3rA':
            resAF = resAF[:len(hmm), :]    # 只有这一个 2j3rA.npy 是多了一维
        elif resAF.shape[0] != len(seq):
            if resAF.shape[0] > len(seq):
                resAF = resAF[:len(seq), :]
            else:
                pad = np.zeros((len(seq) - resAF.shape[0], resAF.shape[1]), dtype=resAF.dtype)
                resAF = np.concatenate([resAF, pad], axis=0)

        # bert embedding
        if os.path.exists(f'Feature/bert_esm2_t36/{protein_id}.npy'):
            bert = np.load(f'Feature/bert_esm2_t36/{protein_id}.npy')
        elif os.path.exists(f'../Feature/Bert/bert_esm2_t36/{protein_id}.npy'):
            bert = np.load(f'../Feature/Bert/bert_esm2_t36/{protein_id}.npy')
        elif os.path.exists(f'EGCPPIS_embedding/bert_esm2_t36/{protein_id}.npy'):
            bert = np.load(f'EGCPPIS_embedding/bert_esm2_t36/{protein_id}.npy')
        else:
            bert = np.zeros((len(seq), 33), dtype=np.float32)

        if bert.shape[0] > len(seq):
            bert = bert[:len(seq), :]
        elif bert.shape[0] < len(seq):
            pad = np.zeros((len(seq) - bert.shape[0], bert.shape[1]), dtype=bert.dtype)
            bert = np.concatenate([bert, pad], axis=0)

        if os.path.exists(f'Feature/distance_map_SC/{protein_id}.npy'):
            distance = np.load(f'Feature/distance_map_SC/{protein_id}.npy')
        elif os.path.exists(f'Feature/distance_map/{protein_id}.npy'):
            distance = np.load(f'Feature/distance_map/{protein_id}.npy')
        else:
            distance = np.eye(len(seq), dtype=np.float32)

        distance_map = process_distance_map(distance)

        label = [int(digit) for digit in str(label)]
        if len(label) < len(seq):
            label = [0] * (len(seq) - len(label)) + label
        elif len(label) > len(seq):
            label = label[:len(seq)]

        # 加载伪位置 3维
        if not hasattr(ProDataset, '_psepos_cache'):
            ProDataset._psepos_cache = {}
            for pse_file in [
                "Feature/psepos/Train335_Test60_psepos_SC.pkl",
                "Feature/psepos/Test315-28_psepos_SC.pkl",
                "Feature/psepos/UBtest31-6_psepos_SC.pkl",
                "Feature/psepos/Train335_psepos_SC.pkl",
                "Feature/psepos/Test60_psepos_SC.pkl",
            ]:
                if os.path.exists(pse_file):
                    try:
                        with open(pse_file, 'rb') as f:
                            ProDataset._psepos_cache.update(pickle.load(f))
                    except Exception:
                        pass

        if protein_id == "2j3rA":
            pos = np.zeros((158, 3))
        elif protein_id in ProDataset._psepos_cache:
            pos = ProDataset._psepos_cache[protein_id]
        else:
            pos = np.zeros((len(seq), 3))

        psepos = torch.from_numpy(pos)
        psepos_1 = torch.sqrt(torch.sum(psepos * psepos, dim=1)).unsqueeze(-1) / 15     # 3维变1维

        n_fea = np.concatenate([seq, dssp, pssm, hmm, resAF, psepos_1], axis=1).astype(np.float32)


        ##      edge_attr   (边数，4)
        # edge_attr=np.load(f'Feature/edge_attr/{protein_id}.npy')
        #####         （节点数，节点数，4）
        # edge_attr=np.load(f'Feature/edge_addr_4/{protein_id}.npy')

        #       坐标      边索引
        # X_ca, seq, name, node_s, node_v, edge_s, edge_v, edge_index, mask, data_MFE = load_protein_from_pkl("MFE_model/MFE_feature/" + protein_id + ".pkl")


        # 原子特征和原子-残基映射关系
        # Atom_path="Tenx/HSSPPI/Atom_model/" + protein_id + ".pkl"
        
        # for other Datasets
        # Atom_path="Tenx/HSSPPI/Atom_model_Test287/" + protein_id + ".pkl"
        # 原子特征和原子-残基映射关系
        Atom_path = "Tenx/HSSPPI/Atom_model_TestUB25/" + protein_id + ".pkl"

        if os.path.exists(Atom_path):
            with open(Atom_path, 'rb') as f:
                atom_data = pickle.load(f)
                atom_graph_node = np.array(atom_data['atom_graph_node'])
                atom_graph_edge = np.array(atom_data['atom_graph_edge'])
                atom_graph_edge = torch.LongTensor(atom_graph_edge)
                a2r_map = np.array(atom_data['a2r_map'])
        else:
            atom_graph_node = np.zeros((len(seq), 37), dtype=np.float32)
            atom_graph_node[:, 0] = 1.0
            atom_graph_edge = torch.stack([torch.arange(len(seq)), torch.arange(len(seq))], dim=0).to(torch.int64)
            a2r_map = np.arange(len(seq), dtype=np.int64)

        return seq, dssp, atom_graph_node, atom_graph_edge, a2r_map, bert, psepos, distance_map, distance, n_fea, np.array(label), seq, seq



        # save_path = "../Feature/PDB/atom_data/" + protein_id + ".pkl"
        # # 原子特征和原子-残基映射关系
        # with open(save_path, 'rb') as f:
        #     atom_data = pickle.load(f)
        #     res2atom = np.array(atom_data['res2atom'])
        #     atom_onehot_embedding = np.array(atom_data['atom_onehot_embedding'])
        #     coords = np.array(atom_data['coords'])
        #     distance_matrix = np.array(atom_data['distance_matrix'])
        #
        # distance_map_atom=process_distance_map(distance_matrix,cutoff=2)
        # edge_index_atom, _ = dense_to_sparse(torch.LongTensor(distance_map_atom))
        # edge_index_atom = edge_index_atom.to(torch.int64)

        # return seq,dssp,    atom_onehot_embedding,edge_index_atom,res2atom,    bert,psepos,distance_map,distance,   n_fea, np.array(label),coords, seq


    def __len__(self):
        return len(self.IDs)














def load_protein_from_pkl(file_path):
    """
    从 pkl 文件中读取蛋白质的特征。

    :param file_path: .pkl 文件的路径。
    :return: 含蛋白质特征的字典
    """
    with open(file_path, 'rb') as f:
        data = pickle.load(f)

    # 解包读取的数据
    X_ca = data['x']
    seq = data['seq']
    name = data['name']
    node_s = data['node_s']
    node_v = data['node_v']
    edge_s = data['edge_s']
    edge_v = data['edge_v']
    edge_index = data['edge_index']
    mask = data['mask']
    return X_ca, seq, name, node_s, node_v, edge_s, edge_v, edge_index, mask,data




def process_distance_map(adjacency_matrix, cutoff=16):
    length = adjacency_matrix.shape[0]
    distance_map = np.zeros((length, length))

    for i in range(length):
        for j in range(length):
            if i == j:
                distance_map[i, j] = 1   #   egnn
            elif adjacency_matrix[i, j] < cutoff:
                distance_map[i, j] = 1   #   egnn
            else:
                distance_map[i, j] = 0

    return distance_map






