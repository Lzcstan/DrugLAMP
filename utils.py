import os
import dgl
import math
import random
import torch
import logging
import numpy as np
import pandas as pd
import networkx as nx
import torch.nn as nn
import torch.nn.functional as F

from rdkit import Chem
from einops import rearrange
from contextlib import contextmanager
from functools import partial, wraps, reduce
from torch_geometric.utils import from_smiles
from torch.utils.checkpoint import checkpoint

REPO_PATH = '~/projects/DrugLAMP/'

def partition_data(data_splits, data, kind='drug'):
    """
    data_splits : data_splits should sum to 1
    data        :
    kind        : "pair" (splits on pairs, DeepDTA-style) or
                  "drugs" (splits on the unique drugs)

    Assume that drugs are novel and searched for while proteins are known
    partition data on the drugs so that drugs in train are not in valid or
    test, and drugs in valid are not in test.
    """
    assert np.sum(data_splits) == 1., 'data_splits should sum to 1'

    drugs = list(data['Drug_ID'].unique())
    n_drug = len(drugs)

    if kind == 'drug':
        n_train = int(round(n_drug * data_splits[0]))
        n_valid = int(round(n_drug * data_splits[1]))
        train = {'drugs': random.sample(drugs, n_train)}
        not_train_drugs = list(np.setdiff1d(drugs, train['drugs']))
        valid = {'drugs': random.sample(not_train_drugs, n_valid)}
        test = {'drugs': list(np.setdiff1d(not_train_drugs, valid['drugs']))}

        train['ids'] = []
        for drug in train['drugs']:
            train['ids'] += list(data.index[data['Drug_ID'] == drug])

        valid['ids'] = []
        for drug in valid['drugs']:
            valid['ids'] += list(data.index[data['Drug_ID'] == drug])

        test['ids'] = []
        for drug in test['drugs']:
            test['ids'] += list(data.index[data['Drug_ID'] == drug])

    elif kind == 'pair':
        n = len(data)
        n_train = int(round(n * data_splits[0]))
        n_valid = int(round(n * data_splits[1]))
        # n_test = int(round(n * data_splits[2]))
        ids = np.arange(n, dtype=int)
        random.shuffle(ids)
        train = {'ids': ids[:n_train]}
        train['drugs'] = data.loc[train['ids'], 'Drug_ID'].unique()
        valid = {'ids': ids[n_train:n_train+n_valid]}
        valid['drugs'] = data.loc[valid['ids'], 'Drug_ID'].unique()
        test = {'ids': ids[n_train+n_valid:]}
        test['drugs'] = data.loc[test['ids'], 'Drug_ID'].unique()

    return train, valid, test, n_drug

def smi2graph(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    c_size = mol.GetNumAtoms()

    features = []
    for atom in mol.GetAtoms():
        feature = atom_features(atom)
        features.append(feature / sum(feature))

    edges = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
    g = nx.Graph(edges).to_directed()
    edge_index = []
    mol_adj = np.zeros((c_size, c_size))
    for e1, e2 in g.edges:
        mol_adj[e1, e2] = 1
        # edge_index.append([e1, e2])
    mol_adj += np.matrix(np.eye(mol_adj.shape[0]))
    index_row, index_col = np.where(mol_adj >= 0.5)
    for i, j in zip(index_row, index_col):
        edge_index.append([i, j])
    # print('smile_to_graph')
    # print(np.array(features).shape)
    return c_size, features, edge_index

def prot2graph(order, seq, 
               contact_dir='', 
               aln_dir=''):
    edge_index = []
    size = len(seq)
    # contact_dir = 'data/' + dataset + '/pconsc4'
    contact_file = os.path.join(contact_dir, order + '.npy')
    contact_map = np.load(contact_file)
    contact_map += np.matrix(np.eye(contact_map.shape[0]))
    index_row, index_col = np.where(contact_map >= 0.5)
    for i, j in zip(index_row, index_col):
        edge_index.append([i, j])
    feature = prot2feature(order, seq, aln_dir)
    edge_index = np.array(edge_index)
    return size, feature, edge_index

def get_node_edges(smiles_edges, index_map):
    """
    """
    node_edges = [[], []]
    for edge in smiles_edges.T:

        id_0 = np.logical_and(index_map['smiles_i0'] <= edge[0],
                              index_map['smiles_i1'] >= edge[0])
        id_1 = np.logical_and(index_map['smiles_i0'] <= edge[1],
                              index_map['smiles_i1'] >= edge[1])
        if id_0.sum() == 1 and id_1.sum() == 1:
            node_edges[0].append(int(index_map[id_0]['token_i']))
            node_edges[1].append(int(index_map[id_1]['token_i']))
        elif id_0.sum() > 1 or id_1.sum() > 1:
            raise ValueError('The edge seems to connect to multiple nodes!')

    return np.array(node_edges, dtype=int)

def smiles_edges_to_token_edges(smiles, tokenizer, reverse_vocab):
    """
    """
    token_ids = tokenizer.encode(smiles)
    index_map = get_indexmap(token_ids, reverse_vocab, smiles)
    smiles_edges = from_smiles(smiles).edge_index
    node_edges = get_node_edges(smiles_edges, index_map)
    # keep only between node edges
    node_edges = node_edges[:, ((node_edges[0] - node_edges[1]) != 0)]
    # remove duplicates. Duplicates can occur when different atoms within the
    # same nodes are connected to each other.
    node_edges = np.unique(node_edges, axis=1)

    return node_edges, index_map

def get_indexmap(token_ids, rev_vocab, smiles):

    index_map = pd.DataFrame(index=range(len(token_ids)),
                             columns=['token_i',
                                      'token',
                                      'token_id',
                                      'keep',
                                      'smiles_i0',
                                      'smiles_i1'])
    start = 0
    token_i = 0
    for i, token_id in enumerate(token_ids):

        token = rev_vocab[token_id]

        if token.isalpha():  # only all alphabetic chars are nodes
            smiles_i0 = smiles[start:].find(token)
            if smiles_i0 >= 0:
                smiles_i0 += start
                smiles_i1 = smiles_i0 + len(token)
                start = smiles_i1

                index_map.loc[i] = (token_i, token, token_id,
                                    True, smiles_i0, smiles_i1 - 1)
                token_i += 1
            else:
                raise ValueError('Node token not found in SMILES.\nCheck that '
                                 'token_ids are computed from smiles.')
        else:
            index_map.loc[i] = (-1, token, token_id, False, -1, -1)

    return index_map

# drug embedding
def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        # print(x)
        raise Exception('input {0} not in allowable set{1}:'.format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))

def one_of_k_encoding_unk(x, allowable_set):
    '''Maps inputs not in the allowable set to the last element.'''
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

def atom_features(atom):
    # 44 + 11 + 11 + 11 + 1 + 3 + 1
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),
                                          ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As',
                                           'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se',
                                           'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr',
                                           'Pt', 'Hg', 'Pb', 'X']) +
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    [atom.GetIsAromatic()]
                    +
                    one_of_k_encoding_unk(atom.GetFormalCharge() , [-1,0,1]) +
                    [atom.IsInRing()]
)

# prot embedding
def prot2feature(order, seq, aln_dir):
    # aln_dir = 'data/' + dataset + '/aln'
    aln_file = os.path.join(aln_dir, order + '.aln')
    feature = prot_feature(aln_file, seq)
    return feature

def prot_feature(aln_file, seq):
    pssm = PSSM_calculation(aln_file, seq)
    other_feature = seq_feature(seq)

    return np.concatenate((np.transpose(pssm, (1, 0)), other_feature), axis=1)

prot_res_table = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y',
                 'X']

def seq_feature(seq):
    pro_hot = np.zeros((len(seq), len(prot_res_table)))
    pro_property = np.zeros((len(seq), 12))
    for i in range(len(seq)):
        pro_hot[i,] = one_of_k_encoding(seq[i], prot_res_table)
        pro_property[i,] = residue_features(seq[i])
    return np.concatenate((pro_hot, pro_property), axis=1)

def PSSM_calculation(aln_file, seq):
    pfm_mat = np.zeros((len(prot_res_table), len(seq)))
    with open(aln_file, 'r') as f:
        line_count = len(f.readlines())
        for line in f.readlines():
            if len(line) != len(seq):
                print('error', len(line), len(seq))
                continue
            count = 0
            for res in line:
                if res not in prot_res_table:
                    count += 1
                    continue
                pfm_mat[prot_res_table.index(res), count] += 1
                count += 1
    pseudocount = 0.8
    ppm_mat = (pfm_mat + pseudocount / 4) / (float(line_count) + pseudocount)
    pssm_mat = ppm_mat
    return pssm_mat

def residue_features(residue):
    res_property1 = [1 if residue in prot_res_aliphatic_table else 0, 1 if residue in prot_res_aromatic_table else 0,
                     1 if residue in prot_res_polar_neutral_table else 0,
                     1 if residue in prot_res_acidic_charged_table else 0,
                     1 if residue in prot_res_basic_charged_table else 0]
    res_property2 = [res_weight_table[residue], res_pka_table[residue], res_pkb_table[residue], res_pkx_table[residue],
                     res_pl_table[residue], res_hydrophobic_ph2_table[residue], res_hydrophobic_ph7_table[residue]]
    return np.array(res_property1 + res_property2)

prot_res_aliphatic_table = ['A', 'I', 'L', 'M', 'V']
prot_res_aromatic_table = ['F', 'W', 'Y']
prot_res_polar_neutral_table = ['C', 'N', 'Q', 'S', 'T']
prot_res_acidic_charged_table = ['D', 'E']
prot_res_basic_charged_table = ['H', 'K', 'R']

res_weight_table = {'A': 71.08, 'C': 103.15, 'D': 115.09, 'E': 129.12, 'F': 147.18, 'G': 57.05, 'H': 137.14,
                    'I': 113.16, 'K': 128.18, 'L': 113.16, 'M': 131.20, 'N': 114.11, 'P': 97.12, 'Q': 128.13,
                    'R': 156.19, 'S': 87.08, 'T': 101.11, 'V': 99.13, 'W': 186.22, 'Y': 163.18}
res_pka_table = {'A': 2.34, 'C': 1.96, 'D': 1.88, 'E': 2.19, 'F': 1.83, 'G': 2.34, 'H': 1.82, 'I': 2.36,
                 'K': 2.18, 'L': 2.36, 'M': 2.28, 'N': 2.02, 'P': 1.99, 'Q': 2.17, 'R': 2.17, 'S': 2.21,
                 'T': 2.09, 'V': 2.32, 'W': 2.83, 'Y': 2.32}
res_pkb_table = {'A': 9.69, 'C': 10.28, 'D': 9.60, 'E': 9.67, 'F': 9.13, 'G': 9.60, 'H': 9.17,
                 'I': 9.60, 'K': 8.95, 'L': 9.60, 'M': 9.21, 'N': 8.80, 'P': 10.60, 'Q': 9.13,
                 'R': 9.04, 'S': 9.15, 'T': 9.10, 'V': 9.62, 'W': 9.39, 'Y': 9.62}
res_pkx_table = {'A': 0.00, 'C': 8.18, 'D': 3.65, 'E': 4.25, 'F': 0.00, 'G': 0, 'H': 6.00,
                 'I': 0.00, 'K': 10.53, 'L': 0.00, 'M': 0.00, 'N': 0.00, 'P': 0.00, 'Q': 0.00,
                 'R': 12.48, 'S': 0.00, 'T': 0.00, 'V': 0.00, 'W': 0.00, 'Y': 0.00}
res_pl_table = {'A': 6.00, 'C': 5.07, 'D': 2.77, 'E': 3.22, 'F': 5.48, 'G': 5.97, 'H': 7.59,
                'I': 6.02, 'K': 9.74, 'L': 5.98, 'M': 5.74, 'N': 5.41, 'P': 6.30, 'Q': 5.65,
                'R': 10.76, 'S': 5.68, 'T': 5.60, 'V': 5.96, 'W': 5.89, 'Y': 5.96}
res_hydrophobic_ph2_table = {'A': 47, 'C': 52, 'D': -18, 'E': 8, 'F': 92, 'G': 0, 'H': -42, 'I': 100,
                             'K': -37, 'L': 100, 'M': 74, 'N': -41, 'P': -46, 'Q': -18, 'R': -26, 'S': -7,
                             'T': 13, 'V': 79, 'W': 84, 'Y': 49}
res_hydrophobic_ph7_table = {'A': 41, 'C': 49, 'D': -55, 'E': -31, 'F': 100, 'G': 0, 'H': 8, 'I': 99,
                             'K': -23, 'L': 97, 'M': 74, 'N': -28, 'P': -46, 'Q': -10, 'R': -14, 'S': -5,
                             'T': 13, 'V': 76, 'W': 97, 'Y': 63}

def set_seed(seed=1000):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def tail_pad(x, maxsize):
    #x should be list   [torch(N,features)] *batch
    b = len(x)
    features = x[0].shape[-1]
    out = torch.zeros(b, maxsize, features)
    for i in range(b):
        a = x[i]
        out[i,:a.shape[-2],:] = a
    return out.to(a.device)

def repeat_pad(x, maxsize):
    b = len(x)
    features = x[0].shape[-1]
    out = torch.zeros(b, maxsize, features)
    for i in range(b):
        a = x[i]
        quot = maxsize // a.shape[-2]
        for j in range(quot):
            st = j * a.shape[-2]
            out[i, st: st + a.shape[-2],:] = a
    return out.to(a.device)

def multimodality_collate_func(x):
    d, p, y, llm, meta = zip(*x)
    d = dgl.batch(d)
    # d_llm = Batch.from_data_list([l['drug'] for l in llm])
    # p_llm = Batch.from_data_list([l['prot'] for l in llm])
    d_llm = tail_pad([l['drug'].x for l in llm], 512)
    # p_llm = tail_pad([l['prot'].x for l in llm], 9 * 256)
    p_llm = repeat_pad([l['prot'].x for l in llm], 9 * 256)
    return d, torch.tensor(np.array(p)), torch.tensor(y), d_llm, p_llm, meta

def mkdir(path):
    path = path.strip()
    path = path.rstrip("\\")
    is_exists = os.path.exists(path)
    if not is_exists:
        os.makedirs(path, exist_ok=True)


CHARPROTSET = {
    "A": 1,
    "C": 2,
    "B": 3,
    "E": 4,
    "D": 5,
    "G": 6,
    "F": 7,
    "I": 8,
    "H": 9,
    "K": 10,
    "M": 11,
    "L": 12,
    "O": 13,
    "N": 14,
    "Q": 15,
    "P": 16,
    "S": 17,
    "R": 18,
    "U": 19,
    "T": 20,
    "W": 21,
    "V": 22,
    "Y": 23,
    "X": 24,
    "Z": 25,
}

def integer_label_protein(sequence, seq_end, max_length=9 * 256):
    """
    Integer encoding for protein string sequence.
    Args:
        sequence (str): Protein string sequence.
        max_length: Maximum encoding length of input protein string.
    """
    encoding = np.zeros(max_length)
    seq = sequence[: seq_end] # Tid: to match the LLM
    for idx, letter in enumerate(seq):
        try:
            letter = letter.upper()
            encoding[idx + 1] = CHARPROTSET[letter] # Tid: Used to as CLS
        except KeyError:
            logging.warning(
                f"character {letter} does not exists in sequence category encoding, skip and treat as " f"padding."
            )
    return encoding

def repeat_integer_label_protein(sequence, seq_end, max_length=9 * 256):
    """
    Integer encoding repeatly for protein string sequence.
    Args:
        sequence (str): Protein string sequence.
        max_length: Maximum encoding length of input protein string.
    """
    encoding = np.zeros(max_length)
    seq = sequence[: seq_end] # Tid: to match the LLM
    quot = max_length // (len(seq) + 2) # Tid: add CLS and SEP
    for i in range(quot):
        st = i * (len(seq) + 2) + 1
        for idx, letter in enumerate(seq):
            try:
                letter = letter.upper()
                encoding[idx + st] = CHARPROTSET[letter] # Tid: Used to as CLS
            except KeyError:
                logging.warning(
                    f"character {letter} does not exists in sequence category encoding, skip and treat as " f"padding."
                )
    return encoding # v_p

# helper functions
def identity(t, *args, **kwargs):
    return t

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

@contextmanager
def null_context():
    yield

def max_neg_value(dtype):
    return -torch.finfo(dtype).max

def cast_tuple(t):
    return t if isinstance(t, (tuple, list)) else (t,)

def masked_mean(t, mask, dim = 1, eps = 1e-6):
    t = t.masked_fill(~mask, 0.)
    numer = t.sum(dim = dim)
    denom = mask.sum(dim = dim).clamp(min = eps)
    return numer / denom

def log(t, eps = 1e-20):
    return torch.log(t + eps)

def l2norm(t):
    return F.normalize(t, dim = -1)

def matrix_diag(t):
    device = t.device
    i, j = t.shape[-2:]
    num_diag_el = min(i, j)
    i_range = torch.arange(i, device = device)
    j_range = torch.arange(j, device = device)
    diag_mask = rearrange(i_range, 'i -> i 1') == rearrange(j_range, 'j -> 1 j')
    diag_el = t.masked_select(diag_mask)
    return rearrange(diag_el, '(b d) -> b d', d = num_diag_el)

# checkpointing helper function
def make_checkpointable(fn):
    @wraps(fn)
    def inner(*args):
        input_needs_grad = any([isinstance(el, torch.Tensor) and el.requires_grad for el in args])

        if not input_needs_grad:
            return fn(*args)

        return checkpoint(fn, *args)

    return inner

# keyword argument helpers
def pick_and_pop(keys, d):
    values = list(map(lambda key: d.pop(key), keys))
    return dict(zip(keys, values))

def group_dict_by_key(cond, d):
    return_val = [dict(),dict()]
    for key in d.keys():
        match = bool(cond(key))
        ind = int(not match)
        return_val[ind][key] = d[key]
    return (*return_val,)

def string_begins_with(prefix, str):
    return str.startswith(prefix)

def group_by_key_prefix(prefix, d):
    return group_dict_by_key(partial(string_begins_with, prefix), d)

def groupby_prefix_and_trim(prefix, d):
    kwargs_with_prefix, kwargs = group_dict_by_key(partial(string_begins_with, prefix), d)
    kwargs_without_prefix = dict(map(lambda x: (x[0][len(prefix):], x[1]), tuple(kwargs_with_prefix.items())))
    return kwargs_without_prefix, kwargs

def find_in_train_set(
        x: str,
        dataset: str,
        split: str,
        label: str
):
    if not label in ['prot', 'drug']:
        raise NotImplementedError
    if label == 'prot':
        col = 'Protein'
    else:
        col = 'SMILES'
        x_mol = Chem.MolFromSmiles(x)
    file_dir = os.path.join(f'{REPO_PATH}/datasets', dataset, split)
    if not os.path.isdir(file_dir):
        raise FileExistsError
    file_paths = [fn for fn in os.listdir(file_dir) if fn.endswith('train.csv')]
    file_dfs = []
    for fn_path in file_paths:
        df = pd.read_csv(os.path.join(file_dir, fn_path))
        file_dfs.append(df[[col]])
    df = pd.concat(file_dfs)

    cnt = 0
    for idx, row in df.iterrows():
        cnt += 1
        if label == 'prot':
            if row[col] == x:
                print(f"{dataset}'s {split}-split has this protein")
                return True, cnt, idx
        else:
            smi = row[col]
            mol = Chem.RemoveHs(Chem.MolFromSmiles(smi), sanitize=False)
            if mol.HasSubstructMatch(x_mol) and x_mol.HasSubstructMatch(mol):
                print(f"{dataset}'s {split}-split has this drug")
                return True, cnt, idx
    return False, -1, -1

# SSL helper function
def mask_with_tokens(t, token_ids):
    init_no_mask = torch.full_like(t, False, dtype=torch.bool)
    mask = reduce(lambda acc, el: acc | (t == el), token_ids, init_no_mask)
    return mask

def get_mask_subset_with_prob(mask, prob):
    batch, seq_len, device = *mask.shape, mask.device
    max_masked = math.ceil(prob * seq_len)

    num_tokens = mask.sum(dim=-1, keepdim=True)
    mask_excess = (mask.cumsum(dim=-1) > (num_tokens * prob).ceil())
    mask_excess = mask_excess[:, :max_masked]

    rand = torch.rand((batch, seq_len), device=device).masked_fill(~mask, -1e9)
    _, sampled_indices = rand.topk(max_masked, dim=-1)
    sampled_indices = (sampled_indices + 1).masked_fill_(mask_excess, 0)

    new_mask = torch.zeros((batch, seq_len + 1), device=device)
    new_mask.scatter_(-1, sampled_indices, 1)
    return new_mask[:, 1:].bool()

def prob_mask_like(t, prob):
    return torch.zeros_like(t).float().uniform_(0, 1) < prob

def flatten(t):
    return t.reshape(t.shape[0], -1)

def tanh_decay(m_ori, n_re, step):
    return m_ori * (1 - np.tanh(2 * (1 - step / n_re)))

def cosine_anneal(m_ori, n_re, step):
    return m_ori * (1 + np.cos(np.pi * (1 - step / n_re))) / 2

def max_cosine_tanh_decay(m_ori, n_re, step):
    return max(m_ori * (1 + np.cos(np.pi * (1 - step / n_re))) / 2, m_ori * (1 - np.tanh(2 * (1 - step / n_re))))

def no_decay(m_ori, n_re, step):
    return m_ori

def sigmoid_cosine_distance_p(x, y, p=1):
    sigmoid = nn.Sigmoid()
    cos_sim = nn.CosineSimilarity()
    return (1 - sigmoid(cos_sim(x, y))) ** p

def singleton(cache_key):
    def inner_fn(fn):
        @wraps(fn)
        def wrapper(self, *args, **kwargs):
            instance = getattr(self, cache_key)
            if instance is not None:
                return instance

            instance = fn(self, *args, **kwargs)
            setattr(self, cache_key, instance)
            return instance
        return wrapper
    return inner_fn