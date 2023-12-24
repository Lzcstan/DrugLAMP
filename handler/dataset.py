import random
import torch
import numpy as np
import pandas as pd
import os.path as osp

from tqdm import tqdm
from functools import partial
from transformers import AutoModel, AutoTokenizer
from torch_geometric.data import Dataset as GraphDataset, Data
from dgllife.utils import smiles_to_bigraph, CanonicalAtomFeaturizer, CanonicalBondFeaturizer
from utils import smiles_edges_to_token_edges, repeat_integer_label_protein

def edges_from_protein_sequence(prot_seq):
    """
    Since we only have the primary protein sequence we only know of the peptide
    bonds between amino acids. I.e. only amino acids linked in the primary
    sequence will have edges between them.
    """
    n = len(prot_seq)
    # first and row in COO format
    # each node is connected to left and right except the first an last.
    edge_index = np.stack([np.repeat(np.arange(n), 2)[1:-1],
                           np.repeat(np.arange(n), 2)[1:-1]], axis=0)
    for i in range(0, n, 2):
        edge_index[1, i], edge_index[1, i+1] = \
            edge_index[1, i+1], edge_index[1, i]

    return torch.tensor(edge_index, dtype=torch.long)

class MultiModalityDataset(GraphDataset):
    def __init__(self, root, df_name, esp_fn, prot_n_layer, device,
                 cutoff: int = None,
                 drug_encoder='DeepChem/ChemBERTa-77M-MTR',
                 max_drug_atoms=512,
                 max_prot_resis=1022,
                 gen_embed=False):
        self.max_drug_atoms = max_drug_atoms
        self.max_prot_resis = max_prot_resis
        self.raw_file_name = df_name
        self.drug_encoder_name = drug_encoder
        self.gen_embed = gen_embed
        self.prot_n_layer = prot_n_layer
        self.device = device

        self.atom_featurizer = CanonicalAtomFeaturizer()
        self.bond_featurizer = CanonicalBondFeaturizer(self_loop=True)
        self.drug_graph_converter = partial(smiles_to_bigraph, add_self_loop=True)

        self.root = root

        if gen_embed:
            try:
                self.drug_encoder = AutoModel.from_pretrained(drug_encoder)
                self.drug_tokenizer = AutoTokenizer.from_pretrained(drug_encoder)
            except:
                import sys
                raise ConnectionError('ChemBERTa connection error')
                sys.exit(1)
            self.prot_encoder, prot_alphabet = esp_fn()
            self.prot_batch_converter = prot_alphabet.get_batch_converter()

            self._save_llm_params(self.drug_encoder.config.hidden_size, self.prot_encoder.embed_dim)
        
        self.n_drug_feature, self.n_prot_feature = self._load_llm_params()

        super(MultiModalityDataset, self).__init__(root, None, None)

        self.cutoff = cutoff

        self.df = pd.read_csv(osp.join(root, df_name))
        self.ids = self.df.index.values

    @property
    def processed_dir(self) -> str:
        return self.root[: self.root.rfind('/')]
    
    @property
    def dataset_name(self) -> str:
        return self.processed_dir[self.processed_dir.rfind('/') + 1:]

    @property
    def raw_dir(self) -> str:
        return self.root
    
    @property
    def raw_file_names(self):
        return self.raw_file_name

    @property
    def processed_file_names(self):
        self.raw_data = pd.read_csv(osp.join(self.processed_dir, 'full.csv'))
        self.prots = self.raw_data['Protein'].unique()
        self.drugs = self.raw_data['SMILES'].unique()
        self.n_prot, self.n_drug = len(self.prots), len(self.drugs)
        self.n_total = self.n_prot + self.n_drug
        self.prot2ord = {seq: ord for seq, ord in zip(self.prots, np.arange(self.n_prot, dtype=int))}
        self.drug2ord = {smi: ord for smi, ord in zip(self.drugs, np.arange(self.n_drug, dtype=int))}
        prot_embed_fnames = [self._build_embed_fname(ord, 'prot') for ord in np.arange(self.n_prot, dtype=int)]
        drug_embed_fnames = [self._build_embed_fname(ord, 'drug') for ord in np.arange(self.n_drug, dtype=int)]

        return prot_embed_fnames + drug_embed_fnames
    
    def download(self):
        pass

    def _save_llm_params(self, n_drug_feature, n_prot_feature):
        txt_fpath = osp.join(self.processed_dir.replace(f'datasets/{self.dataset_name}', 'configs'), f"{self.prot_n_layer}_layers_params.txt")
        if not osp.exists(txt_fpath):
            np.savetxt(txt_fpath, np.array([[n_drug_feature, n_prot_feature]]), fmt='%d', delimiter='\t')

    def _load_llm_params(self):
        txt_fpath = osp.join(self.processed_dir.replace(f'datasets/{self.dataset_name}', 'configs'), f"{self.prot_n_layer}_layers_params.txt")
        f = open(txt_fpath)
        line = f.readline()
        f.close()
        return map(int, line.split('\t'))

    def _build_embed_fname(self, order, modality='drug'):
        if modality == 'prot':
            return f"{self.dataset_name}_{order}_prot_{self.n_prot_feature}_embedded.pt"
        return f"{self.dataset_name}_{order}_{modality}_embedded.pt"
    
    def process(self): # Tid: HACK

        vocab = self.drug_tokenizer.vocab
        reverse_vocab = {key: val for val, key in vocab.items()}

        if self.gen_embed:
            processed_prots, processed_drugs = [], []
            for idx, row in tqdm(self.raw_data.iterrows()):

                # # Only embed a protein if it hasn't already been embedded and saved
                prot_ord = self.prot2ord[row['Protein']]
                fname = self._build_embed_fname(prot_ord, 'prot')
                fpath = osp.join(self.processed_dir, fname)
                if not osp.exists(fpath):
                    prot_labels, prot_strs, tokens = self.prot_batch_converter([
                        (str(prot_ord), row['Protein'][: self.max_prot_resis])
                    ])
                    with torch.no_grad():
                        results = self.prot_encoder(tokens, repr_layers=[self.prot_n_layer], return_contacts=True)
                    embed = results["representations"][self.prot_n_layer]
                    edges = edges_from_protein_sequence(row['Protein'])
                    data = {'embeddings': Data(x=embed, edge_index=edges),
                            'Prot_ID': prot_ord}
                    torch.save(data, fpath)
                    processed_prots.append(prot_ord)

                # Only embed a drug if it hasn't already been embedded and saved
                drug_ord = self.drug2ord[row['SMILES']]
                fname = self._build_embed_fname(drug_ord, 'drug')
                fpath = osp.join(self.processed_dir, fname)
                if not osp.exists(fpath):
                    tokens = \
                        torch.tensor(self.drug_tokenizer.encode(row['SMILES'],
                                                                truncation=True,
                                                                max_length=self.max_drug_atoms))
                    embed = \
                        self.drug_encoder(tokens.reshape(1, -1)).last_hidden_state
                    edges, index_map = \
                        smiles_edges_to_token_edges(row['SMILES'],
                                                    self.drug_tokenizer,
                                                    reverse_vocab)
                    data = {'embeddings':
                            Data(x=embed,
                                edge_index=torch.tensor(edges, dtype=torch.long)),
                            'Drug_ID': drug_ord,
                            'node_ids': index_map['keep'].values.astype('bool')}
                    torch.save(data, fpath)
                    processed_drugs.append(drug_ord)

    def len(self):
        if self.cutoff is not None:
            return min(self.cutoff, len(self.ids))
        return len(self.ids)
    
    def get(self, idx):
        idx = self.ids[idx]
        row = self.df.iloc[idx]
        smi = row['SMILES']
        seq = row['Protein']
        y = row['Y']
        
        # drug llm
        drug_ord = self.drug2ord[smi]
        drug_embed_fname = osp.join(self.processed_dir,
                                    self._build_embed_fname(drug_ord, 'drug'))
        drug_data = torch.load(drug_embed_fname)

        # prot llm
        prot_ord = self.prot2ord[seq]
        prot_embed_fname = osp.join(self.processed_dir,
                                    self._build_embed_fname(prot_ord, 'prot'))
        prot_data = torch.load(prot_embed_fname)

        meta = {'Drug_ID': str(drug_data['Drug_ID']),
                'Prot_ID': str(prot_data['Prot_ID']),
                'raw_Drug_ID': str(drug_ord),
                'raw_Prot_ID': str(prot_ord),
                'Drug': smi[: self.max_drug_atoms],
                'Prot': seq[: self.max_prot_resis],
                'Y': row['Y']}

        drug_data['embeddings'].x.requires_grad = False
        prot_data['embeddings'].x.requires_grad = False

        llm = {'drug':drug_data['embeddings'].to(self.device),
               'prot':  prot_data['embeddings'].to(self.device),
               }

        # drug graph
        v_d = self.drug_graph_converter(smiles=smi, node_featurizer=self.atom_featurizer, edge_featurizer=self.bond_featurizer)
        actual_node_feats = v_d.ndata.pop('h')
        n_actual_node = actual_node_feats.shape[0]
        n_virtual_node = self.max_drug_atoms - n_actual_node
        virtual_node_bit = torch.zeros([n_actual_node, 1])
        actual_node_feats = torch.cat((actual_node_feats, virtual_node_bit), 1)
        v_d.ndata['h'] = actual_node_feats
        virtual_node_feats = torch.cat((torch.zeros(n_virtual_node, 74), torch.ones(n_virtual_node, 1)), 1)
        v_d.add_nodes(n_virtual_node, {"h": virtual_node_feats})
        v_d = v_d.add_self_loop()

        # prot token
        v_p = repeat_integer_label_protein(seq, self.max_prot_resis)

        return v_d.to(self.device), v_p, y, llm, meta