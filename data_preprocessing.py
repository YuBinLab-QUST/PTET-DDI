
import random
import math

import torch
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch
from rdkit import Chem
import pandas as pd
import numpy as np

from tqdm import tqdm
import ast

from chemberta import pre_chemBERTa


def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def atom_features(atom, atom_feat_list,
                  explicit_H=True,
                  use_chirality=True):
    results = one_of_k_encoding_unk(
        atom.GetSymbol(),
        atom_feat_list) + [atom.GetDegree() / 10, atom.GetFormalCharge(), atom.GetNumRadicalElectrons()] + \
              one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6]) + \
              one_of_k_encoding_unk(atom.GetHybridization(), [
                  Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
                  Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.
                                    SP3D, Chem.rdchem.HybridizationType.SP3D2
              ]) + [atom.GetIsAromatic()]
    if explicit_H:
        results = results + [atom.GetTotalNumHs()]

    if use_chirality:
        try:
            results = results + one_of_k_encoding_unk(
                atom.GetProp('_CIPCode'),
                ['R', 'S']) + [atom.HasProp('_ChiralityPossible')]
        except:
            results = results + [False, False
                                 ] + [atom.HasProp('_ChiralityPossible')]

    results = np.array(results).astype(np.float32)

    return torch.from_numpy(results)


def edge_features(bond):
    '''
    Get bond features
    '''
    bond_type = bond.GetBondType()
    return torch.tensor([
        bond_type == Chem.rdchem.BondType.SINGLE,  # 单键
        bond_type == Chem.rdchem.BondType.DOUBLE,  # 双键
        bond_type == Chem.rdchem.BondType.TRIPLE,  # 三键
        bond_type == Chem.rdchem.BondType.AROMATIC,  # 芳香键
        bond.GetIsConjugated(),
        bond.IsInRing()]).long()


def get_mol_edge_list_and_feat_mtx(mol_graph, atom_feat_list):
    n_features = [(atom.GetIdx(), atom_features(atom, atom_feat_list)) for atom in mol_graph.GetAtoms()]
    n_features.sort()
    _, n_features = zip(*n_features)
    n_features = torch.stack(n_features)

    edge_list = torch.LongTensor(
        [(b.GetBeginAtomIdx(), b.GetEndAtomIdx(), *edge_features(b)) for b in mol_graph.GetBonds()])
    edge_list, edge_feats = (edge_list[:, :2], edge_list[:, 2:].float()) if len(edge_list) else (
        torch.LongTensor([]), torch.FloatTensor([]))

    edge_list = torch.cat([edge_list, edge_list[:, [1, 0]]], dim=0) if len(edge_list) else edge_list
    edge_feats = torch.cat([edge_feats] * 2, dim=0) if len(edge_feats) else edge_feats

    new_edge_index = edge_list.T

    z = torch.LongTensor([torch.nonzero(row[:len(atom_feat_list)] == 1).squeeze().item() if torch.sum(row[:len(atom_feat_list)] == 1) == 1 else None for row in n_features])

    return [new_edge_index, n_features, edge_feats, z]


class DrugDataset(Dataset):
    def __init__(self, all_molfeature, drugs_text_feat, tri_list, ratio=1.0, disjoint_split=True, shuffle=True):
        ''''disjoint_split: Consider whether entities should appear in one and only one split of the data
        '''
        self.tri_list = []
        self.ratio = ratio
        self.all_molfeature = all_molfeature
        self.drugs_text_feat = drugs_text_feat


        for h, t, r, n, *_ in tri_list:
            if ((h in all_molfeature) and (t in all_molfeature)):
                self.tri_list.append((h, t, r, n))

        if shuffle:
            random.seed(6)
            random.shuffle(self.tri_list)
        limit = math.ceil(len(self.tri_list) * ratio)  # 数量还要向上取整？？
        self.tri_list = self.tri_list[:limit]

    def __len__(self):
        return len(self.tri_list)

    def __getitem__(self, index):
        return self.tri_list[index]

    def collate_fn(self, batch):

        pos_rels = []
        pos_h_samples = []
        pos_t_samples = []

        neg_rels = []
        neg_h_samples = []
        neg_t_samples = []
        pos_h_chem_samples = []
        pos_t_chem_samples = []
        neg_h_chem_samples = []
        neg_t_chem_samples = []

        for h, t, r, neg_sample in batch:
            pos_rels.append(r)
            Neg_ID, Ntype = neg_sample.split('$')
            h_data = self.__create_graph_data(h)
            t_data = self.__create_graph_data(t)
            n_data = self.__create_graph_data(Neg_ID)
            chem_h_data = Data(x=self.drugs_text_feat[h])
            chem_t_data = Data(x=self.drugs_text_feat[t])
            chem_n_data = Data(x=self.drugs_text_feat[Neg_ID])


            pos_h_samples.append(h_data)
            pos_t_samples.append(t_data)
            pos_h_chem_samples.append(chem_h_data)
            pos_t_chem_samples.append(chem_t_data)

            if Ntype == 'h':
                neg_rels.append(r)
                neg_h_samples.append(n_data)
                neg_t_samples.append(t_data)

                neg_h_chem_samples.append(chem_n_data)
                neg_t_chem_samples.append(chem_t_data)
            else:
                neg_rels.append(r)
                neg_h_samples.append(h_data)
                neg_t_samples.append(n_data)

                neg_h_chem_samples.append(chem_h_data)
                neg_t_chem_samples.append(chem_n_data)

        pos_h_samples = Batch.from_data_list(pos_h_samples)
        pos_t_samples = Batch.from_data_list(pos_t_samples)
        pos_rels = torch.LongTensor(pos_rels).unsqueeze(0)
        pos_h_chem_samples = Batch.from_data_list(pos_h_chem_samples)
        pos_t_chem_samples = Batch.from_data_list(pos_t_chem_samples)

        pos_tri = (pos_h_samples, pos_t_samples, pos_h_chem_samples, pos_t_chem_samples, pos_rels)

        neg_h_samples = Batch.from_data_list(neg_h_samples)
        neg_t_samples = Batch.from_data_list(neg_t_samples)
        neg_rels = torch.LongTensor(neg_rels).unsqueeze(0)
        neg_h_chem_samples = Batch.from_data_list(neg_h_chem_samples)
        neg_t_chem_samples = Batch.from_data_list(neg_t_chem_samples)

        neg_tri = (neg_h_samples, neg_t_samples, neg_h_chem_samples, neg_t_chem_samples, neg_rels)

        return pos_tri, neg_tri

    def __create_graph_data(self, id):
        edge_index = self.all_molfeature[id][0]
        n_features = self.all_molfeature[id][1]
        edge_feature = self.all_molfeature[id][2]

        z = self.all_molfeature[id][3]
        coords = torch.tensor(self.all_molfeature[id][4])
        return Data(x=n_features, edge_index=edge_index, edge_attr=edge_feature, pos=coords, z=z)

class DrugDataLoader(DataLoader):
    def __init__(self, data, **kwargs):
        super().__init__(data, collate_fn=data.collate_fn, **kwargs)

def split_train_valid(data, val_ratio=0.125):
    data = np.array(data)
    cv_split = StratifiedShuffleSplit(n_splits=2, test_size=val_ratio, random_state=1)
    train_index, val_index = next(iter(cv_split.split(X=data, y=data[:, 2])))
    train_tup = data[train_index]
    val_tup = data[val_index]
    train_tup = [(tup[0], tup[1], int(tup[2]), tup[3])for tup in train_tup]
    val_tup = [(tup[0], tup[1], int(tup[2]), tup[3])for tup in val_tup]

    return train_tup, val_tup

def load_data(dataset, data_size_ratio, batch_size, fold_i):
    chem_berta = pre_chemBERTa()

    df_drugs_smiles = pd.read_csv(f'dataset/{dataset}/drug_smiles.csv', dtype=str)

    drug_id_mol_graph_tup = [(id, Chem.MolFromSmiles(smiles.strip())) for id, smiles in
                             tqdm(zip(df_drugs_smiles['drug_id'], df_drugs_smiles['smiles']), desc="Processing SMILES",
                                  ncols=150)]

    drugs_text_feat = {id: chem_berta(smiles.strip()) for id, smiles in
                       tqdm(zip(df_drugs_smiles['drug_id'], df_drugs_smiles['smiles']), desc="Processing Text_feature",
                            ncols=150)}

    df_drugs_coords = pd.read_csv(f'dataset/{dataset}/drug_id_coords_nonHs.csv', dtype=str)
    drug_id_coords = {row.iloc[0]:
                          [ast.literal_eval(item)
                           if item != ' '
                           else [1]
                           for item in row.iloc[1:].dropna().tolist()]
                      for index, row in df_drugs_coords.iterrows()}

    drug_id_coords = {key: np.array(value).astype(np.float32) for key, value in
                      drug_id_coords.items()}

    new_coord = {key: value for key, value in drug_id_coords.items() if key in list(df_drugs_smiles['drug_id'])}

    drug_id_coords = new_coord

    atom_feat_list = set()
    for line in drug_id_mol_graph_tup:
        for atom in line[1].GetAtoms():
            atom_feat_list.add(atom.GetSymbol())
    atom_feat_list = sorted(atom_feat_list)

    MOL_EDGE_LIST_FEAT_MTX = {drug_id: get_mol_edge_list_and_feat_mtx(mol, atom_feat_list)
                              for drug_id, mol in tqdm(drug_id_mol_graph_tup, desc="Processing Mol_feature", ncols=150)}

    MOL_EDGE_LIST_FEAT_MTX = {drug_id: mol for drug_id, mol in
                              tqdm(MOL_EDGE_LIST_FEAT_MTX.items(), desc="Processing Mol_feature", ncols=150) if
                              mol is not None}

    for drug_id, coods in drug_id_coords.items():
        MOL_EDGE_LIST_FEAT_MTX[drug_id].append(torch.from_numpy(coods))

    df_ddi_train = pd.read_csv(f'dataset/{dataset}/train_fold{fold_i}.csv', dtype=str)
    df_ddi_test = pd.read_csv(f'dataset/{dataset}/test_fold{fold_i}.csv', dtype=str)

    train_tup = [(h, t, r, n) for h, t, r, n in
                 zip(df_ddi_train['Drug1_ID'], df_ddi_train['Drug2_ID'], df_ddi_train['Y'],
                     df_ddi_train['Neg samples'])]

    train_tup, val_tup = split_train_valid(train_tup, val_ratio=0.125)
    test_tup = [(h, t, int(r.strip()), n) for h, t, r, n in
                zip(df_ddi_test['Drug1_ID'], df_ddi_test['Drug2_ID'], df_ddi_test['Y'], df_ddi_test['Neg samples'])]

    train_data = DrugDataset(MOL_EDGE_LIST_FEAT_MTX, drugs_text_feat, train_tup, ratio=data_size_ratio)
    val_data = DrugDataset(MOL_EDGE_LIST_FEAT_MTX, drugs_text_feat, val_tup, ratio=data_size_ratio, disjoint_split=False)
    test_data = DrugDataset(MOL_EDGE_LIST_FEAT_MTX, drugs_text_feat, test_tup, ratio=data_size_ratio, disjoint_split=False)

    print(
        f"Fold_{fold_i} Training with {len(train_data)} samples,and valing with {len(val_data)}, and testing with {len(test_data)}")

    train_data_loader = DrugDataLoader(train_data, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4)
    val_data_loader = DrugDataLoader(val_data, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4)
    test_data_loader = DrugDataLoader(test_data, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4)

    return train_data_loader, val_data_loader, test_data_loader