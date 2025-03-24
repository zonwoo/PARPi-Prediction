import torch
import pandas as pd
import os
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data
from rdkit import Chem
from config import root_dir
from config import features

def mol_to_graph_data_obj_simple(mol):
    atom_features_list = []
    atoms = mol.GetAtoms()
    for atom in atoms:
        atomic_num = atom.GetAtomicNum()
        formal_charge = atom.GetFormalCharge()
        chiral_tag = atom.GetChiralTag()
        hybridization = atom.GetHybridization()
        total_num_hs = atom.GetTotalNumHs()
        implicit_valence = atom.GetImplicitValence()
        degree = atom.GetDegree()

        atomic_num_index = features['possible_atomic_num_list'].index(atomic_num)
        formal_charge_index = features['possible_formal_charge_list'].index(formal_charge)
        chiral_tag_index = features['possible_chirality_list'].index(chiral_tag)
        hybridization_index = features['possible_hybridization_list'].index(hybridization)
        num_h_index = features['possible_numH_list'].index(total_num_hs)
        implicit_valence_index = features['possible_implicit_valence_list'].index(implicit_valence)
        degree_index = features['possible_degree_list'].index(degree)

        atom_feature = [atomic_num_index, formal_charge_index, chiral_tag_index, hybridization_index, num_h_index, implicit_valence_index, degree_index]
        atom_features_list.append(atom_feature)

    x = torch.tensor(atom_features_list, dtype=torch.float)

    edges_list = []
    edge_features_list = []
    bonds = mol.GetBonds()
    for bond in bonds:
        begin_atom_idx = bond.GetBeginAtomIdx()
        end_atom_idx = bond.GetEndAtomIdx()
        bond_type = bond.GetBondType()
        bond_dir = bond.GetBondDir()

        bond_type_index = features['possible_bonds'].index(bond_type)
        bond_dir_index = features['possible_bond_dirs'].index(bond_dir)

        edge_feature = [bond_type_index, bond_dir_index]
        edges_list.append((begin_atom_idx, end_atom_idx))
        edge_features_list.append(edge_feature)
        edges_list.append((end_atom_idx, begin_atom_idx))
        edge_features_list.append(edge_feature)

    if edges_list:
        edge_index = torch.tensor(edges_list, dtype=torch.long).T
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    if edge_features_list:
        edge_attr = torch.tensor(edge_features_list, dtype=torch.long)
    else:
        edge_attr = torch.empty((0, 2), dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    return data

def process_data(df):
    graphs = []
    smiles_list = df['canonical_smiles'].tolist()
    label_list = df['standard_value'].tolist()
    for smiles, label in zip(smiles_list, label_list):
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            graph = mol_to_graph_data_obj_simple(mol)
            label_tensor = torch.tensor([int(label)], dtype=torch.float)
            graph.y = label_tensor
            graphs.append(graph)
    return graphs

class MoleculeDataset(InMemoryDataset):
    def __init__(self, root, file_name, transform=None, pre_transform=None):
        self.root = root
        self.file_name = file_name
        processed_file_name = f'{self.file_name.split("/")[-1].split(".")[0]}_processed.pt'
        self.processed_path = os.path.join(self.root, 'processed', processed_file_name)
        super().__init__(root, transform, pre_transform)

        if os.path.exists(self.processed_path):
            loaded_data = torch.load(self.processed_path)
            self.data, self.slices = loaded_data
        else:
            self.data, self.slices = self.process()

    @property
    def raw_file_names(self):
        return [self.file_name]

    @property
    def processed_file_names(self):
        processed_file_name = f'{self.file_name.split("/")[-1].split(".")[0]}_processed.pt'
        return [processed_file_name]

    @property
    def processed_dir(self):
        return os.path.join(self.root, 'processed')

    def process(self):
        file_path = os.path.join(self.root, self.file_name)
        df = pd.read_csv(file_path)
        graphs = process_data(df)
        data_list = graphs

        for i, graph in enumerate(data_list):
            num_nodes = graph.num_nodes
            batch_data = [i] * num_nodes
            batch_tensor = torch.tensor(batch_data, dtype=torch.long)
            graph.batch = batch_tensor

        collated_data = self.collate(data_list)
        data, slices = collated_data
        torch.save((data, slices), self.processed_path)
        return data, slices