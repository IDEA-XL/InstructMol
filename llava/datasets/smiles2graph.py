"""
ref from https://github.com/UCSD-AI4H/drugchat/blob/main/dataset/smiles2graph.py
"""
from rdkit import Chem
import numpy as np
import json
import pickle
import os
from tqdm import tqdm
import random
from typing import Dict
from rdkit.Chem.rdchem import BondType, BondDir, ChiralType
import selfies as sf


BOND_TYPE = {BondType.SINGLE: 0, BondType.DOUBLE: 1, BondType.TRIPLE: 2, BondType.AROMATIC: 3}
BOND_DIR = {BondDir.NONE: 0, BondDir.ENDUPRIGHT: 1, BondDir.ENDDOWNRIGHT: 2}
CHI = {ChiralType.CHI_UNSPECIFIED: 0, ChiralType.CHI_TETRAHEDRAL_CW: 1, ChiralType.CHI_TETRAHEDRAL_CCW: 2, ChiralType.CHI_OTHER: 3}

def bond_dir(bond):
    d = bond.GetBondDir()
    return BOND_DIR[d]

def bond_type(bond):
    t = bond.GetBondType()
    return BOND_TYPE[t]

def atom_chiral(atom):
    c = atom.GetChiralTag()
    return CHI[c]

def atom_to_feature(atom):
    num = atom.GetAtomicNum() - 1
    if num == -1:
        # atom.GetAtomicNum() is 0, which is the generic wildcard atom *, may be used to symbolize an unknown atom of any element.
        # See https://biocyc.org/help.html?object=smiles
        num = 118  # normal num is [0, 117], so we use 118 to denote wildcard atom *
    return [num, atom_chiral(atom)]

def bond_to_feature(bond):
    return [bond_type(bond), bond_dir(bond)]

def smiles2graph(smiles_string)->Dict:
    """
    Converts SMILES string to graph Data object
    :input: SMILES string (str)
    :return: graph object
    """

    mol = Chem.MolFromSmiles(smiles_string)

    # atoms
    atom_features_list = []
    for atom in mol.GetAtoms():
        atom_features_list.append(atom_to_feature(atom))
    x = np.array(atom_features_list, dtype = np.int64)

    # bonds
    num_bond_features = 2
    if len(mol.GetBonds()) > 0: # mol has bonds
        edges_list = []
        edge_features_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()

            edge_feature = bond_to_feature(bond)

            # add edges in both directions
            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)

        # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
        edge_index = np.array(edges_list, dtype = np.int64).T

        # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
        edge_attr = np.array(edge_features_list, dtype = np.int64)

    else:   # mol has no bonds
        edge_index = np.empty((2, 0), dtype = np.int64)
        edge_attr = np.empty((0, num_bond_features), dtype = np.int64)

    graph = dict()
    graph['edge_index'] = edge_index
    graph['edge_feat'] = edge_attr
    graph['node_feat'] = x
    graph['num_nodes'] = len(x)

    return graph 


def construct_instruct_question(selfies_str:str=None):
    """
    Construct instruct question for each graph
    """
    question_pools = [
        'Could you give me a brief overview of this molecule?',
        'Could you provide a description of this molecule?',
        'Describe this molecule.',
        'Please give me some details about this molecule.',
        'Provide a brief overview of this molecule.',
        'Provide a description of this molecule.',
        'What can you tell me about this molecule?'
    ]
    question = random.choice(question_pools)
    if selfies_str is not None:
        question += f" The compound SELFIES sequence is: {selfies_str}."
    if random.random() < 0.5:
        question = "<image>\n" + question
    else:
        question = question + "\n<image>"
    return question


def convert_chembl(qa_json, out_dir=None):
    assert os.path.exists(qa_json), f"{qa_json} not exists"
    qa_name = os.path.basename(qa_json).split(".")[0]
    with open(qa_json, "rt") as f:
        js = json.load(f)
    out = []
    for smi, rec in tqdm(js.items()):
        if len(rec) == 0:
            continue
        graph = smiles2graph(smi)
        for question, answer in rec:
            out.append({
                "graph": graph, 
                "conversations": [
                    {"from": "human", "value": construct_instruct_question() },
                    {"from": "gpt", "value": answer}
                ],
            })
    print(f"Successfully convert {len(out)} samples.")

    if out_dir is None:
        out_dir = os.path.dirname(qa_json)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    with open(os.path.join(out_dir, qa_name+'.pkl'), "wb") as f:
        pickle.dump(out, f)
        
        
def convert_chebi20(txt, out_dir=None, add_selfies=False):
    assert os.path.exists(txt), f"{txt} not exists"
    qa_name = os.path.basename(txt).split(".")[0]
    out = []
    with open(txt, "rt") as f:
        f.readline()
        for i, line in enumerate(f.readlines()):
            cid, smi, desc = line.strip().split("\t")
            selfies_str = None
            if add_selfies:
                try:
                    selfies_str = sf.encoder(smi)
                except:
                    selfies_str = ""
            graph = smiles2graph(smi)
            out.append({
                "graph": graph, 
                "conversations": [
                    {"from": "human", "value": construct_instruct_question(selfies_str) },
                    {"from": "gpt", "value": desc}
                ],
            })
    print(f"Successfully convert {len(out)} samples.")
    if out_dir is None:
        out_dir = os.path.dirname(txt)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    if add_selfies:
        qa_name += "+selfies"
    with open(os.path.join(out_dir, qa_name+'.pkl'), "wb") as f:
        pickle.dump(out, f)
        


if __name__ == '__main__':
    # graph = smiles2graph('O1C=C[C@H]([C@H]1O2)c3c2cc(OC)c4c3OC(=O)C5=C4CCC(=O)5')
    # print(graph)
    # qa_json = '/comp_robot/rentianhe/caohe/AIDD/DATA/MolFM/pubcgraphemsft_desc/test.json'
    # convert_chembl(qa_json)
    
    for split in ['train', 'test', 'validation']:
        txt = f'/cto_labs/AIDD/DATA/MolT5/ChEBI-20_data/{split}.txt'
        convert_chebi20(txt, add_selfies=True)
