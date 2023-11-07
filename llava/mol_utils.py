from rdkit import Chem

def check_smiles_validity(smiles:str)->bool:
    # check if valid smiles
    m = Chem.MolFromSmiles(smiles,sanitize=False)
    if m is None:
        return False
    return True

