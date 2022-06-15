import numpy as np
import pandas as pd
from rdkit import Chem
from mordred import Calculator, descriptors
from typing import List, Dict, Any
import concurrent.futures

def get_data():
    
    global SELECTED_DESCRIPTOR_NAMES
    global SMILES_OF_ACTIVE_COMPOUNDS
    
    # load selected mordered descriptors
    with open("data/mordred_descriptors.txt", "r") as descriptor_file:
        SELECTED_DESCRIPTOR_NAMES = descriptor_file.read().splitlines()

    # load compounds that were found active against HC by dl_mlp_class_v1_4
    active_compound_df: pd.DataFrame = pd.read_csv(
        filepath_or_buffer="data/zinc_15_m1002a_active.csv",
        names=[ # col-names somehow got lost - restored them from dl_mlp_class_v1_4.py l.474
            "id",
            "infile_smiles",
            "infile_property",
            "decoded_infile_property",
            "predicted_infile_property",
            ],
        )
    
    SMILES_OF_ACTIVE_COMPOUNDS = active_compound_df.decoded_infile_property.values
    
def insert_compound_descriptors(enum):
    i, smiles = enum
    if i % 500 == 0:
        print(i)
    all_molecule_descriptors: Dict[str, Any] = calc(Chem.MolFromSmiles(smiles))
    selected_molecule_descriptors: np.ndarray = np.array(
        [all_molecule_descriptors[d] for d in SELECTED_DESCRIPTOR_NAMES],
        dtype=np.float32
    )
    # insert selected descriptors in our template
    return selected_molecule_descriptors


if __name__ == "__main__":

    get_data()

    calc = Calculator(descriptors, ignore_3D=True)

    with concurrent.futures.ProcessPoolExecutor() as executor:
        descriptors_of_all_molecules = executor.map(insert_compound_descriptors, enumerate(SMILES_OF_ACTIVE_COMPOUNDS))
    
    df_for_saving = pd.DataFrame(
        data=list(descriptors_of_all_molecules),
        index=SMILES_OF_ACTIVE_COMPOUNDS,
        columns=SELECTED_DESCRIPTOR_NAMES,
    ).reset_index()

    df_for_saving.to_feather("data/molecule_descriptors.feather")
    print("\nDone!")
