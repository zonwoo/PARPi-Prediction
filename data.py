import os
from pathlib import Path
from typing import Dict, Tuple
from rdkit import Chem

BASE_DIR = Path("/PARP")
RAW_DIR = BASE_DIR / "raw"
PROCESSED_DIR = BASE_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"

def setup_directories(directories: Tuple[Path, ...]) -> None:
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

def generate_parp_file_paths(parp_types: Tuple[str, ...]) -> Dict[str, Dict[str, Path]]:
    parp_files = {}
    for parp_type in parp_types:
        parp_files[parp_type] = {
            "train": RAW_DIR / f"{parp_type}_train.csv",
            "val": RAW_DIR / f"{parp_type}_val.csv",
            "test": RAW_DIR / f"{parp_type}_test.csv",
        }
    return parp_files

PARP_TYPES = ("parp1", "parp2", "parp5a", "parp5b")

setup_directories((PROCESSED_DIR, MODELS_DIR))

PARP_FILES = generate_parp_file_paths(PARP_TYPES)


features = {
    'possible_atomic_num_list': list(range(1, 119)),
    'possible_formal_charge_list': [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5],
    'possible_chirality_list': [
        Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
        Chem.rdchem.ChiralType.CHI_OTHER,
    ],
    'possible_hybridization_list': [
        Chem.rdchem.HybridizationType.S,
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2,
        Chem.rdchem.HybridizationType.UNSPECIFIED,
    ],
    'possible_numH_list': list(range(9)),
    'possible_implicit_valence_list': list(range(7)),
    'possible_degree_list': list(range(11)),
    'possible_bonds': [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC,
    ],
    'possible_bond_dirs': [
        Chem.rdchem.BondDir.NONE,
        Chem.rdchem.BondDir.ENDUPRIGHT,
        Chem.rdchem.BondDir.ENDDOWNRIGHT,
    ],
}