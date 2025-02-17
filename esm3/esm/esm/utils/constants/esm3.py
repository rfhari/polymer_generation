import os
from functools import cache
from pathlib import Path

from huggingface_hub import snapshot_download

SEQUENCE_BOS_TOKEN = 0
SEQUENCE_PAD_TOKEN = 1
SEQUENCE_EOS_TOKEN = 2
SEQUENCE_CHAINBREAK_TOKEN = 31
SEQUENCE_MASK_TOKEN = 32

VQVAE_CODEBOOK_SIZE = 4096
VQVAE_SPECIAL_TOKENS = {
    "MASK": VQVAE_CODEBOOK_SIZE,
    "EOS": VQVAE_CODEBOOK_SIZE + 1,
    "BOS": VQVAE_CODEBOOK_SIZE + 2,
    "PAD": VQVAE_CODEBOOK_SIZE + 3,
    "CHAINBREAK": VQVAE_CODEBOOK_SIZE + 4,
}
VQVAE_DIRECTION_LOSS_BINS = 16
VQVAE_PAE_BINS = 64
VQVAE_MAX_PAE_BIN = 31.0
VQVAE_PLDDT_BINS = 50

STRUCTURE_MASK_TOKEN = VQVAE_SPECIAL_TOKENS["MASK"]
STRUCTURE_BOS_TOKEN = VQVAE_SPECIAL_TOKENS["BOS"]
STRUCTURE_EOS_TOKEN = VQVAE_SPECIAL_TOKENS["EOS"]
STRUCTURE_PAD_TOKEN = VQVAE_SPECIAL_TOKENS["PAD"]
STRUCTURE_CHAINBREAK_TOKEN = VQVAE_SPECIAL_TOKENS["CHAINBREAK"]
STRUCTURE_UNDEFINED_TOKEN = 955

SASA_PAD_TOKEN = 0

SS8_PAD_TOKEN = 0

INTERPRO_PAD_TOKEN = 0

RESIDUE_PAD_TOKEN = 0

CHAIN_BREAK_STR = "|"

SEQUENCE_BOS_STR = "<cls>"
SEQUENCE_EOS_STR = "<eos>"

MASK_STR_SHORT = "_"
SEQUENCE_MASK_STR = "<mask>"
SASA_MASK_STR = "<unk>"
SS8_MASK_STR = "<unk>"

# fmt: off
SEQUENCE_VOCAB = [
    "<cls>", "<pad>", "<eos>", "<unk>",
    "L", "A", "G", "V", "S", "E", "R", "T", "I", "D", "P", "K",
    "Q", "N", "F", "Y", "M", "H", "W", "C", "X", "B", "U", "Z",
    "O", ".", "-", "|",
    "<mask>",
]
# fmt: on

SSE_8CLASS_VOCAB = "GHITEBSC"
SSE_3CLASS_VOCAB = "HEC"
SSE_8CLASS_TO_3CLASS_MAP = {
    "G": "H",
    "H": "H",
    "I": "H",
    "T": "C",
    "E": "E",
    "B": "E",
    "S": "C",
    "C": "C",
}

SASA_DISCRETIZATION_BOUNDARIES = [
    0.8,
    4.0,
    9.6,
    16.4,
    24.5,
    32.9,
    42.0,
    51.5,
    61.2,
    70.9,
    81.6,
    93.3,
    107.2,
    125.4,
    151.4,
]

MAX_RESIDUE_ANNOTATIONS = 16


TFIDF_VECTOR_SIZE = 58641


@staticmethod
@cache
def data_root(model: str):
    if "INFRA_PROVIDER" in os.environ:
        return Path("")
    # Try to download from hugginface if it doesn't exist
    if model.startswith("esm3"):
        path = Path(snapshot_download(repo_id="EvolutionaryScale/esm3-sm-open-v1"))
    elif model.startswith("esmc-300"):
        path = Path(snapshot_download(repo_id="EvolutionaryScale/esmc-300m-2024-12"))
    elif model.startswith("esmc-600"):
        path = Path(snapshot_download(repo_id="EvolutionaryScale/esmc-600m-2024-12"))
    else:
        raise ValueError(f"{model=} is an invalid model name.")
    return path


IN_REPO_DATA_FOLDER = Path(__file__).parents[2] / "data"

INTERPRO_ENTRY = IN_REPO_DATA_FOLDER / "entry_list_safety_29026.list"
INTERPRO_HIERARCHY = IN_REPO_DATA_FOLDER / "ParentChildTreeFile.txt"
INTERPRO2GO = IN_REPO_DATA_FOLDER / "ParentChildTreeFile.txt"
INTERPRO_2ID = "data/tag_dict_4_safety_filtered.json"

LSH_TABLE_PATHS = {"8bit": "data/hyperplanes_8bit_58641.npz"}

KEYWORDS_VOCABULARY = (
    IN_REPO_DATA_FOLDER / "keyword_vocabulary_safety_filtered_58641.txt"
)
KEYWORDS_IDF = IN_REPO_DATA_FOLDER / "keyword_idf_safety_filtered_58641.npy"

RESID_CSV = "data/uniref90_and_mgnify90_residue_annotations_gt_1k_proteins.csv"
INTERPRO2KEYWORDS = IN_REPO_DATA_FOLDER / "interpro_29026_to_keywords_58641.csv"

elements = ['H', 'He',	 'e',	 'Li',	 'L',	 'i',	 'Be',	 'B',	 'C',	 'N',	 'O',	 'F',	 'Ne',	 'Na',	 'a',	 'Mg',	 'M',	 'g',	 'Al',	 'A',	 'l',	 'Si',	 'S',	 'P',	 'Cl',	 'Ar',	 'r',	 'K',	 'Ca',	 'Sc',	 'c',	 'Ti',	 'T',	 'V',	 'Cr',	 'Mn',	 'n',	 'Fe',	 'Co',	 'o',	 'Ni',	 'Cu',	 'u',	 'Zn',	 'Z',	 'Ga',	 'G',	 'Ge',	 'As',	 's',	 'Se',	 'Br',	 'Kr',	 'Rb',	 'R',	 'b',	 'Sr',	 'Y',	 'Zr',
 'Nb',	 'Mo',	 'Tc',	 'Ru',	 'Rh',	 'h',	 'Pd',	 'd',	 'Ag',	 'Cd',	 'In',	 'I',	 'Sn',	 'Sb',	 'Te',	 'Xe',	 'X',	 'Cs',	 'Ba',	 'La',	 'Hf',	 'f',	 'Ta',	 'W',	 'Re',	 'Os',	 'Ir',	 'Pt',	 't',	 'Au',	 'Hg',	 'Tl',	 'Pb',	 'Bi',	 'Po',	 'At',	 'Rn',	 'Fr',	 'Ra',	 'Ac',	 'Rf',	 'Db',	 'D',	 'Sg',	 'Bh',	 'Hs',	 'Mt',	 'Ds',	 'Rg',	 'Cn',	 'Nh',	 'Fl',	 'Mc',	 'Lv',	 'v',	 'Ts',	 'Og',	 'Ce',	 'Pr',
 'Nd',	 'Pm',	 'm',	 'Sm',	 'Eu',	 'E',	 'Gd',	 'Tb',	 'Dy',	 'y',	 'Ho',	 'Er',	 'Tm',	 'Yb',	 'Lu',	 'Th',	 'Pa',	 'U',	 'Np',	 'p',	 'Pu',	 'Am',	 'Cm',	 'Bk',	 'k',	 'Cf',	 'Es',	 'Fm',	 'Md',	 'No',	 'Lr']
 
# elements = ['H', 'He', 'e', 'Li', 'L', 'i', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr']
# elements = [ 'H',	 'He',	 'H',	 'e',	 'Li',	 'L',	 'i',	 'Be',	 'B',	 'e',	 'B',	 'C',	 'N',	 'O',	 'F',	 'Ne',	 'N',	 'e',	 'Na',	 'N',	 'a',	 'Mg',	 'M',	 'g',	 'Al',	 'A',	 'l',	 'Si',	 'S',	 'i',	 'P',	 'S',	 'Cl',	 'C',	 'l',	 'Ar',	 'A',	 'r',	 'K',	 'Ca',	 'C',	 'a',	 'Sc',	 'S',	 'c',	 'Ti',	 'T',	 'i',	 'V',	 'Cr',	 'C',	 'r',	 'Mn',	 'M',	 'n',	 'Fe',	 'F',	 'e',	 'Co',	 'C',	 'o',	 'Ni',	 'N',	 'i',	 'Cu',	 'C',	 'u',	 'Zn',	 'Z',	 'n',	 'Ga',	 'G',	 'a',	 'Ge',	 'G',	 'e',	 'As',	 'A',	 's',	 'Se',	 'S',	 'e',	 'Br',	 'B',	 'r',	 'Kr',	 'K',	 'r',	 'Rb',	 'R',	 'b',	 'Sr',	 'S',	 'r',	 'Y',	 'Zr',	 'Z',	 'r',	 'Nb',	 'N',	 'b',	 'Mo',	 'M',	 'o',	 'Tc',	 'T',	 'c',	 'Ru',	 'R',	 'u',	 'Rh',	 'R',	 'h',	 'Pd',	 'P',	 'd',	 'Ag',	 'A',	 'g',	 'Cd',	 'C',	 'd',	 'In',	 'I',	 'n',	 'Sn',	 'S',	 'n',	 'Sb',	 'S',	 'b',	 'Te',	 'T',	 'e',	 'I',	 'Xe',	 'X',	 'e',	 'Cs',	 'C',	 's',	 'Ba',	 'B',	 'a',	 'La',	 'L',	 'a',	 'Hf',	 'H',	 'f',																										
#             'Ta',	 'T',	 'a',	 'W',	 'Re',	 'R',	 'e',	 'Os',	 'O',	 's',	 'Ir',	 'I',	 'r',	 'Pt',	 'P',	 't',	 'Au',	 'A',	 'u',	 'Hg',	 'H',	 'g',	 'Tl',	 'T',	 'l',	 'Pb',	 'P',	 'b',	 'Bi',	 'B',	 'i',	 'Po',	 'P',	 'o',	 'At',	 'A',	 't',	 'Rn',	 'R',	 'n',	 'Fr',	 'F',	 'r',	 'Ra',	 'R',	 'a',	 'Ac',	 'A',	 'c',	 'Rf',	 'R',	 'f',	 'Db',	 'D',	 'b',	 'Sg',	 'S',	 'g',	 'Bh',	 'B',	 'h',	 'Hs',	 'H',	 's',	 'Mt',	 'M',	 't',	 'Ds',	 'D',	 's',	 'Rg',	 'R',	 'g',	 'Cn',	 'C',	 'n',	 'Nh',	 'N',	 'h',	 'Fl',	 'F',	 'l',	 'Mc',	 'M',	 'c',	 'Lv',	 'L',	 'v',	 'Ts',	 'T',	 's',	 'Og',	 'O',	 'g',	 'Ce',	 'C',	 'e',	 'Pr',	 'P',	 'r',	 'Nd',	 'N',	 'd',	 'Pm',	 'P',	 'm',	 'Sm',	 'S',	 'm',	 'Eu',	 'E',	 'u',	 'Gd',	 'G',	 'd',	 'Tb',	 'T',	 'b',	 'Dy',	 'D',	 'y',	 'Ho',	 'H',	 'o',	 'Er',	 'E',	 'r',	 'Tm',	 'T',	 'm',	 'Yb',	 'Y',	 'b',	 'Lu',	 'L',	 'u',	 'Th',	 'T',	 'h',	 'Pa',	 'P',	 'a',	 'U',	 'Np',	 'N',	 'p',	 'Pu',	 'P',	 'u',	 'Am',	 'A',	 'm',	 'Cm',	 'C',	 'm',	 'Bk',	 'B',	 'k',	 'Cf',	 'C',	 'f',	 'Es',	 'E',	 's',	 'Fm',	 'F',	 'm',	 'Md',	 'M',	 'd',	 'No',	 'N',	 'o',	 'Lr',	 'L',	 'r'
#             ]

small_elements = [i.lower() for i in elements] 

# els = ['C', 'Br', 'Cl', 'N', 'O', 'S', 'P', 'F', 'I', 'b', 'c', 'n', 'o', 's', 'p', "H", "Si"]
POLYMER_VOCAB =[
    "[*]", "*",
    "(", ")", "=", "@", "#",
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
    "-", "+",
    "%", "[", "]","<unk>"
]

POLYMER_VOCAB += elements # + small_elements