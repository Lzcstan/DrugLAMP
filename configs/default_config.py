import os
from yacs.config import CfgNode as CN

_C = CN() # DRUG, PROTEIN, DECODER, SOLVER, RESULT, RS, COMET

# Drug feature extractor
_C.DRUG = CN()
_C.DRUG.NODE_IN_FEATS = 75
_C.DRUG.MAX_NODES = 512
_C.DRUG.PADDING = True

# Protein feature extractor
_C.PROTEIN = CN()
_C.PROTEIN.KERNEL_SIZE = [3, 6, 9]
_C.PROTEIN.PADDING = True

# Protein LLM
_C.PROTEIN.SEQ_LEN = 9 * 256
_C.PROTEIN.SITE_LEN = 9

# MLP decoder - change for PMMA
_C.DECODER = CN()
_C.DECODER.NAME = "MLP"
_C.DECODER.IN_DIM = 256
_C.DECODER.HIDDEN_DIM = 512
_C.DECODER.OUT_DIM = 128
_C.DECODER.BINARY = -1 # .yml

# SOLVER
_C.SOLVER = CN()
_C.SOLVER.MAX_EPOCH = -1 # .yml
_C.SOLVER.BATCH_SIZE = -1 # .yml
_C.SOLVER.NUM_WORKERS = -1 # .yml
_C.SOLVER.LR = -1. # .yml
_C.SOLVER.SSL_LR = -1. # .yml
_C.SOLVER.CM_LR = -1. # .yml
_C.SOLVER.SEED = -1 # Will be set in main.py

# RESULT
_C.RESULT = CN()
_C.RESULT.OUTPUT_DIR = f"{os.getcwd()}/results/" # Just a top directory
# _C.RESULT.SAVE_MODEL = True

# Real Scenario TODO
_C.RS = CN()
_C.RS.TASK = False # Will be open if the split meets specific conditions
_C.RS.METHOD = "2C2P"
_C.RS.SSL = False # .yml
_C.RS.CM = False # .yml
_C.RS.INIT_EPOCH = -1 # .yml
_C.RS.EPOCH_STEP = -1 # .yml
_C.RS.MAX_MARGIN = -1. # .yml
_C.RS.RESET_EPOCH = -1 # .yml

# Comet config, ignore it If not installed.
_C.COMET = CN()
# Please change to your own workspace name on comet.
_C.COMET.WORKSPACE = "lzcstan"
_C.COMET.PROJECT_NAME = "DrugLAMP" # .yml
_C.COMET.USE = True
_C.COMET.TAG = 'Reproduce' # Will be set in .yml & main.py


def get_cfg_defaults():
    return _C.clone()

def get_lamp_config(hidden_size):
    """Returns the PMMA configuration."""
    config = CN()
    config.n_output = 1
    config.hidden_size = hidden_size * 2
    config.num_features_llm = config.hidden_size
    config.mlha_dropout = 0

    config.transformer = CN()
    config.transformer.num_heads = 4
    config.transformer.num_p_plus_s_layers = 4
    config.transformer.attention_dropout_rate = 0 # 0.0 - 0.2
    config.transformer.dropout_rate = 0.1 # 0.1 - 0.3
    config.classifier = 'token'
    config.representation_size = None
    config.mol_len = 512
    config.feat_len = 256
    return config

def get_model_defaults(hidden_size):
    config = get_lamp_config(hidden_size)
    config.mol_len = config.feat_len
    return config