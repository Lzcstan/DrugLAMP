comet_support = True
try:
    from comet_ml import Experiment
except ImportError as e:
    print("Comet ML is not installed, ignore the comet experiment monitor")
    comet_support = False

import time
import esm.pretrained as esp
import argparse, sys, warnings, os

n_layer2esp_fns = {
    48: esp.esm2_t48_15B_UR50D,
    36: esp.esm2_t36_3B_UR50D,
    33: esp.esm2_t33_650M_UR50D,
    30: esp.esm2_t30_150M_UR50D,
    12: esp.esm2_t12_35M_UR50D
}

parser = argparse.ArgumentParser(description="DrugLAMP for DTI prediction") # Tid: HACK
parser.add_argument('--seed', default=42, help="which seed to use", type=int)
parser.add_argument('--no-comet', help="do not use comet.ml", action='store_true')
parser.add_argument('--data', required=True, type=str, metavar='TASK', help='dataset')
parser.add_argument('--model', required=True, help="which model to do DTI prediction", type=str)
parser.add_argument('--n-layer', default=30, help="which esp.esm2 llm to use", type=int, choices=list(n_layer2esp_fns.keys()))
parser.add_argument('--split', default='random', type=str, metavar='S', help="split task", choices=['random', 'cold', 'cluster', 'Tcpi'])
parser.add_argument('--devices', type=str, help='CUDA visible devices')

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.devices if args.devices else ""

from rich import print
from model import MInterface
from trainer import ExpModule
from configs import get_cfg_defaults
from handler import MultiModalityDataset
from pytorch_lightning.loggers import CometLogger
from utils import set_seed, multimodality_collate_func, mkdir

import torch
from torch.utils.data import DataLoader
torch.set_float32_matmul_precision('medium')
device = torch.device('cpu')

def main():
    model_name = args.model
    model_cfg = f'configs/{model_name}.yaml'
    n_layer = args.n_layer
    ds_name = args.data
    ds_split = args.split
    seed = args.seed
    comet_support = not args.no_comet

    torch.cuda.empty_cache()
    cfg = get_cfg_defaults()
    cfg.merge_from_file(model_cfg)
    cfg.SOLVER.SEED = seed
    set_seed(cfg.SOLVER.SEED)
    timestamp = time.strftime("%m%d_%H%M") # TODO
    exp_name = f"{ds_name}-{ds_split}-"\
        + f"{model_cfg[model_cfg.rfind('/') + 1: model_cfg.rfind('.')]}-"\
            + timestamp
    cfg.RESULT.OUTPUT_DIR += exp_name.replace('-', '/')
    mkdir(cfg.RESULT.OUTPUT_DIR)

    ds_folder = f'datasets/{ds_name}'
    ds_folder = os.path.join(ds_folder, ds_split)
    if ds_split == 'cluster' or ds_split == 'Tcpi':
        cfg.RS.TASK = True

    if not comet_support:
        cfg.COMET.USE = False
        print('Choose not to use the Comet.ml...')

    print(f"Config yaml: {model_cfg}")
    print(f"Hyperparameters: {dict(cfg)}")

    esp_fn = n_layer2esp_fns[n_layer]
    gen_embed = cfg.SOLVER.SEED == 40
    max_drug_atoms = cfg.DRUG.MAX_NODES
    try:
        if cfg.RS.TASK:
            train_dataset = MultiModalityDataset(ds_folder, 'source_train.csv', esp_fn, n_layer, device, gen_embed=gen_embed, max_drug_atoms=max_drug_atoms)
            test_target_dataset = MultiModalityDataset(ds_folder, 'target_test.csv', esp_fn, n_layer, device, max_drug_atoms=max_drug_atoms)
        else:
            train_dataset = MultiModalityDataset(ds_folder, 'train.csv', esp_fn, n_layer, device, gen_embed=gen_embed, max_drug_atoms=max_drug_atoms)
            val_dataset = MultiModalityDataset(ds_folder, "val.csv", esp_fn, n_layer, device, max_drug_atoms=max_drug_atoms)
            test_dataset = MultiModalityDataset(ds_folder, "test.csv", esp_fn, n_layer, device, max_drug_atoms=max_drug_atoms)
    except ConnectionError as e:
        print(e)
        sys.exit(1)
        
    logger = None
    if cfg.COMET.USE and comet_support:
        # LLM specific config
        cfg.COMET.TAG += f' with CM normed p-tri-margin loss. m_ori={cfg.RS.MAX_MARGIN}, n_re={cfg.RS.RESET_EPOCH}'

        logger = CometLogger(
            project_name=cfg.COMET.PROJECT_NAME,   
            workspace=cfg.COMET.WORKSPACE,
            auto_output_logging="simple",
            log_graph=True,
            log_code=False,
            log_git_metadata=False,
            log_git_patch=False,
            auto_param_logging=False,
            auto_metric_logging=False
        )
        hyper_params = {
            "LR": cfg.SOLVER.LR,
            "Output_dir": cfg.RESULT.OUTPUT_DIR,
            "SSL_use": cfg.RS.SSL,
            "CM_use": cfg.RS.CM,
            "RS_task": cfg.RS.TASK,
        }
        if hyper_params['SSL_use']:
            ssl_hyper_params = {
                "SSL_epoch_step": cfg.RS.EPOCH_STEP,
                "SSL_optim_lr": cfg.SOLVER.SSL_LR
            }
            hyper_params.update(ssl_hyper_params)
        if hyper_params['CM_use']:
            cm_hyper_params = {
                "CM_init_epoch": cfg.RS.INIT_EPOCH,
                "CM_optim_lr": cfg.SOLVER.CM_LR
            }
            hyper_params.update(cm_hyper_params)
        logger.experiment.log_parameters(hyper_params)
        if cfg.COMET.TAG is not None:
            logger.experiment.add_tag(cfg.COMET.TAG)
        logger.experiment.set_name(exp_name)

    params = {'batch_size': cfg.SOLVER.BATCH_SIZE, 'shuffle': True, 'num_workers': cfg.SOLVER.NUM_WORKERS,
              'drop_last': True, 'collate_fn': multimodality_collate_func}
    if cfg.RS.TASK:
        train_generator = DataLoader(train_dataset, **params)
        params['shuffle'] = False
        params['drop_last'] = False
        params['batch_size'] = 1
        val_generator = DataLoader(test_target_dataset, **params)
        test_generator = DataLoader(test_target_dataset, **params)
    else:
        train_generator = DataLoader(train_dataset, **params)
        params['shuffle'] = False
        params['drop_last'] = False
        params['batch_size'] = 1
        val_generator = DataLoader(val_dataset, **params)
        test_generator = DataLoader(test_dataset, **params)

    model_interface = MInterface(model_name, cfg)
    model = model_interface.load_model(**vars(train_dataset))

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.SOLVER.LR)
    opt_ssl = torch.optim.AdamW(model.parameters(), lr=cfg.SOLVER.SSL_LR) if cfg.RS.SSL else None
    opt_cm = torch.optim.AdamW(model.parameters(), lr=cfg.SOLVER.CM_LR) if cfg.RS.CM else None

    torch.backends.cudnn.benchmark = True
    trainer = ExpModule(model, opt, train_generator, val_generator, test_generator,
                        opt_ssl=opt_ssl,
                        opt_cm=opt_cm,
                        split=ds_split,
                        logger=logger, **cfg)
    trainer.run_experiment()

    print('Rank: ', trainer.global_rank)
    print(f"Directory for saving result: {cfg.RESULT.OUTPUT_DIR}")
    
if __name__ == '__main__':
    s = time.time()
    main()
    e = time.time()
    print(f"Total running time: {round(e - s, 2)}s")
