from ast import Global
import json
import optuna
import numpy as np
import pandas as pd
from logging import getLogger
from pathlib import Path
from daisyRec.daisy.model.MFRecommender import MF
from daisyRec.daisy.model.FMRecommender import FM
from daisyRec.daisy.model.NFMRecommender import NFM
from daisyRec.daisy.model.NGCFRecommender import NGCF
from daisyRec.daisy.model.EASERecommender import EASE
from daisyRec.daisy.model.SLiMRecommender import SLiM
from daisyRec.daisy.model.VAECFRecommender import VAECF
from daisyRec.daisy.model.NeuMFRecommender import NeuMF
from daisyRec.daisy.model.PopRecommender import MostPop
from daisyRec.daisy.model.KNNCFRecommender import ItemKNNCF
from daisyRec.daisy.model.PureSVDRecommender import PureSVD
from daisyRec.daisy.model.Item2VecRecommender import Item2Vec
from daisyRec.daisy.model.LightGCNRecommender import LightGCN
from daisyRec.daisy.utils.config import init_seed, init_logger, init_config_tuning
from daisyRec.daisy.utils.metrics import MAP, NDCG, Recall, Precision, HR, MRR
from daisyRec.daisy.utils.sampler import BasicNegtiveSampler, SkipGramNegativeSampler
from daisyRec.daisy.utils.dataset import AEDataset, BasicDataset, CandidatesDataset, get_dataloader
from daisyRec.daisy.utils.utils import get_history_matrix, get_ur, build_candidates_set, ensure_dir, get_inter_matrix

model_config = {
    'mostpop': MostPop,
    'slim': SLiM,
    'itemknn': ItemKNNCF,
    'puresvd': PureSVD,
    'mf': MF,
    'fm': FM,
    'ngcf': NGCF,
    'neumf': NeuMF,
    'nfm': NFM,
    'multi-vae': VAECF,
    'item2vec': Item2Vec,
    'ease': EASE,
    'lightgcn': LightGCN,
}

metrics_config = {
    "recall": Recall,
    "mrr": MRR,
    "ndcg": NDCG,
    "hr": HR,
    "map": MAP,
    "precision": Precision,
}

tune_params_config = {
    'mostpop': [],
    'itemknn': ['maxk'],
    'puresvd': ['factors'],
    'slim': ['alpha', 'elastic'],
    'mf': ['num_ng', 'factors', 'lr', 'batch_size', 'reg_1', 'reg_2'],
    'fm': ['num_ng', 'factors', 'lr', 'batch_size', 'reg_1', 'reg_2'],
    'neumf': ['num_ng', 'factors', 'num_layers', 'dropout', 'lr', 'batch_size', 'reg_1', 'reg_2'],
    'nfm': ['num_ng', 'factors', 'num_layers', 'dropout', 'lr', 'batch_size', 'reg_1', 'reg_2'],
    'ngcf': ['num_ng', 'factors', 'node_dropout', 'mess_dropout', 'batch_size', 'lr', 'reg_1', 'reg_2'],
    'multi-vae': ['latent_dim', 'dropout','batch_size', 'lr', 'anneal_cap'],
    'ease': ['reg'],
    'item2vec': ['context_window', 'rho', 'lr', 'factors'],
    'lightgcn': ['num_ng', 'factors', 'batch_size', 'lr', 'reg_1', 'reg_2', 'num_layers'],
}

param_type_config = {
    'num_layers': 'int',
    'maxk': 'int',
    'factors': 'int',
    'alpha': 'float',
    'elastic': 'float',
    'num_ng': 'int',
    'lr': 'float',
    'batch_size': 'int',
    'reg_1': 'float',
    'reg_2': 'float',
    'dropout': 'float',
    'node_dropout': 'float',
    'mess_dropout': 'float',
    'latent_dim': 'int',
    'anneal_cap': 'float',
    'reg': 'float',
    'context_window': 'int',
    'rho': 'float'
}

TRIAL_CNT = 0

def convert_df(users, item_id):

    df = pd.DataFrame()
    df["user"] = users
    df["item"] = item_id
    df["rating"] = [1 for i in range(len(item_id))]
         
    df["user"] = df["user"].astype("int64")
    df["item"] = df["item"].astype("int64")
    df["rating"] = df["rating"].astype("int64")
    return df


def leave_one_out_split(df):
    df = df.copy()

    # Number of interactions per user
    df["_n_interactions"] = df.groupby("user")["item"].transform("size")

    # Position of each interaction for each user
    df["_position"] = df.groupby("user").cumcount()

    # Last interaction goes to validation
    # only for users having more than 2 interactions
    validation_mask = (
        (df["_n_interactions"] > 2) &
        (df["_position"] == df["_n_interactions"] - 1)
    )

    train = df[~validation_mask].copy()
    validation = df[validation_mask].copy()

    # Remove temporary columns
    train = train[["user", "item", "rating"]]
    validation = validation[["user", "item", "rating"]]

    # Reset index
    train.reset_index(drop=True, inplace=True)
    validation.reset_index(drop=True, inplace=True)

    return train, validation


def load_data(path):
    users = list()
    item_id = list()

    with open(path, "r") as f:
        for line in f:
            temp = line.strip()
            temp = temp.split()

            item_id.extend(temp[1:])
            users.extend([temp[0] for i in range(len(temp[1:]))])

    df = pd.DataFrame()
    df["user"] = users
    df["item"] = item_id
    df["rating"] = [1 for i in range(len(item_id))]

    df["user"] = df["user"].astype("int64")
    df["item"] = df["item"].astype("int64")
    df["rating"] = df["rating"].astype("int64")

    user_mapping = {
        user: idx
        for idx, user in enumerate(df["user"].unique())
    }

    item_mapping = {
        item: idx
        for idx, item in enumerate(df["item"].unique())
    }

    df["user"] = df["user"].map(user_mapping)
    df["item"] = df["item"].map(item_mapping)

    return df

if __name__ == '__main__':
    ''' summarize hyper-parameter part (basic yaml + args + model yaml) '''
    config = init_config_tuning()

    ''' init seed for reproducibility '''
    init_seed(config['seed'], config['reproducibility'])

    ''' init logger '''
    init_logger(config)
    logger = getLogger()
    logger.info(config)
    config['logger'] = logger

    ''' unpack hyperparameters to tune '''
    param_dict = config['tune_pack']
    algo_name = config['algo_name']

    kpi_name = config['optimization_metric']
    tune_param_names = tune_params_config[algo_name]

    ''' open logfile to record tuning process '''
    # begin tuning here
    tune_log_path = "./results/daisyrec/"+config["algo_name"]+"/"+config["dataset"]+"/"
    ensure_dir(tune_log_path)

    f = open(tune_log_path + f"best_params_{config['loss_type']}_{config['algo_name']}_{config['dataset']}.csv", 'w', encoding='utf-8')
    line = ','.join(tune_param_names) + f',{kpi_name}'
    f.write(line + '\n')
    f.flush()


    if config['algo_name'] == "ngcf":
            path = "ngcf_data/"+"/"+config["dataset"]
            print(path)

    if config['algo_name'] == "lightgcn":
            path = "lightgcn_data/"+"/"+config["dataset"]
            print(path)

    path =   Path(path)
    train_path = path / "train.txt"
    
    data = load_data(train_path)
    if config["cand_num"] == "full":
            config["cand_num"]  = len(  data["item"].unique() )

    config['user_num'] = len(  data["user"].unique()   )
    config['item_num'] = len(  data["item"].unique() )
    train, validation = leave_one_out_split(data)
    
    ''' define optimization target function '''
    def objective(trial):
        global TRIAL_CNT
        for param in tune_param_names:
            if param not in param_dict.keys(): continue
                
            if isinstance(param_dict[param], list):
                config[param] = trial.suggest_categorical(param, param_dict[param])
            elif isinstance(param_dict[param], dict):
                if param_type_config[param] == 'int':
                    step = param_dict[param]['step']
                    config[param] = trial.suggest_int(
                        param, param_dict[param]['min'], param_dict[param]['max'], 1 if step is None else step)
                elif param_type_config[param] == 'float':
                    config[param] = trial.suggest_float(
                        param, param_dict[param]['min'], param_dict[param]['max'], step=param_dict[param]['step'])
                else:
                    raise ValueError(f'Invalid parameter type for {param}...')
            else:
                raise ValueError(f'Invalid parameter settings for {param}, Current is {param_dict[param]}...')
        


        cnt, kpis = 1, []
        ''' get ground truth '''
        val_ur = get_ur(validation)
        train_ur = get_ur(train)
        config['train_ur'] = train_ur

        ''' build and train model '''
        if config['algo_name'].lower() in ['itemknn', 'puresvd', 'slim', 'mostpop', 'ease']:
            model = model_config[config['algo_name']](config)
            model.fit(train)
            
        elif config['algo_name'].lower() in ['multi-vae']:
            history_item_id, history_item_value, _  = get_history_matrix(train, config, row='user')
            config['history_item_id'], config['history_item_value'] = history_item_id, history_item_value
            model = model_config[config['algo_name']](config)
            train_dataset = AEDataset(train, yield_col=config['UID_NAME'])
            train_loader = get_dataloader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4)
            model.fit(train_loader)

        elif config['algo_name'].lower() in ['mf', 'fm', 'neumf', 'nfm', 'ngcf', 'lightgcn']:
                if config['algo_name'].lower() in ['lightgcn', 'ngcf']:
                    config['inter_matrix'] = get_inter_matrix(train, config)
                model = model_config[config['algo_name']](config)
                sampler = BasicNegtiveSampler(train, config)
                train_samples = sampler.sampling()
                train_dataset = BasicDataset(train_samples)
                train_loader = get_dataloader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4)
                model.fit(train_loader, config)

        elif config['algo_name'].lower() in ['item2vec']:
            model = model_config[config['algo_name']](config)
            sampler = SkipGramNegativeSampler(train, config)
            train_samples = sampler.sampling()
            train_dataset = BasicDataset(train_samples)
            train_loader = get_dataloader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4)
            model.fit(train_loader)
        else:
            raise NotImplementedError('Something went wrong when building and training...')
        logger.info(f'Finish {cnt} train-validation experiment(s)...')
        cnt += 1

        ''' build candidates set '''
        logger.info('Start Calculating Metrics...')
        val_u, val_ucands = build_candidates_set(val_ur, train_ur, config)

        ''' get predict result '''
        logger.info('==========================')
        logger.info('Generate recommend list...')
        logger.info('==========================')
        val_dataset = CandidatesDataset(val_ucands)
        val_loader = get_dataloader(val_dataset, batch_size=128, shuffle=False, num_workers=0)
        preds = model.rank(val_loader) 

        ''' calculating KPIs '''
        kpi = metrics_config[kpi_name](val_ur, preds, val_u)
        kpis.append(kpi)
        
        TRIAL_CNT += 1
        logger.info(f'Finish {TRIAL_CNT} trial...')
        return np.mean(kpis)

    ''' init optuna workspace '''
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=2022))
    study.optimize(objective, n_trials=config['hyperopt_trail'])

    ''' record the best choices '''
    logger.info(f'Trial {study.best_trial.number} get the best {kpi_name}({study.best_trial.value}) with params: {study.best_trial.params}')
    line = ','.join([str(study.best_params[param]) if param in param_dict.keys() else str(config[param]) for param in tune_param_names]) + f',{study.best_value:.4f}\n'
    f.write(line)
    f.flush()
    f.close()