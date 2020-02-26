import pandas as pd
import os

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
DATASETS_ROOT = os.path.join(PROJECT_ROOT, 'datasets')
SYNTH_DATASETS_ROOT = os.path.join(PROJECT_ROOT, 'synth_datasets')
TRAINING_ROOT = os.path.join(PROJECT_ROOT, 'training')
MODELS_ROOT = os.path.join(PROJECT_ROOT, 'models')
RESULTS_ROOT = os.path.join(PROJECT_ROOT, 'results')


def get_dataset_split_path(dataset, split, is_train):
    tag = 'train' if is_train else 'test'
    return os.path.join(DATASETS_ROOT, split, '{}_{}.csv'.format(dataset, tag))


def get_synth_dataset_folder_path(model, dataset, split):
    return os.path.join(SYNTH_DATASETS_ROOT, model, dataset, split)


def load_df(path):
    return pd.read_csv(path, names=['t', 'agent_id', 'x', 'y'], sep='\s+')


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as e:
        pass


def get_time(t):
    m, s = divmod(t, 60)
    h, m = divmod(m, 60)
    time_string = '\n{:02d}:{:02d}:{:02d}\n'.format(
        int(h), int(m), int(s)
    )
    return time_string
