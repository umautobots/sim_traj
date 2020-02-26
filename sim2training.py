import numpy as np
import os
from glob import glob
from tqdm import tqdm
import pandas as pd
import utils as ut
import model as mo


def make_dataset_full(dfs):
    batches_df = pd.DataFrame(columns=['t', 'agent_id', 'x', 'y'])
    t_final = 0
    max_agent_id = 0
    with tqdm(total=len(dfs)) as bar:
        for df in dfs:
            run_ids = df['run_id'].unique()
            for run_id in run_ids:
                df_run = df[df['run_id'] == run_id].copy()
                t_final_run = df_run['t'].unique().size
                max_agent_id_run = df_run['agent_id'].unique().size
                df_run['t'] += t_final
                df_run['agent_id'] += max_agent_id
                t_final += t_final_run
                max_agent_id += max_agent_id_run
                batches_df = batches_df.append(
                    df_run[['t', 'agent_id', 'x', 'y']], ignore_index=True)
            bar.update(1)
    return batches_df


def get_split_names_by_folder(root, tr_split=0.8, seed=0, filter_fcn=None):
    filter_fcn = filter_fcn or (lambda x: True)
    np.random.seed(seed)
    hdf_names = [os.path.basename(hdf_name)
                 for hdf_name in get_data_paths(root)
                 if filter_fcn(hdf_name)]
    n = len(hdf_names)
    np.random.shuffle(hdf_names)

    split_ind = int(tr_split * n)
    tr_names = hdf_names[:split_ind]
    val_names = hdf_names[split_ind:]
    return tr_names, val_names


def load_dataframes(save_paths):
    from storage import TrackWriter, HdfTrackWriter
    dfs = []
    for save_path in save_paths:
        if '.csv' in save_path:
            tw = TrackWriter(save_path)
        elif '.h5' in save_path:
            tw = HdfTrackWriter(save_path)
        else:
            tw = HdfTrackWriter(save_path)
        df = tw.load()
        if not validate_dataframe(df):
            print('{}\n has NANs, discarding!'.format(save_path))
            continue
        dfs.append(df)
    return dfs


def validate_dataframe(df):
    nan_rows = df.iloc[np.where(df.isnull().any(axis=1))[0]]
    return nan_rows.shape[0] == 0


def get_data_paths(root, is_npz=False):
    # all matching pairs of full paths
    hdf_list = glob(os.path.join(root, '*.h5'))
    if is_npz:
        npz_list = glob(os.path.join(root, '*.npz'))
        hdf_npz_names = [(hdf, hdf.replace('.h5', '.npz')) for hdf in hdf_list
                         if hdf.replace('.h5', '.npz') in npz_list]
        return hdf_npz_names
    else:
        return hdf_list


def convert_sim2training():
    method_name = mo.ModelType(is_large=True, is_sd_s=False, is_sd_p=False).get_name()
    dataset_split = 'split_1.0_0'
    dataset_names = [
        'eth',
        'hotel',
        'zara',
        'univ',
    ]

    dataset2df = {}
    for dataset in dataset_names:
        tr_save_paths = [p for p in glob(os.path.join(ut.get_synth_dataset_folder_path(
            method_name, dataset, dataset_split), '*.h5'))]
        dataset2df[dataset] = load_dataframes(tr_save_paths)

    for dataset in dataset_names:
        for set_i, other_dataset in enumerate(dataset_names):
            if dataset == other_dataset:
                continue
            tr_dfs = dataset2df[other_dataset]
            df_tr = make_dataset_full(tr_dfs)
            df_val = make_dataset_full(tr_dfs[:1])
            df_val = df_val[df_val['t'] < 100]
            for df, split_name in zip([df_tr, df_val], ['train', 'val']):
                if split_name == 'val' and set_i > 1:
                    continue
                df_save_name = other_dataset + split_name + '.csv'
                output_path = os.path.join(
                    ut.TRAINING_ROOT, method_name, dataset_split,
                    dataset,
                    split_name,
                    df_save_name
                )
                print('Saving to {}'.format(output_path))
                ut.mkdir_p(output_path.replace(df_save_name, ''))
                df.to_csv(output_path, index=False, sep=' ', header=False)
    ut.mkdir_p(os.path.join(ut.MODELS_ROOT, method_name, dataset_split))


if __name__ == '__main__':

    convert_sim2training()
