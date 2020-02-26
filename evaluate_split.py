import numpy as np
from functools import partial
import pandas as pd
import time
import os
import utils as ut
import model as mo
import adapters as ad


class Predictor(object):

    def __init__(self, save_name, n_obs, n_preds, batch_df2input_fcn, predict,
                 raw_pred2pred, validate_batch=None):
        self.save_name = save_name
        self.n_obs = n_obs
        self.n_preds = n_preds
        self.batch_df2input_fcn = batch_df2input_fcn
        self.predict = predict
        self.raw_pred2pred = raw_pred2pred
        self.validate_batch = validate_batch


def get_sgan_predictor(method_name, test_dataset, dataset_split, make_pred_fcn=None):
    """
    
    :param method_name: 
    :param test_dataset: 
    :param dataset_split: 
    :param make_pred_fcn: (n_samples, model_path) -> pred_fcn
    - pred_fcn: (abs_xy, rel_xy, seq_start_end) -> sampled prediction list
    - prediction list[i] = n_preds, n_peds, 2 | list of sampled predictions
    :return: 
    """
    model_path = os.path.join(ut.MODELS_ROOT, '{}/{}/{}.pt'.format(method_name, dataset_split, test_dataset))
    print(model_path)
    n_samples = 100
    n_obs = 8
    n_preds = 8
    batch_df2input_fcn = partial(ad.batch_df2batch, n_obs=n_obs)
    predict_fcn = make_pred_fcn(n_samples, model_path)
    raw_pred2pred = partial(ad.raw_pred2df, tform=np.eye(3))
    save_name = 'SGAN_{}_{}'.format(n_samples, test_dataset)
    predictor = Predictor(save_name, n_obs, n_preds,
                          batch_df2input_fcn, predict_fcn, raw_pred2pred)
    return predictor


def main():
    methods = [
        (mo.ModelType(is_large=True, is_sd_s=True, is_sd_p=True).get_name(), get_sgan_predictor),
    ]
    dataset_names = [
        # 'eth',
        # 'hotel',
        # 'zara',
        'univ',
    ]
    dataset_split = 'split_1.0_0'
    # dataset_split = 'split_0.2_2'
    is_evaluate_cross = False
    is_skip_if_exists = False
    is_display = False
    is_test_on_all = True

    dataset2gt_name = {
        'eth': 'batches_eth_test',
        'hotel': 'batches_hotel_test',
        'zara': 'batches_zara_test',
        'univ': 'batches_univ_test',
    }
    for method_name, get_predictor in methods:
        for train_dataset in dataset_names:
            for test_dataset in dataset_names:
                if train_dataset != test_dataset and not is_evaluate_cross:
                    continue
                try:
                    predictor = get_predictor(
                        method_name, test_dataset, dataset_split)
                except FileNotFoundError as e:
                    print(e)
                    continue
                print('Evaluating: {} {} {}'.format(
                    method_name, train_dataset, test_dataset, dataset_split))
                save_path = os.path.join(
                    ut.RESULTS_ROOT,
                    method_name,
                    test_dataset,
                    dataset_split,
                    predictor.save_name + '.csv'
                )
                print('saving to {}'.format(save_path))
                if is_skip_if_exists and os.path.exists(save_path):
                    print(': file exists -> skipping')
                    continue
                results_df = evaluate(predictor, test_dataset, dataset_split,
                                      dataset2gt_name,
                                      is_display=is_display, is_test_on_all=is_test_on_all)

                print(save_path)
                ut.mkdir_p(os.path.dirname(save_path))
                results_df.to_csv(save_path, index=False, sep=' ')


def evaluate(predictor, test_dataset, dataset_split, dataset2gt_name,
             is_display=False, is_test_on_all=False):
    from metrics import evaluate_expected_distance_xy, evaluate_min_distance_xy

    gt_name = dataset2gt_name[test_dataset]
    if is_test_on_all:
        batches_df_path = os.path.join(ut.DATASETS_ROOT, 'split_1.0_0', 'batches', gt_name + '.csv')
    else:
        batches_df_path = os.path.join(ut.DATASETS_ROOT, dataset_split, 'batches', gt_name + '.csv')
    n_obs = predictor.n_obs
    n_preds = predictor.n_preds
    scale = 1.

    # --
    if is_display:
        from display import display_xy_predictions_vs_gt
        import matplotlib
        matplotlib.use('TkAgg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax = fig.gca()
        plt.grid(True)
    t2pred_xyp = {i: [] for i in range(n_preds)}
    t2gt_xy_list = {i: [] for i in range(n_preds)}
    is_gt_linear_list = []
    cdf_bin_edges = np.arange(0, 20., .05)
    t2cdf_bin_counts = {i: np.zeros((cdf_bin_edges.size-1,), dtype=np.int) for i in range(n_preds)}
    distance_quantiles = np.arange(0.01, 1., .01)
    t2distance_quantile_values = {i: np.zeros((distance_quantiles.size,), dtype=np.float) for i in range(n_preds)}
    batches_df = pd.read_csv(batches_df_path, sep=' ', header=0)  # dc.load_dataframe(batches_df_path)
    batches_df = batches_df[batches_df['agent_type'] != 1]  # no vehicles
    batches_inds = np.unique(np.sort(batches_df['batch_ind']))
    skip = get_skip(batches_df[batches_df['batch_ind'] == batches_inds[0]])
    max_obs = 8
    time_elapsed = 0
    n_peds_evaluated = 0
    for batch_ind in batches_inds:
        batch_df = batches_df[batches_df['batch_ind'] == batch_ind]
        t = int(batch_df['t'].min())
        t_obs0 = t+(max_obs-n_obs)*skip
        t_pred0 = t+max_obs*skip
        t_end = t+(max_obs+n_preds)*skip
        # evaluated agents must be present in every frame
        obs_df = batch_df[batch_df['t'].isin(range(t_obs0, t_pred0))]
        evaluate_ids, evaluate_inds = get_eval_agents(batch_df, obs_df, 16, t_end)
        if len(evaluate_ids) == 0:
            # print(batch_ind)
            continue

        predictor_input = predictor.batch_df2input_fcn(obs_df, evaluate_ids)
        start_time = time.time()
        y_raw_pred = predictor.predict(*predictor_input)
        time_elapsed += time.time() - start_time
        y_pred_df = predictor.raw_pred2pred(y_raw_pred, evaluate_ids, evaluate_inds)

        print(batch_ind, len(evaluate_ids))
        if is_display:
            print(batch_ind)
            title_str = 'batch {}: '.format(batch_ind) + 't = {}, agent = {}'
            display_xy_predictions_vs_gt(
                ax, batch_df, y_pred_df, max_obs, evaluate_ids,
                pause=0.5, max_step=8, title_str=title_str,
                xlim=[-10, 20], ylim=[-10, 20])

        n_peds_evaluated += len(evaluate_ids)
        for j in range(len(evaluate_ids)):
            for i in range(n_preds):
                t_ind = t_pred0 + i*skip if 'Simulator' in predictor.save_name else i
                t2pred_xyp[i].append(
                    y_pred_df[(y_pred_df['t'] == t_ind) &
                              (y_pred_df['agent_id'] == evaluate_ids[j])][['x', 'y', 'p']].values
                )
                gt_t_df = batch_df[(batch_df['t'] == t_pred0 + i*skip) &
                                   (batch_df['agent_id'] == evaluate_ids[j])]
                # assert gt_t_df.shape[0] > 0
                t2gt_xy_list[i].append(gt_t_df[['x', 'y']].values)
                # distance cdf
                distances = np.sqrt(((t2pred_xyp[i][-1][:, :2] - t2gt_xy_list[i][-1])**2).sum(axis=1))
                t2cdf_bin_counts[i] += np.histogram(distances, bins=cdf_bin_edges)[0]
                assert distances.max() < cdf_bin_edges[-1]
                # distance quantiles
                t2distance_quantile_values[i] += np.percentile(distances, distance_quantiles*100, interpolation='lower')

            obs_df_j = batch_df[(batch_df['t'].isin(range(t_obs0, t_pred0))) & (batch_df['agent_id'] == evaluate_ids[j])]
            gt_df_j = batch_df[(batch_df['t'].isin(range(t_pred0, t_end))) & (batch_df['agent_id'] == evaluate_ids[j])]
            is_gt_linear_list.append(is_trajectory_linear(obs_df_j, gt_df_j, n_preds))
    # wdist(t) = ADE(t)
    eval_metrics = {name: [] for name in [
        'wdist', 'Min_dist_0.01', 'wdist_linear', 'wdist_nonlinear',
        'Min_dist_0.01_linear', 'Min_dist_0.01_nonlinear', 'dist_cdf',
        'dist_quantile'
    ]}
    for i in range(n_preds):
        print('\nTime t={} stats'.format(i+1))
        wdist = np.array([evaluate_expected_distance_xy(pred_xy, gt_xy)
                          for pred_xy, gt_xy
                          in zip(t2pred_xyp[i], t2gt_xy_list[i])])
        wdist = wdist[~np.isnan(wdist)]/scale
        print('wdist {:.2f}'.format(np.mean(wdist)))
        eval_metrics['wdist'].append(np.mean(wdist))

        wdist = np.array([evaluate_min_distance_xy(
            pred_xy, gt_xy, 0)
            for pred_xy, gt_xy
            in zip(t2pred_xyp[i], t2gt_xy_list[i])])
        wdist = wdist[~np.isnan(wdist)]/scale
        print('min dist_0.01 {:.2f}'.format(np.mean(wdist)))
        eval_metrics['Min_dist_0.01'].append(np.mean(wdist))

        # avg-linear/nonlinear
        wdist = np.array([evaluate_expected_distance_xy(pred_xy, gt_xy)
                          for pred_xy, gt_xy, is_linear
                          in zip(t2pred_xyp[i], t2gt_xy_list[i], is_gt_linear_list) if is_linear])
        wdist = wdist[~np.isnan(wdist)] / scale
        print('wdist_linear {:.2f}'.format(np.mean(wdist)))
        eval_metrics['wdist_linear'].append(np.mean(wdist))
        wdist = np.array([evaluate_expected_distance_xy(pred_xy, gt_xy)
                          for pred_xy, gt_xy, is_linear
                          in zip(t2pred_xyp[i], t2gt_xy_list[i], is_gt_linear_list) if ~is_linear])
        wdist = wdist[~np.isnan(wdist)] / scale
        print('wdist_nonlinear {:.2f}'.format(np.mean(wdist)))
        eval_metrics['wdist_nonlinear'].append(np.mean(wdist))

        # min-linear/nonlinear
        wdist = np.array([evaluate_min_distance_xy(
            pred_xy, gt_xy, 0)
            for pred_xy, gt_xy, is_linear
            in zip(t2pred_xyp[i], t2gt_xy_list[i], is_gt_linear_list) if is_linear])
        wdist = wdist[~np.isnan(wdist)] / scale
        print('min dist_0.01_linear {:.2f}'.format(np.mean(wdist)))
        eval_metrics['Min_dist_0.01_linear'].append(np.mean(wdist))
        wdist = np.array([evaluate_min_distance_xy(
            pred_xy, gt_xy, 0)
            for pred_xy, gt_xy, is_linear
            in zip(t2pred_xyp[i], t2gt_xy_list[i], is_gt_linear_list) if ~is_linear])
        wdist = wdist[~np.isnan(wdist)] / scale
        print('min dist_0.01_nonlinear {:.2f}'.format(np.mean(wdist)))
        eval_metrics['Min_dist_0.01_nonlinear'].append(np.mean(wdist))

        # distance cdf
        eval_metrics['dist_cdf'].append(np.array2string(t2cdf_bin_counts[i], threshold=10000000))
        print(len(np.array2string(t2cdf_bin_counts[i])), 'dist cdf length')

        # dist_quantile plot
        t2distance_quantile_values[i] /= float(n_peds_evaluated)
        eval_metrics['dist_quantile'].append(np.array2string(t2distance_quantile_values[i], threshold=10000000))

    n_steps_predicted = len(batches_inds) * n_preds
    average_prediction_time = time_elapsed / n_steps_predicted
    print('Average prediction time {:.8f}s'.format(average_prediction_time))
    eval_metrics['Time'] = [average_prediction_time] * len(eval_metrics[list(eval_metrics.keys())[0]])

    eval_df = pd.DataFrame.from_dict(eval_metrics)
    pd.set_option('display.max_colwidth', -1)
    return eval_df


def get_skip(batch_df):
    """

    :param batch_df: 
    :return: skip: 1 = no skip, 2 = every other frame skipped
    :rtype int
    """
    t_inds = np.unique(np.sort(batch_df['t'].values))
    return int(t_inds[1] - t_inds[0])


def get_eval_agents(df, obs_df, n_frames, t_max):
    # only eval on agents present in all n_frames of observations+pred
    # - for indexing, based on obs_df
    agent_ids = np.sort(obs_df['agent_id'].unique())
    evaluate_ids = []
    evaluate_inds = []
    for i, agent_id in enumerate(agent_ids):
        n_present = df[(df['agent_id'] == agent_id) &
                       (df['t'] < t_max)].shape[0]
        if n_frames == n_present:
            evaluate_ids.append(agent_id)
            evaluate_inds.append(i)
    return evaluate_ids, evaluate_inds


def is_trajectory_linear(obs_df, gt_df, n_preds):
    xy = obs_df[['x', 'y']].values
    xp = np.mean(xy[1:, 0] - xy[:-1, 0]) * np.arange(1, n_preds + 1) + xy[-1, 0]
    yp = np.mean(xy[1:, 1] - xy[:-1, 1]) * np.arange(1, n_preds + 1) + xy[-1, 1]
    pred_xy = np.array([xp, yp]).T
    diff = (pred_xy - gt_df[['x', 'y']].values)**2
    rmse = np.sqrt(np.sum(diff)/n_preds)  # in meters
    return rmse < 0.5


if __name__ == '__main__':
    main()
