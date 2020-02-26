import numpy as np
import pandas as pd


def batch_df2batch(df, evaluate_ids=(), n_obs=-1, tform=np.eye(3), is_vehicles_evaluated=False):
    """
    Convert dataframe to SGAN input
    :param df: 
    :param evaluate_ids: 
    :param n_obs: number of timesteps observed
    :param tform: 
    :param is_vehicles_evaluated: 
    :return: 
    """
    if is_vehicles_evaluated:
        agent_ids = np.unique(df['agent_id'])
    else:
        agent_ids = np.unique(df[df['agent_type'] == 0]['agent_id'])  # peds only

    # input transform
    df = tform_df(df, tform)

    # assume min t is the start
    t_inds = np.unique(np.sort(df['t']))
    t0 = t_inds[0]
    skip = t_inds[1] - t_inds[0]

    abs_xy = np.zeros((n_obs, agent_ids.size, 2), dtype=np.float32)
    rel_xy = np.zeros_like(abs_xy)
    for i, agent_id in enumerate(agent_ids):
        for step, t in enumerate(range(t0, t0+n_obs*skip, skip)):
            xy = df[(df['agent_id'] == agent_id) & (df['t'] == t)][['x', 'y']]
            if xy.size > 0:
                abs_xy[step, i, :] = xy.values[0]
            else:
                abs_xy[step, i, :] = np.nan
        # for relative, 1st entry is 0,0, rest are the differences
        rel_xy[1:, i, :] = abs_xy[1:, i, :] - abs_xy[:-1, i, :]
    # handle observations w/zeros
    abs_xy[np.isnan(abs_xy)] = 0.
    rel_xy[np.isnan(rel_xy)] = 0.
    seq_start_end = [(0, agent_ids.size)]
    return abs_xy, rel_xy, seq_start_end


def raw_pred2df(pred_list, evaluate_ids, evaluate_inds, tform=np.eye(3)):
    """
    
    :param pred_list: [i] = n_preds, n_peds, 2 | list of sampled predictions
     - n_preds = number of timesteps predicted into future
    :param evaluate_ids: list of agent ids
    :param evaluate_inds: [i] = index of agent_id=evaluate_ids[i] in prediction  
    :param tform:  (3,3) | transformation matrix
    :return: 
    """
    merged_peds = np.stack(pred_list, axis=-1)  # (n_preds, n_peds, 2, n_samples)
    n_preds = merged_peds.shape[0]
    n_samples = merged_peds.shape[3]
    cols = ['t', 'agent_id', 'x', 'y', 'sample_id', 'p']
    INT_COLUMNS = [cols[i] for i in [0, 1, -2]]
    data = []
    for ind, id in zip(evaluate_inds, evaluate_ids):
        for t in range(n_preds):
            z = np.zeros((n_samples, 1))
            agent_t_info = np.hstack([
                t + z,
                id + z,
                merged_peds[t, ind, :, :].T,
                np.arange(n_samples).reshape((n_samples, 1)),
                1./n_samples + z,
            ])
            data.append(agent_t_info)
    df = pd.DataFrame(np.vstack(data), columns=cols)
    df[['x', 'y']] = tform_2d_mat(df[['x', 'y']].values, tform)
    df[INT_COLUMNS] = df[INT_COLUMNS].astype(np.int)
    return df


def tform_df(df, tform):
    xy = df[['x', 'y']]
    ret_df = df.copy()
    ret_df[['x', 'y']] = tform_2d_mat(xy, tform)
    return ret_df


def tform_2d_mat(xy, tform):
    xy1 = np.hstack([xy, np.ones((xy.shape[0], 1))])
    xy1_p = (tform.dot(xy1.T)).T
    return xy1_p[:, :2]
