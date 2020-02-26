import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def display_xy_predictions_vs_gt(
        ax, batch_df, y_pred_df, n_obs, evaluate_ids,
        pause=0.5, max_step=8, title_str='', xlim=(), ylim=()):
    df = batch_df.copy()
    t_batch = df['t'].unique()
    t_batch_pred = t_batch[n_obs:]
    t_pred = y_pred_df['t'].unique()
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)
    for j, agent_id in enumerate(evaluate_ids):
        ego_xy = df2xy(y_pred_df, agent_id, t_pred[0])
        gt_xy = df2xy(df, agent_id, t_batch_pred[0])
        ogt_xy = df2xy(df, agent_id, t_batch_pred[0], is_all_others=True)
        ar_ego_xy, = ax.plot(
            ego_xy[:, 1], ego_xy[:, 0],
            marker='o', linestyle='', color='blue', alpha=0.5)
        ar_gts, = ax.plot(
            gt_xy[:, 1], gt_xy[:, 0],
            marker='>', linestyle='', color='black', alpha=0.5)
        ar_ogts, = ax.plot(
            ogt_xy[:, 1], ogt_xy[:, 0],
            marker='x', linestyle='', color='black', alpha=0.2)
        plt.pause(pause/2)
        for i in range(min(len(t_pred), max_step)):
            ego_xy = df2xy(y_pred_df, agent_id, t_pred[i])
            gt_xy = df2xy(df, agent_id, t_batch_pred[i])
            ogt_xy = df2xy(df, agent_id, t_batch_pred[i], is_all_others=True)
            ar_ego_xy.set_data(ego_xy[:, 1], ego_xy[:, 0])
            ar_gts.set_data(gt_xy[:, 1], gt_xy[:, 0])
            ar_ogts.set_data(ogt_xy[:, 1], ogt_xy[:, 0])
            if title_str:
                ax.set_title(title_str.format(i+1, j))
            plt.pause(pause)
        ar_ego_xy.remove()
        ar_gts.remove()
        ar_ogts.remove()


def df2xy(df, agent_id, t, is_all_others=False, is_p=False):
    cols = ['x', 'y']
    if is_p:
        cols += ['p']
    agents_mask = (df['agent_id'] != agent_id) if is_all_others else (df['agent_id'] == agent_id)
    xy = df[(df['t'] == t) & agents_mask][cols].values
    return xy
