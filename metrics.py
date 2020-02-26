import numpy as np


def evaluate_expected_distance_xy(pred_xyp, gt_xy):
    """
    :param pred_xyp: distribution over xy at a given time lag, in m
    assume sum p = 1
    :param gt_xy: coordinates in m
    :return:
    """
    if gt_xy.size == 0:
        return np.nan
    use_pred_xyp = pred_xyp[pred_xyp[:, 2] > 0, :]
    dist = np.sqrt(((use_pred_xyp[:, :2] - gt_xy)**2).sum(axis=1))
    weighted_dist = (dist * use_pred_xyp[:, 2]).sum()
    return weighted_dist


def evaluate_min_distance_xy(pred_xyp, gt_xy, p_threshold):
    """
    :param pred_xyp: distribution over xy at a given time lag, in m
    assume sum p = 1
    :param gt_xy: coordinates in m
    :param p_threshold: minimum probability prediction to be considered
    :return:
    """
    if gt_xy.size == 0:
        return np.nan
    use_pred_xyp = pred_xyp[pred_xyp[:, 2] >= p_threshold, :]
    dist = np.sqrt(((use_pred_xyp[:, :2] - gt_xy)**2).sum(axis=1))
    return dist.min()
