import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
import utils as ut


class ModelType(object):

    def __init__(self, is_large=True, is_sd_s=True, is_sd_p=True):
        self.is_large = is_large
        self.is_sd_s = is_sd_s
        self.is_sd_p = is_sd_p

    def get_name(self):
        amount = 'large' if self.is_large else 'eq'
        name = 'synth_{}'.format(amount)
        if not self.is_sd_s:
            name += '_no_sd_s'
        if not self.is_sd_p:
            name += '_no_sd_p'
        return name

    def get_n_tracks(self, n_steps, dataset, split):
        if self.is_large:
            n_tracks = 500 if dataset not in ['univ'] else 100
        else:
            df = ut.load_df(ut.get_dataset_split_path(dataset, split, is_train=True))
            n_frames = df['t'].unique().size
            n_tracks = n_frames // (n_steps + 1)  # n_steps -> n_steps+1 positions
        return n_tracks


class Sampler(object):

    def __init__(self, traj_list, r=0., mu_peds=-1., sd_peds=-1., sd_speed=-1., dt=0.4, **kwargs):
        self.traj_list = traj_list
        self.r = r
        self.mu_peds = mu_peds
        self.sd_peds = sd_peds
        self.sd_speed = sd_speed
        self.dt = dt

    def sample_n_peds(self):
        return sample_truncated(self.mu_peds, self.sd_peds, 0, np.inf)

    def sample_trajectory(self):
        traj = perturb_trajectory(self.traj_list, self.r)
        mu_speed = get_mu_speed(traj, self.dt)
        speed = get_random_speed(mu_speed, self.sd_speed)
        if traj.shape[0] > 2:
            heading_dif = traj[-1] - traj[-2]
            heading = np.arctan2(heading_dif[1], heading_dif[0])
            dist = get_dist(traj)
            fitted_traj = extend_path(
                traj, heading, n=int(np.ceil((40 - dist)/(mu_speed*self.dt))),
                dist=mu_speed*self.dt)
        else:
            fitted_traj = traj  # just two points
        path = fit_spline(fitted_traj)
        return speed, path


class SplineTxy(object):

    def __init__(self, x_spl, y_spl):
        self.x_spl = x_spl
        self.y_spl = y_spl

    def __call__(self, t, **kwargs):
        return np.array([self.x_spl(t), self.y_spl(t)]).T

    def derivative(self, t):
        return np.array([self.x_spl.derivatives(t)[-1],
                         self.y_spl.derivatives(t)[-1]])


def build_sampler(model, dataset, split, dt):
    params = dict(sd_peds=0, sd_speed=0)
    traj_list = load_trajectories(ut.load_df(ut.get_dataset_split_path(dataset, split, is_train=True)))
    df = ut.load_df(ut.get_dataset_split_path(dataset, 'split_1.0_0', is_train=True))
    if model.is_sd_s:
        params['sd_speed'] = calculate_sd_speed(df, dt)
    mu_peds, sd_peds = calculate_mu_sd_n_peds(df)
    params['mu_peds'] = mu_peds
    if model.is_sd_p:
        params['sd_peds'] = sd_peds
    r = 4.0
    sampler = Sampler(traj_list, r=r, dt=dt, **params)
    return sampler


def load_trajectories(df, tau_steps=2, dt=0.4):
    traj_list = []
    agent_ids = df['agent_id'].unique()
    for agent_id in agent_ids:
        # assume no gaps, sorted
        traj = df[df['agent_id'] == agent_id][['x', 'y']].values
        n = traj.shape[0]
        if n < tau_steps or len({tuple(row) for row in traj}) < n:
            continue
        dist = get_dist(traj)
        if dist < 25:
            heading_dif = traj[-1] - traj[-2]
            heading = np.arctan2(heading_dif[1], heading_dif[0])
            dif = traj[1:, :] - traj[:-1, :]
            dif = np.sqrt(np.sum(dif ** 2, axis=1))
            mu_speed = np.mean(dif) / dt
            traj = extend_path(traj, heading, n=int(np.ceil((25 - dist)/(mu_speed*dt))), dist=mu_speed*dt)
            traj_list.append(traj)
        else:
            traj_list.append(traj)
    return traj_list


def calculate_mu_sd_n_peds(df):
    groups = df.groupby('t', as_index=False)['agent_id'].count()
    cts = groups['agent_id']
    mu = cts.mean()
    sd = np.sqrt(cts.var())
    return mu, sd


def calculate_sd_speed(df, dt):
    difs = []
    agent_ids = df['agent_id'].unique()
    for agent_id in agent_ids:
        dfi = df[df['agent_id'] == agent_id]
        if dfi.shape[0] < 2:
            continue
        dx = dfi['x'].values[1:] - dfi['x'].values[:-1]
        dy = dfi['y'].values[1:] - dfi['y'].values[:-1]
        s = np.sqrt(dx**2 + dy**2) / dt
        bar_s = s.sum()/s.size
        difs.append(s - bar_s)
    difs = np.hstack(difs)
    sd_s_difs = np.sqrt((difs ** 2).sum() / (difs.size - agent_ids.size))
    return sd_s_difs


def fit_spline(path_xy):
    dists = np.hstack([0, np.cumsum(np.linalg.norm(path_xy[1:, :] - path_xy[:-1, :], axis=1))])
    k = 1
    x_spl = InterpolatedUnivariateSpline(dists, path_xy[:, 0], k=k)
    y_spl = InterpolatedUnivariateSpline(dists, path_xy[:, 1], k=k)
    return SplineTxy(x_spl, y_spl)


def perturb_trajectory(traj_list, r):
    # is reversed
    step = 1 if np.random.rand() > 0.5 else -1

    # shift in xy0
    shift = (np.random.rand(1, 2) * 2 - 1) * r

    ind = np.random.choice(len(traj_list))
    traj = traj_list[ind][::step, :] + shift

    start_ind = 0 if traj.shape[0] <= 20 else np.random.choice(traj.shape[0] - 20)
    traj = traj[start_ind:, :]
    return traj.copy()


def sample_truncated(mu, sd, low, high):
    while True:
        noise = np.random.randn()*sd
        if low < int(round(mu + noise)) < high:
            return int(round(mu + noise))
        elif low < int(round(mu - noise)) < high:
            return int(round(mu - noise))


def get_mu_speed(traj, dt):
    dif = traj[1:, :] - traj[:-1, :]
    dif = np.sqrt(np.sum(dif**2, axis=1))
    return np.mean(dif) / dt


def get_random_speed(mu, sd):
    shock = np.random.randn() * sd
    speed = shock + mu
    if speed < 0:
        speed = mu - shock
    return speed


def get_dist(path_xy):
    return np.cumsum(np.linalg.norm(path_xy[1:, :] - path_xy[:-1, :], axis=1))[-1]


def extend_path(path_xy, heading, n=15, dist=1.0):
    # extend with a straight line
    extended_xy = path_xy[-1] + np.array([
        np.cos(heading) * np.arange(1, n) * dist,
        np.sin(heading) * np.arange(1, n) * dist,
    ]).T
    return np.vstack([path_xy, extended_xy])

