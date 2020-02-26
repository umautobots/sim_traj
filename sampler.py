import model as mo
import utils as ut
import storage as st
import numpy as np
import time
import os


def visualize_state(ax, s, other_id2xy=(), artists=None):
    # plot all obj locations, current with more alpha
    # plot current state
    t = int(s[-1])
    other_id2xy = other_id2xy if len(other_id2xy) else {}
    other_xy = np.array([v for v in other_id2xy.values()])
    other_kwargs = dict(
        marker='o', color='black', alpha=0.5, linestyle=''
    )
    if artists:
        artists['ar1'].set_data(s[0], s[1])
        if len(other_xy) > 0:
            artists['ar2'].set_data(other_xy[:, 0], other_xy[:, 1])
        else:
            artists['ar2'].set_data([], [])

        for i, oxy in other_id2xy.items():
            text_str = '{:2.0f}'.format(i)
            artists[i].set_position(oxy + .2)
            artists[i].set_text(text_str)
    else:
        a1, = ax.plot(s[0], s[1], marker='>', color='black')
        if len(other_xy) > 0:
            a2, = ax.plot(other_xy[:, 0], other_xy[:, 1], **other_kwargs)
        else:
            a2, = ax.plot([], [], **other_kwargs)
        artists = {'ar1': a1, 'ar2': a2}
        for i, oxy in other_id2xy.items():
            text_str = '{:2.0f}'.format(i)
            artists[i] = ax.text(oxy[0], oxy[1], text_str, alpha=0.5, fontsize=8)

    return artists


def visualize_trajectory(ax, xy, other_xy_list=(), artists=None):
    artists = artists or []
    for t, xyi in enumerate(xy):
        s = np.hstack([xyi, t]) if len(xyi) < 3 else xyi
        # other_xy = np.array([
        #     other[t] for other in other_xy_list if t < len(other)])
        other_id2xy = {i: other[t] for i, other in enumerate(other_xy_list) if t < len(other)}
        artists = visualize_state(
            ax, s, other_id2xy=other_id2xy, artists=artists)
        info_str = 't={:3.0f}'.format(t)
        ax.set_title(info_str)
        yield None


def sample_batch(sampler, n_steps, dt):
    n_peds = sampler.sample_n_peds()
    ts = np.arange(0, (n_steps+1)*dt, dt)
    trajectories = []
    for i in range(n_peds):
        speed_i, path_i = sampler.sample_trajectory()
        # print(speed_i)
        trajectory_i = path_i(speed_i * ts)
        trajectories.append(trajectory_i)
    # print('---')
    return trajectories


def main_sample(is_large, is_sd_s, is_sd_p, dataset, split):
    n_steps = 20
    dt = 0.4
    seed = 0
    model = mo.ModelType(is_large, is_sd_s, is_sd_p)
    sampler = mo.build_sampler(model, dataset, split, dt)
    n_tracks = model.get_n_tracks(n_steps, dataset, split)
    save_path = os.path.join(ut.SYNTH_DATASETS_ROOT, model.get_name(), dataset, split,
                             'seed={}_n_tracks={}.h5'.format(seed, n_tracks))
    print(os.path.dirname(save_path))
    ut.mkdir_p(os.path.dirname(save_path))

    # import matplotlib  # for mac
    # matplotlib.use('TkAgg')  # for mac
    # import matplotlib.pyplot as plt
    # plt.ion()
    # for i in range(0, 100):
    #     trajectories = sample_batch(sampler, n_steps, dt)
    #
    #     for traj in trajectories:
    #         p = traj
    #         dif = p[1:, :] - p[:-1, :]
    #         dif = np.sqrt(np.sum(dif ** 2, axis=1))
    #         print(np.mean(dif) / 0.4)
    #
    #     fig, ax = plt.subplots()
    #     ax.set_xlim([-20, 20])
    #     ax.set_ylim([-20, 20])
    #     ax.grid()
    #     plt.show()
    #     for _ in visualize_trajectory(ax, trajectories[0], trajectories[1:]):
    #         plt.pause(0.6)
    #     plt.close()


    def trajectory_xy_generator(n):
        ind = 0
        while ind < n:
            trajectories = sample_batch(sampler, n_steps, dt)
            yield trajectories, st.StatusCode.good
            ind += 1

    print('saving {}'.format(save_path))
    t0 = time.time()
    status = st.save_tracks_error_handing(
        save_path, trajectory_xy_generator, n_tracks, is_big=True, n_log=100,
        tau_error=0.3, tau_time=10.
    )
    print('Finishing with status: ', status)
    print(ut.get_time(time.time() - t0))
    print('saved to {}'.format(save_path))


def main():
    is_large = True
    is_sd_s = False
    is_sd_p = False
    dataset = 'zara'  # eth hotel univ zara
    split = 'split_1.0_0'
    main_sample(is_large, is_sd_s, is_sd_p, dataset, split)


if __name__ == '__main__':
    main()
