import numpy as np
import pandas as pd
import os
import time


def save_tracks_error_handing(
        savepath, track_generator, n, is_big=False, n_log=100,
        tau_error=0.2, tau_time=10):
    if is_big:
        tw = HdfTrackWriter(savepath)
    else:
        tw = TrackWriter(savepath)

    bad_time = -1
    times_list = np.zeros((2*n,)) + bad_time
    start_time = time.time()
    for run_id, (tracks, status) in enumerate(track_generator(n)):
        if len(tracks) == 0 or status == StatusCode.bad:
            error_rate = (times_list[:run_id+1] == bad_time).sum()/(run_id + 1.)
            print('error rate: {:0.2f}'.format(error_rate))
            if error_rate > tau_error and run_id > 10:
                print('Exiting {}'.format(savepath))
                print('Bad error rate: {}'.format(error_rate))
                return StatusCode.bad_error_rate
            mean_time = -1 if len(tracks) == 0 else \
                times_list[times_list != bad_time].mean()
            if mean_time > tau_time:
                print('Exiting {}'.format(savepath))
                print('Bad mean time: {}s'.format(mean_time))
                return StatusCode.bad_time
            start_time = time.time()
            continue
        times_list[run_id] = time.time() - start_time
        for agent_id, track in enumerate(tracks):
            tw.add_track(track, agent_id, run_id)

        if (run_id + 1) % n_log == 0:
            print('{:6.0f} runs averaging {:7.3f}'.format(
                run_id + 1, times_list[times_list != bad_time].mean()))
        start_time = time.time()
    tw.save()
    return StatusCode.good


class StatusCode(object):

    good = 1
    bad = 0
    bad_error_rate = -1
    bad_time = -2


def save_tracks(savepath, track_generator, n, is_big=False, n_log=100):
    """
    Basic saving for multiple runs
    :param savepath: 
    :param track_generator: n -> per yield, list of (l, 2) in order of agent id
    :param n: number of runs
    :param is_big: == use HDF5
    :param n_log: == use HDF5
    :return: 
    """
    if is_big:
        tw = HdfTrackWriter(savepath)
    else:
        tw = TrackWriter(savepath)

    times_list = []
    start_time = time.time()
    for run_id, tracks in enumerate(track_generator(n)):
        times_list.append(time.time() - start_time)
        if len(tracks) == 0:
            start_time = time.time()
            continue
        for agent_id, track in enumerate(tracks):
            tw.add_track(track, agent_id, run_id)

        if len(times_list) % n_log == 0:
            print('{:6.0f} runs averaging {:7.3f}'.format(
                len(times_list), np.mean(times_list)))
        start_time = time.time()
    tw.save()


def get_matching_npz_path(savepath):
    ext = savepath[savepath.rfind('.'):]  # simple extensions
    return savepath.replace(ext, '.npz')


class TrackWriter(object):

    def __init__(self, savepath):
        self.savepath = savepath

        self.cols = ['x', 'y', 't', 'agent_id', 'run_id']
        self._int_cols = ['t', 'agent_id', 'run_id']
        self.track_df = pd.DataFrame(columns=self.cols)

    def add_track(self, track, agent_id, run_id, t_offset=0):
        n = track.shape[0]
        add_data_df = pd.DataFrame(np.hstack([
            track,
            t_offset + np.arange(n).reshape((-1, 1)),
            agent_id + np.zeros((n, 1), dtype=np.int),
            run_id + np.zeros((n, 1), dtype=np.int),
        ]), columns=self.cols)
        self.track_df = self.track_df.append(
            add_data_df, ignore_index=True)

    def save(self):
        self.track_df[self._int_cols] = self.track_df[self._int_cols].astype(np.int)
        self.track_df.to_csv(self.savepath, index=False, sep=' ')

    def load(self):
        self.track_df = pd.read_csv(
            self.savepath, sep=' ', header=0, names=self.cols)
        return self.track_df

    def remove(self):
        if os.path.isfile(self.savepath):
            os.remove(self.savepath)


class HdfTrackWriter(object):

    def __init__(self, savepath):
        self.savepath = savepath

        self.cols = ['x', 'y', 't', 'agent_id', 'run_id']
        self._int_cols = ['t', 'agent_id', 'run_id']
        self.track_df = pd.DataFrame(columns=self.cols)
        self.load_ind = 0

        self.df_name = 'tracks'
        self.hdf = None

        self.is_ready = False
        self._max_rows = 5000

    def add_track(self, track, agent_id, run_id, t_offset=0):
        self._ready()
        n = track.shape[0]
        add_data_df = pd.DataFrame(np.hstack([
            track,
            t_offset + np.arange(n).reshape((-1, 1)),
            agent_id + np.zeros((n, 1), dtype=np.int),
            run_id + np.zeros((n, 1), dtype=np.int),
        ]), columns=self.cols)
        self.track_df = self.track_df.append(
            add_data_df, ignore_index=True)

        if self.track_df.shape[0] > self._max_rows:
            self._put_data()

    def save(self):
        self._put_data()
        self.hdf.close()

    def _ready(self):
        if self.is_ready:
            return
        self.hdf = pd.HDFStore(self.savepath)
        self.hdf.put(key=self.df_name, value=self.track_df, format='t',  data_columns=True)
        self.is_ready = True

    def _put_data(self):
        self.track_df[self._int_cols] = self.track_df[self._int_cols].astype(np.int)
        self.hdf.put(key=self.df_name, value=self.track_df,
                     format='t', data_columns=True, append=True)
        self.track_df = pd.DataFrame(columns=self.cols)

    def load(self):
        with pd.HDFStore(self.savepath, mode='r') as hdf:
            print(hdf.keys())
            self.track_df = hdf.get(key=self.df_name)
        return self.track_df

    def remove(self):
        self.hdf.close()
        if os.path.isfile(self.savepath):
            os.remove(self.savepath)
