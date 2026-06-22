import numpy as np
import pickle


def _install_numpy_pickle_shim():
    import sys
    import numpy.core as _core
    sys.modules["numpy._core"] = _core
    sys.modules["numpy._core.multiarray"] = _core.multiarray


def atleast_2d(x):
    while x.ndim < 2:
        x = np.expand_dims(x, axis=-1)
    return x

class ReplayBuffer:

    def __init__(self, max_n_episodes, max_path_length, termination_penalty):
        self._dict = {
            'path_lengths': np.zeros(max_n_episodes, dtype=int),
        }
        self._count = 0
        self.max_n_episodes = max_n_episodes
        self.max_path_length = max_path_length
        self.termination_penalty = termination_penalty

    def __repr__(self):
        def _shape(val):
            if hasattr(val, 'shape'):
                return val.shape
            try:
                return f'list(len={len(val)})'
            except Exception:
                return type(val).__name__
        return '[ datasets/buffer ] Fields:\n' + '\n'.join(
            f'    {key}: {_shape(val)}'
            for key, val in self.items()
        )

    def __getitem__(self, key):
        return self._dict[key]

    def __setitem__(self, key, val):
        self._dict[key] = val
        self._add_attributes()

    @property
    def n_episodes(self):
        return self._count

    @property
    def n_steps(self):
        return sum(self['path_lengths'])

    def _add_keys(self, path):
        if hasattr(self, 'keys'):
            return
        self.keys = list(path.keys())

    def _add_attributes(self):
        '''
            can access fields with `buffer.observations`
            instead of `buffer['observations']`
        '''
        for key, val in self._dict.items():
            setattr(self, key, val)

    def items(self):
        return {k: v for k, v in self._dict.items()
                if k not in ('path_lengths', 'meta')}.items()


    def _allocate(self, key, array):
        assert key not in self._dict
        dim = array.shape[-1]
        shape = (self.max_n_episodes, self.max_path_length, dim)
        self._dict[key] = np.zeros(shape, dtype=np.float32)
        # print(f'[ utils/mujoco ] Allocated {key} with size {shape}')

    def add_path(self, path):
        path_length = len(path['observations'])
        assert path_length <= self.max_path_length

        ## if first path added, set keys based on contents
        self._add_keys(path)

        ## add tracked keys in path
        for key in self.keys:
            array = atleast_2d(path[key])
            if key not in self._dict: self._allocate(key, array)
            self._dict[key][self._count, :path_length] = array

        ## penalize early termination
        if path['terminals'].any() and self.termination_penalty is not None:
            assert not path['timeouts'].any(), 'Penalized a timeout episode for early termination'
            self._dict['rewards'][self._count, path_length - 1] += self.termination_penalty

        ## record path length
        self._dict['path_lengths'][self._count] = path_length

        ## increment path counter
        self._count += 1

    def truncate_path(self, path_ind, step):
        old = self._dict['path_lengths'][path_ind]
        new = min(step, old)
        self._dict['path_lengths'][path_ind] = new

    def finalize(self):
        ## remove extra slots
        for key in self.keys + ['path_lengths']:
            self._dict[key] = self._dict[key][-self._count:]
        self._add_attributes()
        print(f'[ datasets/buffer ] Finalized replay buffer | {self._count} episodes')
        if 'info_goals_reached' in self._dict:
            self.successful_episode_idxes = np.where(self._dict['info_goals_reached'] == 1)[0]
        else:
            self.successful_episode_idxes = np.array([])
        print(f'[ datasets/buffer ] Found {len(self.successful_episode_idxes)} successful episodes')

    def load_paths_from_pickles(self, paths, single_view=False):
        '''Multi-task variant: load a *list* of per-task pkls, pad each to the
        global max along the time and keypose axes, and concatenate along the
        episode axis. Output fields look like a single big pkl.

        Padding rules:
          - Per-step fields (E, T, ...): pad T-axis with zeros to global max_T.
            path_lengths and n_keyposes track real lengths so padded steps are
            ignored downstream.
          - keypose_indices (E, max_kp): pad max_kp-axis with -1 sentinel
            (existing convention from preprocess) to global max_kp.
          - language (list[list[str]] of len E): concatenate lists.
          - per-episode scalars (E,): concatenate.
        '''
        _install_numpy_pickle_shim()

        loaded = []
        for path in paths:
            with open(path, "rb") as f:
                loaded.append((path, pickle.load(f)))

        # Compute global maxes across all tasks.
        global_max_T = 0
        global_max_kp = 0
        for _, d in loaded:
            if 'observations' in d:
                global_max_T = max(global_max_T, int(d['observations'].shape[1]))
            if 'keypose_indices' in d:
                global_max_kp = max(global_max_kp, int(d['keypose_indices'].shape[1]))

        # Per-key accumulator across tasks.
        accumulated = {}
        keys_seen = []

        for path, paths_dict in loaded:
            meta = paths_dict.get('meta', None)
            if isinstance(meta, dict):
                K_expected = meta.get('K', None) or meta.get('num_entity', None)
            else:
                K_expected = None

            for key, val in paths_dict.items():
                if key == 'meta':
                    # Keep the first task's meta; downstream uses dlp_cfg etc. from config not meta.
                    if 'meta' not in self._dict:
                        self._dict['meta'] = val
                    continue

                if key not in keys_seen:
                    keys_seen.append(key)

                # Single-view slicing identical to single-pkl loader.
                if single_view and (key == 'observations' or key == 'goals'):
                    if isinstance(val, np.ndarray) and val.ndim == 4:
                        E, T, K, D = val.shape
                        if K_expected is not None:
                            if K == 2 * K_expected:
                                val = val[:, :, :K_expected, :]
                            elif K == K_expected:
                                pass
                            else:
                                raise ValueError(
                                    f"[buffer] unexpected particle dim K={K}, expected {K_expected} or {2*K_expected}"
                                )

                # Pad along T or max_kp depending on key.
                val_padded = self._pad_for_concat(key, val, global_max_T, global_max_kp)
                accumulated.setdefault(key, []).append(val_padded)

        # Concatenate per key.
        for key, vals in accumulated.items():
            if key == 'language':
                merged = []
                for v in vals:
                    merged.extend(list(v))
                self._dict[key] = merged
            elif key == 'path_lengths':
                self._dict[key] = np.concatenate(vals, axis=0).astype(np.int32)
            elif key == 'variation_number':
                self._dict[key] = np.concatenate(vals, axis=0).astype(np.int32)
            elif key in ('keypose_indices', 'n_keyposes'):
                self._dict[key] = np.concatenate(vals, axis=0).astype(np.int32)
            else:
                self._dict[key] = np.concatenate(vals, axis=0).astype(np.float32)

        self._count = self._dict['observations'].shape[0]

        if 'path_lengths' not in self._dict:
            self._dict['path_lengths'] = np.array([len(obs) for obs in self._dict['observations']])

        self.keys = [k for k in keys_seen if k != 'meta']

        if 'gripper_state' in self._dict:
            gs = self._dict['gripper_state']
            print(f'[ datasets/buffer ] Found gripper_state: shape={gs.shape}, '
                  f'range=[{gs.min():.3f}, {gs.max():.3f}]')

        if 'bg_features' in self._dict:
            bg = self._dict['bg_features']
            print(f'[ datasets/buffer ] Found bg_features: shape={bg.shape}, '
                  f'range=[{bg.min():.3f}, {bg.max():.3f}]')

        print(f'[ datasets/buffer ] Loaded multi-task: {self._count} episodes from '
              f'{len(paths)} pkls; global max_T={global_max_T}, max_kp={global_max_kp}')

    @staticmethod
    def _pad_for_concat(key, val, global_max_T, global_max_kp):
        '''Pad an array along the relevant axis so per-task tensors line up
        before episode-axis concat in load_paths_from_pickles.'''
        if key == 'language':
            return list(val)
        # Per-episode scalars (no per-axis padding).
        if key in ('path_lengths', 'variation_number', 'n_keyposes',
                   'info_goals_reached', 'info_goal_success_frac'):
            return val
        # keypose_indices: shape (E, max_kp_per_task) -- pad max_kp to global, sentinel -1.
        if key == 'keypose_indices':
            E, K = val.shape
            if K < global_max_kp:
                pad = np.full((E, global_max_kp - K), -1, dtype=val.dtype)
                val = np.concatenate([val, pad], axis=1)
            return val
        # Per-step fields (E, T_per_task, ...): pad T-axis with zeros.
        if isinstance(val, np.ndarray) and val.ndim >= 2:
            E, T = val.shape[:2]
            if T < global_max_T:
                pad_shape = (E, global_max_T - T) + val.shape[2:]
                pad = np.zeros(pad_shape, dtype=val.dtype)
                val = np.concatenate([val, pad], axis=1)
            return val
        return val

    def load_paths_from_pickle(self, path, single_view=False):
        _install_numpy_pickle_shim()
        with open(path, "rb") as f:
            paths_dict = pickle.load(f)

        # allow meta dicts etc
        meta = paths_dict.get('meta', None)
        if isinstance(meta, dict):
            K_expected = meta.get('K', None) or meta.get('num_entity', None)
        else:
            K_expected = None

        for key, val in paths_dict.items():
            # keep meta as-is
            if key == 'meta':
                self._dict[key] = val
                continue

            # Single-view handling ONLY if data is actually packed as 2K
            if single_view and (key == 'observations' or key == 'goals'):
                if isinstance(val, np.ndarray) and val.ndim == 4:
                    # val: (E,T,K,D)
                    E, T, K, D = val.shape

                    # If we know expected K, only slice when K == 2*K_expected
                    if K_expected is not None:
                        if K == 2 * K_expected:
                            val = val[:, :, :K_expected, :]
                        elif K == K_expected:
                            pass
                        else:
                            raise ValueError(
                                f"[buffer] unexpected particle dim K={K}, expected {K_expected} or {2*K_expected}"
                            )
                    else:
                        # If we DON'T know expected K, do NOT slice.
                        # (Your dataset is already single-view K=64.)
                        pass

            if key == 'path_lengths':
                self._dict[key] = val.astype(np.int32)
            elif key == 'language':
                # list[list[str]] per episode -- keep as Python object, don't cast
                self._dict[key] = list(val)
            elif key == 'variation_number':
                self._dict[key] = np.asarray(val, dtype=np.int32)
            elif key in ('keypose_indices', 'n_keyposes'):
                # Keypose-aware preprocess fields. Keep as int32; -1 sentinel for padding.
                self._dict[key] = np.asarray(val, dtype=np.int32)
            else:
                self._dict[key] = val.astype(np.float32)

        self._count = self._dict['observations'].shape[0]

        if 'path_lengths' not in paths_dict:
            self._dict['path_lengths'] = np.array([len(obs) for obs in self._dict['observations']])

        self.keys = [k for k in paths_dict.keys() if k != 'meta']

        # Report gripper state if present
        if 'gripper_state' in self._dict:
            gs = self._dict['gripper_state']
            print(f'[ datasets/buffer ] Found gripper_state: shape={gs.shape}, '
                  f'range=[{gs.min():.3f}, {gs.max():.3f}]')

        # Report bg_features if present
        if 'bg_features' in self._dict:
            bg = self._dict['bg_features']
            print(f'[ datasets/buffer ] Found bg_features: shape={bg.shape}, '
                  f'range=[{bg.min():.3f}, {bg.max():.3f}]')

        print(f'[ datasets/buffer ] Loaded {self._count} episodes from {path}')
