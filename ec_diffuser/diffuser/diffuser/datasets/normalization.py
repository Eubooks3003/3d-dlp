import numpy as np
import scipy.interpolate as interpolate
import pdb

POINTMASS_KEYS = ['observations', 'actions', 'next_observations', 'deltas']

#-----------------------------------------------------------------------------#
#--------------------------- multi-field normalizer --------------------------#
#-----------------------------------------------------------------------------#

class DatasetNormalizer:

    def __init__(self, dataset, normalizer, particle_normalizer=None, path_lengths=None,
                 gripper_normalizer=None):
        self.observation_dim = dataset['observations'].shape[-1]
        self.action_dim = dataset['actions'].shape[-1]
        self.gripper_dim = dataset['gripper_state'].shape[-1] if 'gripper_state' in dataset._dict else 0

        if type(normalizer) == str:
            normalizer = eval(normalizer)
        if particle_normalizer is not None and type(particle_normalizer) == str:
            particle_normalizer = eval(particle_normalizer)
            goal_X = None  # goals no longer used for normalization
        if gripper_normalizer is not None and type(gripper_normalizer) == str:
            gripper_normalizer = eval(gripper_normalizer)

        dataset = flatten(dataset, path_lengths)
        self.normalizers = {}
        for key, val in dataset.items():
            try:
                if key == 'observations' and particle_normalizer is not None:
                    ### concatenate goals to observations for normalizing
                    if goal_X is not None:
                        obs_goal = np.concatenate([val, goal_X], axis=0)
                    else:
                        obs_goal = val
                    self.normalizers[key] = particle_normalizer(obs_goal)
                elif key == 'goals' and particle_normalizer is not None:
                    continue
                elif key == 'gripper_state':
                    # Use gripper_normalizer if provided, else use standard normalizer
                    if gripper_normalizer is not None:
                        self.normalizers[key] = gripper_normalizer(val)
                    else:
                        self.normalizers[key] = GaussianNormalizer(val)
                    print(f'[ utils/normalization ] Gripper state normalizer: {self.normalizers[key]}')
                else:
                    self.normalizers[key] = normalizer(val)
            except Exception as e:
                print(e)
                print(f'[ utils/normalization ] Skipping {key} | {normalizer}')

    def __repr__(self):
        string = ''
        for key, normalizer in self.normalizers.items():
            string += f'{key}: {normalizer}]\n'
        return string

    def __call__(self, *args, **kwargs):
        return self.normalize(*args, **kwargs)

    def normalize(self, x, key):
        return self.normalizers[key].normalize(x)

    def unnormalize(self, x, key):
        return self.normalizers[key].unnormalize(x)

    def get_field_normalizers(self):
        return self.normalizers

    def sanity_check_roundtrip(self, dataset, n_samples=3, save_path=None):
        """
        Verify that normalize -> unnormalize recovers the original data.
        Prints diagnostics and optionally saves to a file.

        Args:
            dataset: dict with 'actions', 'observations', etc.
            n_samples: number of samples to check
            save_path: if provided, saves detailed report to this file
        """
        print("\n" + "=" * 70)
        print("NORMALIZER ROUND-TRIP SANITY CHECK")
        print("=" * 70)

        report_lines = []
        all_passed = True

        for key in ['actions', 'observations']:
            if key not in self.normalizers:
                continue
            if key not in dataset:
                continue

            data = dataset[key]
            normalizer = self.normalizers[key]

            # Flatten if needed to get individual samples
            if data.ndim == 3:  # (N, T, D) -> take first n_samples trajectories, first timestep
                samples = data[:n_samples, 0, :]
            elif data.ndim == 4:  # (N, T, K, D) particles -> take first n_samples, first timestep
                samples = data[:n_samples, 0, :, :]
                samples = samples.reshape(n_samples, -1)  # flatten particles for comparison
            else:
                samples = data[:n_samples]

            # Round-trip: original -> normalized -> unnormalized
            normalized = normalizer.normalize(samples)
            recovered = normalizer.unnormalize(normalized)

            # Compute error
            abs_error = np.abs(samples - recovered)
            max_error = abs_error.max()
            mean_error = abs_error.mean()

            passed = max_error < 1e-5
            all_passed = all_passed and passed
            status = "PASS" if passed else "FAIL"

            header = f"\n[{key.upper()}] Round-trip check: {status}"
            print(header)
            report_lines.append(header)

            # Show normalizer info
            norm_info = f"  Normalizer type: {type(normalizer).__name__}"
            print(norm_info)
            report_lines.append(norm_info)

            if hasattr(normalizer, 'means') and hasattr(normalizer, 'stds'):
                stats = f"  means: {np.round(normalizer.means[:7], 4)}..."
                print(stats)
                report_lines.append(stats)
                stats = f"  stds:  {np.round(normalizer.stds[:7], 4)}..."
                print(stats)
                report_lines.append(stats)
                if hasattr(normalizer, 'z') and normalizer.z != 1.0:
                    z_info = f"  z (temperature): {normalizer.z}"
                    print(z_info)
                    report_lines.append(z_info)
            elif hasattr(normalizer, 'mins') and hasattr(normalizer, 'maxs'):
                stats = f"  mins: {np.round(normalizer.mins[:7], 4)}..."
                print(stats)
                report_lines.append(stats)
                stats = f"  maxs: {np.round(normalizer.maxs[:7], 4)}..."
                print(stats)
                report_lines.append(stats)

            error_info = f"  Max error: {max_error:.2e}, Mean error: {mean_error:.2e}"
            print(error_info)
            report_lines.append(error_info)

            # Show sample data for first sample
            for i in range(min(2, samples.shape[0])):
                sample_header = f"\n  Sample {i}:"
                print(sample_header)
                report_lines.append(sample_header)

                # Show first 7 dimensions (typical action dim)
                orig_str = f"    Original:     [{', '.join([f'{x:+.4f}' for x in samples[i, :7]])}]"
                norm_str = f"    Normalized:   [{', '.join([f'{x:+.4f}' for x in normalized[i, :7]])}]"
                recv_str = f"    Recovered:    [{', '.join([f'{x:+.4f}' for x in recovered[i, :7]])}]"
                diff_str = f"    Diff (|o-r|): [{', '.join([f'{x:.2e}' for x in abs_error[i, :7]])}]"

                print(orig_str)
                print(norm_str)
                print(recv_str)
                print(diff_str)
                report_lines.extend([orig_str, norm_str, recv_str, diff_str])

        summary = f"\n{'=' * 70}\nOVERALL: {'ALL CHECKS PASSED' if all_passed else 'SOME CHECKS FAILED'}\n{'=' * 70}\n"
        print(summary)
        report_lines.append(summary)

        if save_path:
            with open(save_path, 'w') as f:
                f.write('\n'.join(report_lines))
            print(f"Saved detailed report to: {save_path}")

        return all_passed

def flatten(dataset, path_lengths):
    """
    Flatten dataset fields by concatenating across episodes, respecting path_lengths.

    - observations: (N,T,K,D) -> (sumT, K, D)   (keep particle structure)
    - goals:        (N,T,K,D) -> (sumT, K, D)   (optional; often skipped later)
    - actions:      (N,T,A)   -> (sumT, A)
    - rewards/etc:  (N,T,1)   -> (sumT, 1)
    Skips non-array fields like meta dicts and info_* keys.
    """
    import numpy as np
    import torch

    path_lengths = np.asarray(path_lengths).astype(int)
    N = len(path_lengths)

    flattened = {}
    for key, xs in dataset.items():
        # skip non-array fields
        if xs is None or isinstance(xs, dict):
            continue
        if 'info' in key:
            continue

        if torch.is_tensor(xs):
            xs = xs.detach().cpu().numpy()
        if not isinstance(xs, np.ndarray):
            continue
        if xs.ndim < 2:
            continue
        if xs.shape[0] != N:
            continue

        parts = []
        for i, L in enumerate(path_lengths):
            parts.append(xs[i, :L])

        cat = np.concatenate(parts, axis=0)  # concatenate along time across episodes

        # IMPORTANT: do NOT flatten particle dims for observations/goals
        # Keep cat as (sumT, K, D) for particle normalizer.
        # For non-particle fields, cat will already be (sumT, A) or (sumT, 1)
        flattened[key] = cat

    return flattened


#-----------------------------------------------------------------------------#
#-------------------------- single-field normalizers -------------------------#
#-----------------------------------------------------------------------------#

class Normalizer:
    '''
        parent class, subclass by defining the `normalize` and `unnormalize` methods
    '''

    def __init__(self, X):
        self.X = X.astype(np.float32)
        self.mins = X.min(axis=0)
        self.maxs = X.max(axis=0)

    def __repr__(self):
        return (
            f'''[ Normalizer ] dim: {self.mins.size}\n    -: '''
            f'''{np.round(self.mins, 2)}\n    +: {np.round(self.maxs, 2)}\n'''
        )

    def __call__(self, x):
        return self.normalize(x)

    def normalize(self, *args, **kwargs):
        raise NotImplementedError()

    def unnormalize(self, *args, **kwargs):
        raise NotImplementedError()


class DebugNormalizer(Normalizer):
    '''
        identity function
    '''

    def normalize(self, x, *args, **kwargs):
        return x

    def unnormalize(self, x, *args, **kwargs):
        return x


class GaussianNormalizer(Normalizer):
    '''
        normalizes to zero mean and unit variance, with optional temperature scaling.

        z (temperature) controls the effective range:
          - z=1.0: standard Gaussian normalization (default)
          - z<1.0: widens the action range (model output ±1 maps to larger values)
          - z>1.0: narrows the action range

        normalize:   (x - mean) / (z * std)
        unnormalize: x * (z * std) + mean
    '''

    def __init__(self, *args, z=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.means = self.X.mean(axis=0)
        self.stds = self.X.std(axis=0)
        self.z = z  # temperature scaling factor

    def __repr__(self):
        return (
            f'''[ GaussianNormalizer ] dim: {self.mins.size}\n    '''
            f'''means: {np.round(self.means, 4)}\n    '''
            f'''stds:  {np.round(self.stds, 4)}\n    '''
            f'''z (temp): {self.z}\n    '''
            f'''effective_stds (z*std): {np.round(self.z * self.stds, 4)}\n'''
        )

    def normalize(self, x):
        return (x - self.means) / (self.z * self.stds + 1e-6)

    def unnormalize(self, x):
        return x * (self.z * self.stds) + self.means


class LimitsNormalizer(Normalizer):
    '''
        maps [ xmin, xmax ] to [ -1, 1 ]
    '''

    def normalize(self, x):
        ## [ 0, 1 ]
        x = (x - self.mins) / (self.maxs - self.mins)
        ## [ -1, 1 ]
        x = 2 * x - 1
        return x

    def unnormalize(self, x, eps=1e-4):
        '''
            x : [ -1, 1 ]
        '''
        if x.max() > 1 + eps or x.min() < -1 - eps:
            # print(f'[ datasets/mujoco ] Warning: sample out of range | ({x.min():.4f}, {x.max():.4f})')
            x = np.clip(x, -1, 1)

        ## [ -1, 1 ] --> [ 0, 1 ]
        x = (x + 1) / 2.

        return x * (self.maxs - self.mins) + self.mins

class SafeLimitsNormalizer(LimitsNormalizer):
    '''
        functions like LimitsNormalizer, but can handle data for which a dimension is constant
    '''

    def __init__(self, *args, eps=1, **kwargs):
        super().__init__(*args, **kwargs)
        for i in range(len(self.mins)):
            if self.mins[i] == self.maxs[i]:
                print(f'''
                    [ utils/normalization ] Constant data in dimension {i} | '''
                    f'''max = min = {self.maxs[i]}'''
                )
                self.mins -= eps
                self.maxs += eps

#-----------------------------------------------------------------------------#
#-------------------------- Particle normalizers -----------------------------#
#-----------------------------------------------------------------------------#

class ParticleNormalizer:
    '''
        parent class, subclass by defining the `normalize` and `unnormalize` methods
    '''

    def __init__(self, X):
        assert len(X.shape) == 3
        self.x_dim = X.shape[-1]
        self.X = X.astype(np.float32)
        self.mins = X.reshape(-1, X.shape[-1]).min(axis=0)
        self.maxs = X.reshape(-1, X.shape[-1]).max(axis=0)

    def __repr__(self):
        return (
            f'''[ Normalizer ] dim: {self.mins.size}\n    -: '''
            f'''{np.round(self.mins, 2)}\n    +: {np.round(self.maxs, 2)}\n'''
        )

    def __call__(self, x):
        return self.normalize(x)

    def normalize(self, *args, **kwargs):
        raise NotImplementedError()

    def unnormalize(self, *args, **kwargs):
        raise NotImplementedError()

class ParticleGaussianNormalizer(ParticleNormalizer):
    '''
        normalizes to zero mean and unit variance, input is N x number of particles x dim
    '''

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.means = self.X.reshape(-1, self.X.shape[-1]).mean(axis=0)
        self.stds = self.X.reshape(-1, self.X.shape[-1]).std(axis=0)
        self.z = 1

    def __repr__(self):
        return (
            f'''[ Normalizer ] dim: {self.mins.size}\n    '''
            f'''means: {np.round(self.means, 2)}\n    '''
            f'''stds: {np.round(self.z * self.stds, 2)}\n'''
        )

    def normalize(self, x):
        x_unflat = x.reshape(-1, self.x_dim)
        ret_unflat = (x_unflat - self.means) / (self.stds + 1e-6)
        return ret_unflat.reshape(x.shape)

    def unnormalize(self, x):
        x_unflat = x.reshape(-1, self.x_dim)
        ret_unflat = x_unflat * self.stds + self.means
        return ret_unflat.reshape(x.shape)

class ParticleLimitsNormalizer(ParticleNormalizer):
    '''
        normalizes to zero mean and unit variance, input is N x number of particles x dim
    '''

    def __init__(self, X, eps=1):
        super().__init__(X)
        for i in range(len(self.mins)):
            if self.mins[i] == self.maxs[i]:
                print(f'''
                    [ utils/normalization ] Constant data in dimension {i} | '''
                    f'''max = min = {self.maxs[i]}'''
                )
                self.mins[i] -= eps
                self.maxs[i] += eps

    def normalize(self, x):
        x_unflat = x.reshape(-1, self.x_dim)
        ret_unflat = (x_unflat - self.mins) / (self.maxs - self.mins) ## [0, 1]
        ret_unflat = 2 * ret_unflat - 1 # [-1, 1]
        return ret_unflat.reshape(x.shape)

    def unnormalize(self, x, eps=1e-4):
        x_unflat = x.reshape(-1, self.x_dim)
        if x_unflat.max() > 1 + eps or x_unflat.min() < -1 - eps:
            print(f'[ datasets/dlp ] Warning: sample out of range | ({x_unflat.min():.4f}, {x_unflat.max():.4f})')
            x_unflat = np.clip(x_unflat, -1, 1)

        ## [ -1, 1 ] --> [ 0, 1 ]
        x_unflat = (x_unflat + 1) / 2.
        ret_unflat = x_unflat * (self.maxs - self.mins) + self.mins
        return ret_unflat.reshape(x.shape)

class ParticleDoNothingNormalizer(ParticleNormalizer):
    '''
        normalizes to zero mean and unit variance, input is N x number of particles x dim
    '''

    def __init__(self, X, eps=1):
        super().__init__(X)

    def normalize(self, x):
        return x

    def unnormalize(self, x, eps=1e-4):
        return x
#-----------------------------------------------------------------------------#
#------------------------------- CDF normalizer ------------------------------#
#-----------------------------------------------------------------------------#

class CDFNormalizer(Normalizer):
    '''
        makes training data uniform (over each dimension) by transforming it with marginal CDFs
    '''

    def __init__(self, X):
        super().__init__(atleast_2d(X))
        self.dim = self.X.shape[1]
        self.cdfs = [
            CDFNormalizer1d(self.X[:, i])
            for i in range(self.dim)
        ]

    def __repr__(self):
        return f'[ CDFNormalizer ] dim: {self.mins.size}\n' + '    |    '.join(
            f'{i:3d}: {cdf}' for i, cdf in enumerate(self.cdfs)
        )

    def wrap(self, fn_name, x):
        shape = x.shape
        ## reshape to 2d
        x = x.reshape(-1, self.dim)
        out = np.zeros_like(x)
        for i, cdf in enumerate(self.cdfs):
            fn = getattr(cdf, fn_name)
            out[:, i] = fn(x[:, i])
        return out.reshape(shape)

    def normalize(self, x):
        return self.wrap('normalize', x)

    def unnormalize(self, x):
        return self.wrap('unnormalize', x)

class CDFNormalizer1d:
    '''
        CDF normalizer for a single dimension
    '''

    def __init__(self, X):
        assert X.ndim == 1
        self.X = X.astype(np.float32)
        quantiles, cumprob = empirical_cdf(self.X)
        self.fn = interpolate.interp1d(quantiles, cumprob)
        self.inv = interpolate.interp1d(cumprob, quantiles)

        self.xmin, self.xmax = quantiles.min(), quantiles.max()
        self.ymin, self.ymax = cumprob.min(), cumprob.max()

    def __repr__(self):
        return (
            f'[{np.round(self.xmin, 2):.4f}, {np.round(self.xmax, 2):.4f}'
        )

    def normalize(self, x):
        x = np.clip(x, self.xmin, self.xmax)
        ## [ 0, 1 ]
        y = self.fn(x)
        ## [ -1, 1 ]
        y = 2 * y - 1
        return y

    def unnormalize(self, x, eps=1e-4):
        '''
            X : [ -1, 1 ]
        '''
        ## [ -1, 1 ] --> [ 0, 1 ]
        x = (x + 1) / 2.

        if (x < self.ymin - eps).any() or (x > self.ymax + eps).any():
            print(
                f'''[ dataset/normalization ] Warning: out of range in unnormalize: '''
                f'''[{x.min()}, {x.max()}] | '''
                f'''x : [{self.xmin}, {self.xmax}] | '''
                f'''y: [{self.ymin}, {self.ymax}]'''
            )

        x = np.clip(x, self.ymin, self.ymax)

        y = self.inv(x)
        return y

def empirical_cdf(sample):
    ## https://stackoverflow.com/a/33346366

    # find the unique values and their corresponding counts
    quantiles, counts = np.unique(sample, return_counts=True)

    # take the cumulative sum of the counts and divide by the sample size to
    # get the cumulative probabilities between 0 and 1
    cumprob = np.cumsum(counts).astype(np.double) / sample.size

    return quantiles, cumprob

def atleast_2d(x):
    if x.ndim < 2:
        x = x[:,None]
    return x

