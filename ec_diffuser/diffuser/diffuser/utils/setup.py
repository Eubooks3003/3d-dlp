import os
import importlib
import random
import numpy as np
import torch
from tap import Tap
import pdb

from .serialization import mkdir
from .git_utils import (
    get_git_rev,
    save_git_diff,
)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def watch(args_to_watch):
    def _fn(args):
        exp_name = []
        for key, label in args_to_watch:
            if not hasattr(args, key):
                continue
            val = getattr(args, key)
            if type(val) == dict:
                val = '_'.join(f'{k}-{v}' for k, v in val.items())
            exp_name.append(f'{label}{val}')
        exp_name = '_'.join(exp_name)
        exp_name = exp_name.replace('/_', '/')
        exp_name = exp_name.replace('(', '').replace(')', '')
        exp_name = exp_name.replace(', ', '-')
        return exp_name
    return _fn

def lazy_fstring(template, args):
    ## https://stackoverflow.com/a/53671539
    return eval(f"f'{template}'")

class Parser(Tap):

    def save(self):
        fullpath = os.path.join(self.savepath, 'args.json')
        print(f'[ utils/setup ] Saved args to {fullpath}')
        super().save(fullpath, skip_unpicklable=True)

    def parse_args(self, experiment=None, savepath=None):
        args = super().parse_args(known_only=True)
        mode = str(args.num_entity)+'C_'+args.input_type
        exp_name = f'{args.num_entity}C_{args.input_type}_' + args.exp_note
        if args.overfit:
            exp_name = 'overfit_' + exp_name
        if args.kitchen:
            exp_name = 'kitchen_' + exp_name
            mode = mode + '_kitchen'
        if args.push_t:
            exp_name = 'PushT_' + exp_name 
            mode = mode + '_pusht'
        if args.rand_color:
            exp_name = exp_name + '_randcolor'
            mode = mode + '_randcolor'
        if args.planning_only:
            mode = mode[3:]
            exp_name = 'plan_' + exp_name
        args.mode = mode
        args.exp_name = exp_name
        args.wandb_group_name = exp_name

        ### Important: read config file
        args = self.read_config(args, experiment)
        self.add_extras(args)
        self.eval_fstrings(args)
        self.set_seed(args)
        self.get_commit(args)
        self.set_loadbase(args)
        self.generate_exp_name(args)
        if savepath is not None:
            args.savepath = savepath
        else:
            self.mkdir(args)
        self.save_diff(args)
        return args

    def read_config(self, args, experiment):
        '''
            Load parameters from config file
        '''
        mode = args.mode
        exp_name = args.exp_name
        print(f'[ utils/setup ] Reading config: {args.config}:{mode}')
        module = importlib.import_module(args.config)
        params = getattr(module, 'base')[experiment]
        from dataset_paths import get_diffuser_dataset_path
        mode_to_args = getattr(module, 'mode_to_args')
        if experiment == 'diffusion':
            # for the diffusion experiment, we will overwrite the params for each key in mode_to_args
            if mode not in mode_to_args:
                raise KeyError(f"Mode '{mode}' not in {args.config}.mode_to_args. Available: {list(mode_to_args.keys())}")

            # If the config provides dataset_path (or override_dataset_path), DO NOT overwrite it
            override = mode_to_args[mode].get('override_dataset_path', None)
            if override is not None:
                mode_to_args[mode]['dataset_path'] = override
            elif mode_to_args[mode].get('dataset_path', None) is None:
                mode_to_args[mode]['dataset_path'] = get_diffuser_dataset_path(
                    args.dataset_loadbase,
                    args.num_entity,
                    args.input_type,
                    rand_color=args.rand_color,
                    push_t=args.push_t,
                    kitchen=args.kitchen,
                )

            params.update(mode_to_args[mode])

        elif experiment == 'plan':
            # for the planning experiment, we will filter out a subset of the keys from mode_to_args to overwrite the params
            mode_to_args_filter = {}
            for k in ['device', 'horizon', 'n_diffusion_steps','vis_freq', 'env_config_dir', 'policy', 'diffusion_loadpath', 'dataset', 'override_dataset_path']:
                if k in mode_to_args[mode].keys():
                    mode_to_args_filter[k] = mode_to_args[mode][k]

            ### Loading for planning
            if hasattr(module, 'entity_to_steps') or hasattr(module, 'PushT_entity_to_steps') or hasattr(module, 'kitchen_entity_to_steps'):
                if args.push_t:
                    mode_to_args_filter['max_episode_length'] = module.PushT_entity_to_steps[args.num_entity]
                elif args.kitchen:
                    mode_to_args_filter['max_episode_length'] = module.kitchen_entity_to_steps[args.num_entity]
                else:
                    mode_to_args_filter['max_episode_length'] = module.entity_to_steps[args.num_entity]
            else:
                mode_to_args_filter['max_episode_length'] = mode_to_args[mode]['max_path_length']
            
            params.update(mode_to_args_filter)
        
        # Append action type (absolute vs relative) to exp_name if specified
        use_absolute = params.get('use_absolute_actions', None)
        if use_absolute is True:
            exp_name = exp_name + '_absolute'
        elif use_absolute is False:
            exp_name = exp_name + '_relative'

        params['prefix'] = params['prefix'] + exp_name
        params['seed'] = args.seed

        self._dict = {}
        for key, val in params.items():
            setattr(args, key, val)
            self._dict[key] = val

        return args

    def add_extras(self, args):
        '''
            Override config parameters with command-line arguments
        '''
        extras = args.extra_args
        if not len(extras):
            return

        print(f'[ utils/setup ] Found extras: {extras}')
        assert len(extras) % 2 == 0, f'Found odd number ({len(extras)}) of extras: {extras}'
        for i in range(0, len(extras), 2):
            key = extras[i].replace('--', '')
            val = extras[i+1]
            assert hasattr(args, key), f'[ utils/setup ] {key} not found in config: {args.config}'
            old_val = getattr(args, key)
            old_type = type(old_val)
            print(f'[ utils/setup ] Overriding config | {key} : {old_val} --> {val}')
            if val == 'None':
                val = None
            elif val == 'latest':
                val = 'latest'
            elif old_type in [bool, type(None)]:
                try:
                    val = eval(val)
                except:
                    print(f'[ utils/setup ] Warning: could not parse {val} (old: {old_val}, {old_type}), using str')
            else:
                val = old_type(val)
            setattr(args, key, val)
            self._dict[key] = val

    def eval_fstrings(self, args):
        for key, old in self._dict.items():
            if type(old) is str and old[:2] == 'f:':
                val = old.replace('{', '{args.').replace('f:', '')
                new = lazy_fstring(val, args)
                print(f'[ utils/setup ] Lazy fstring | {key} : {old} --> {new}')
                setattr(self, key, new)
                self._dict[key] = new

    def set_seed(self, args):
        if not hasattr(args, 'seed') or args.seed is None or args.seed < 0:
            args.seed = random.randint(0, 1000000)
        print(f'[ utils/setup ] Setting seed: {args.seed}')
        set_seed(args.seed)

    def set_loadbase(self, args):
        if hasattr(args, 'loadbase') and args.loadbase is None:
            print(f'[ utils/setup ] Setting loadbase: {args.logbase}')
            args.loadbase = args.logbase

    def generate_exp_name(self, args):
        if not 'exp_name' in dir(args):
            return
        exp_name = getattr(args, 'exp_name')
        if callable(exp_name):
            exp_name_string = exp_name(args)
            print(f'[ utils/setup ] Setting exp_name to: {exp_name_string}')
            setattr(args, 'exp_name', exp_name_string)
            self._dict['exp_name'] = exp_name_string

    def mkdir(self, args):
        if 'logbase' in dir(args) and 'dataset' in dir(args) and 'exp_name' in dir(args):
            args.savepath = os.path.join(args.logbase, args.dataset, args.exp_name)
            self._dict['savepath'] = args.savepath
            if 'suffix' in dir(args):
                args.savepath = os.path.join(args.savepath, args.suffix)
            if mkdir(args.savepath):
                print(f'[ utils/setup ] Made savepath: {args.savepath}')
            self.save()

    def get_commit(self, args):
        args.commit = get_git_rev()

    def save_diff(self, args):
        try:
            save_git_diff(os.path.join(args.savepath, 'diff.txt'))
        except:
            print('[ utils/setup ] WARNING: did not save git diff')
