"""
A script to extract metadata from pickle files 
and save them sepratley per episode

Per-observation actions are stored in the actions directory in the episodes folder

""" 
from tqdm import tqdm
import glob
import torch
import os
import pickle
import sys
import types
from typing import Iterable

class MockClass:
    """Mock class that can be instantiated"""
    def __init__(self, *args, **kwargs):
        self._args = args
        self._kwargs = kwargs
        # Store any attributes set on the instance
        for k, v in kwargs.items():
            setattr(self, k, v)
    
    def __getattr__(self, name):
        # Return another mock for any attribute access
        return MockClass()
    
    def __call__(self, *args, **kwargs):
        # Allow the mock to be called
        return MockClass(*args, **kwargs)

class MockModule(types.ModuleType):
    """Mock module that returns mock classes"""
    def __getattr__(self, name):
        # Return a mock class that can be instantiated
        return MockClass

class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        try:
            return super().find_class(module, name)
        except ModuleNotFoundError:
            # Create mock module on the fly
            if module not in sys.modules:
                sys.modules[module] = MockModule(module)
            return getattr(sys.modules[module], name)

class CustomUnpickler2(pickle.Unpickler):
    def find_class(self, module, name):
        try:
            return super().find_class(module, name)
        except (ModuleNotFoundError, AttributeError):
            # Create a simple class that accepts any attributes
            if module.startswith('rlbench'):
                # Return a namedtuple-like class or simple object
                def __init__(self, *args, **kwargs):
                    self.__dict__.update(*args)
                    
                    if args:
                        # Handle positional args if any
                        for i, arg in enumerate(args):
                            setattr(self, f'_arg{i}', arg)
                    print(self.__dict__)
                
                cls = type(name, (object,), {'__init__': __init__})
                return cls
            raise

def to_json(raw_pkl_load):
    parsed_pkl = {}
    for k,vs in raw_pkl_load.__dict__.items():
        if isinstance(vs,list):
            vs = [v.__dict__ for v in vs]
            parsed_pkl[k] = vs
        else:
            parsed_pkl[k] = vs
    return parsed_pkl 
                

def refactor(data,episode_fp):
    """
    data should be a a json file

    see save_demo in https://github.com/stepjam/RLBench/blob/master/rlbench/dataset_generator.py
    """
    actions_fp = os.path.join(episode_fp,"actions")
    os.makedirs(actions_fp,exist_ok=True) 
    for i, obs in enumerate(data["_observations"]):
        obs_act_fp = os.path.join(actions_fp,f"{i}.pt")
        torch.save(obs,obs_act_fp)
    

def refactor_dataset(rootfp,splits,tasks):
    paths = []
    if len(splits) == 0 and len(tasks) == 0:
        allfps = os.path.join(rootfp,"*","*","all_variations","episodes","*")
        paths += glob.glob(allfps)
    elif len(splits) == 0:
        for task in tasks:
            allfps = os.path.join(rootfp,"*",task,"all_variations","episodes","*")
            print(allfps)
            paths += glob.glob(allfps)
    elif len(tasks) == 0:
        for split in splits:
            allfps = os.path.join(rootfp,split,"*","all_variations","episodes","*")
            paths += glob.glob(allfps)
    else: 
        for split in splits:
            for task in tasks:
                allfps = os.path.join(rootfp,split,task,"all_variations","episodes","*")
                print(f"{allfps=}")
                paths += glob.glob(allfps)

    variation_nums = {}
    variation_descriptions = {} # maps episode to description 
    for p in tqdm(paths):
        pkls = (
            # os.path.join(p,"low_dim_obs.pkl"),
                os.path.join(p,"variation_descriptions.pkl"),
                os.path.join(p,"variation_number.pkl")) 
        for pkl_path in pkls:
            with open(pkl_path, 'rb') as f:
                raw_data = CustomUnpickler2(f).load()
                print(f"{pkl_path=}",raw_data)
                # if 'variation_number' in pkl_path:
                #     variation_count = variation_nums.get(raw_data,0)
                #     variation_nums[raw_data] = variation_count+1
                # data = to_json(raw_data)
                # print(data)
                # refactor(data=data,episode_fp=p)
    print(variation_nums)
    print(sorted(list(variation_nums.keys())))
        

import argparse 
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--dataset_path",
        default =  "../data/rlbench"
    )
    parser.add_argument(
        "-s",
        "--splits",
        default = "",
        help = "if not specified, will do all"
    )
    parser.add_argument(
        "-t",
        "--tasks",
        default = "",
        help = "if not specified, will do all"
    )

    args = parser.parse_args()
    DATASET_ROOT =  args.dataset_path
    SPLIT = [split for split in args.splits.split(",") if split != '']
    TASKS = [task for task in args.tasks.split(",") if task != '']


    refactor_dataset(DATASET_ROOT,SPLIT,TASKS)

    
    
    

    # # print(os.listdir("../data/rlbench/train/open_drawer/all_variations/episodes/episode0"))
    # with open(d, 'rb') as f:
    #     raw_data = CustomUnpickler2(f).load()
    #     data = to_json(raw_data)
    #     refactor(data=data,root_fp="../data/rlbench/train/open_drawer/all_variations/episodes/episode0/")

    