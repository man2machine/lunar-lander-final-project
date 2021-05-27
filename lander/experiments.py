# -*- coding: utf-8 -*-
"""
Created on Wed May 26 05:51:43 2021

@author: Shahir
"""

import json

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from lander.lunar_lander import LunarLander

SEED_MAX = 100000

class LunarLanderRunner:
    def __init__(self, env, policy, max_iters=400):
        self.env = env
        self.policy = policy
        self.max_iters = max_iters
        self.data = {
            'inside_helipad': None,
            'land_upright': None,
            'land_slowly': None,
            'touched_ground': None,
            'out_of_bounds': None,
            'num_steps': None,
            'end_state': None,
            'total_reward': None,
            'total_fuel': None,
            'ilqr_final_cost': None,
            'sim_states': None,
            'sim_actions': None,
        }
    
    def run(self, seed, render=False):
        self.env.seed(seed)
        obs = self.env.reset()
        num_steps = 0
        total_reward = 0
        total_fuel = 0
        xs = []
        us = []
        for i in range(self.max_iters):
            action = self.policy.predict(obs)
            xs.append(self.env.get_state())
            us.append(action)
            obs, reward, done, info = self.env.step(action)
            if render:
                self.env.render()
            num_steps += 1
            total_reward += reward
            total_fuel += info['fuel']
            if done:
                break
        
        self.data['inside_helipad'] = bool(self.env.detect_inside_helipad())
        self.data['land_upright'] = bool(self.env.detect_land_upright())
        self.data['land_slowly'] = bool(self.env.detect_land_slowly())
        self.data['touched_ground'] = bool(self.env.detect_collision())
        self.data['out_of_bounds'] = bool(self.env.detect_out_of_bounds())
        self.data['num_steps'] = int(num_steps)
        self.data['end_state'] = self.env.get_state().tolist()
        self.data['total_reward'] = int(total_reward)
        self.data['total_fuel'] = float(total_fuel)
        self.data['sim_states'] = np.array(xs).astype(float).tolist()
        self.data['sim_actions'] = np.array(us).astype(float).tolist()
        
        self.env.close()
    
    def update_ilqr_final_cost(self, cost):
        self.data['ilqr_final_cost'] = float(cost)

    def get_metrics(self):
        return self.data

def calc_stats(runs=None, metrics=None):
    metrics_np = {
        'inside_helipad': None,
        'land_upright': None,
        'land_slowly': None,
        'touched_ground': None,
        'out_of_bounds': None,
        'num_steps': None,
        'end_state': None,
        'total_reward': None,
        'total_fuel': None,
        'ilqr_final_cost': None,
        'sim_states': None,
        'sim_actions': None
    }
    
    if metrics is None:
        metrics = [r.get_metrics() for r in runs]
    
    for m in metrics:
        for name, val in m.items():
            if val is not None:
                if metrics_np[name] is None:
                    metrics_np[name] = [val]
                else:
                    metrics_np[name].append(val)
    
    ignore = ['sim_states', 'sim_actions']
    stats = {}
    for name in metrics_np:
        if name not in ignore and metrics_np[name] is not None:
            stats[name] = np.average(np.array(metrics_np[name]), axis=0)

    return metrics_np, stats

def run_ilqr_experiments(num_runs=50, num_steps=150):
    from lander.ilqr import ILQRPlaybackPolicy, run_ilqr

    seed_rng = np.random.RandomState(0)
    env = LunarLander()

    runners = []
    for _ in tqdm(range(num_runs)):
        seed = seed_rng.randint(0, SEED_MAX)

        env.seed(seed)
        env.reset()
        x_trj, u_trj, cost_trace, *_ = run_ilqr(env, num_steps)
        policy = ILQRPlaybackPolicy(x_trj, u_trj)
        runner = LunarLanderRunner(env, policy)

        runner.run(seed)
        runner.update_ilqr_final_cost(cost_trace[-1])
        runners.append(runner)
    
    return runners

def run_rl_experiments(model, num_runs=50):
    from lander.ppo import RLPolicyWrapper

    seed_rng = np.random.RandomState(0)
    env = LunarLander()

    policy = RLPolicyWrapper(model)

    runners = []
    for _ in tqdm(range(num_runs)):
        seed = seed_rng.randint(0, SEED_MAX)

        runner = LunarLanderRunner(env, policy)
        
        runner.run(seed)
        runners.append(runner)
    
    return runners

def save_runner_metrics(runners, fname):
    data = [r.get_metrics() for r in runners]
    with open(fname, 'w') as f:
        json.dump(data, f)

def load_runner_metrics(fname):
    with open(fname, 'r') as f:
        return json.load(f)
