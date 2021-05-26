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
from lander.ilqr import ILQRPlaybackPolicy, run_ilqr

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
        }
    
    def run(self, render=False):
        obs = self.env.get_state()
        num_steps = 0
        total_reward = 0
        total_fuel = 0
        for i in range(self.max_iters):
            action = self.policy.predict(obs)
            obs, reward, done, info = self.envenv.step(action)
            if render:
                self.env.render()
            num_steps += 1
            total_reward += reward
            total_fuel += info['fuel']
            if done:
                break
        
        self.data['inside_helipad'] = self.env.detect_inside_helipad()
        self.data['land_upright'] = self.env.detect_land_upright()
        self.data['land_slowly'] = self.detect_land_slowly()
        self.data['touched_ground'] = self.detect_collision()
        self.data['out_of_bounds'] = self.detect_out_of_bounds()
        self.data['num_steps'] = num_steps
        self.data['end_state'] = self.env.get_state().to_list()
        self.data['total_reward'] = int(total_reward)
        self.data['total_fuel'] = float(total_fuel)

        self.env.close()
    
    def update_ilqr_final_cost(self, cost):
        self.data['ilqr_final_cost'] = float(cost)

    def get_metrics(self):
        return self.data

def calc_stats(runs, show=True):
    runs_np = {
            'inside_helipad': None,
            'land_upright': None,
            'land_slowly': None,
            'touched_ground': None,
            'out_of_bounds': None,
            'num_steps': None,
            'end_state': None,
            'total_reward': None,
            'total_fuel': None,
            'ilqr_final_cost': None
        }
    
    for run in runs:
        for metric in runs_np:
            if runs_np[metric] is None:
                runs_np[metric] = np.array([run[metric]])
            else:
                runs_np[metric] = np.append(runs_np[metric], run[metric])
    stats = {}
    for metric in runs_np:
        stats[metric] = np.average(runs_np[metric])
    
    if show:
        print("Averages")
        print(stats)

        reward = runs_np['total_reward']
        plt.hist(reward)
        plt.title("Lunar Lander Reward")
        plt.show()

        fuel = runs_np['total_fuel']
        plt.hist(fuel)
        plt.title("Lunar Lander Fuel")
        plt.show()

        costs = runs_np['ilqr_cost']
        if costs[0] is not None:
            plt.hist(costs)
            plt.title("Lunar Lander ILQR Cost")
            plt.show()

    return runs_np, stats

def run_ilqr_experiments(num_steps=150, num_runs=20):
    env = LunarLander()

    runners = []
    for _ in tqdm(range(num_runs)):
        env.reset()
        x_trj, u_trj, cost_trace, *_ = run_ilqr(env, num_steps)
        policy = ILQRPlaybackPolicy(x_trj, u_trj)
        runner = LunarLanderRunner(env, policy)
        runner.run()
        runner.update_ilqr_final_cost(cost_trace[-1])
        runners.append(runner)
    
    return runners

def run_rl_experiments(model, num_runs=20):
    env = LunarLander()

    runners = []
    for _ in tqdm(range(num_runs)):
        env.reset()
        runner = LunarLanderRunner(env, model)
        runner.run()
        runners.append(runner)
    
    return runners

def save_runner_metrics(runners, fname):
    data = [r.get_metrics() for r in runners]
    with open(fname, 'w') as f:
        json.dump(data, f)
