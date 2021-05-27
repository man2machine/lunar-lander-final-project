import numpy as np
import matplotlib.pyplot as plt

import json

JSON_FILE = "lander/ilqr_results.json"
with open(JSON_FILE, 'r') as f:
    data = json.load(f)

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
        plt.xlabel("Reward")
        plt.ylabel("Count")
        plt.show()

        fuel = runs_np['total_fuel']
        plt.hist(fuel)
        plt.title("Lunar Lander Fuel")
        plt.xlabel("Fuel")
        plt.ylabel("Count")
        plt.show()

        costs = runs_np['ilqr_final_cost']
        if costs[0] is not None:
            plt.hist(costs)
            plt.title("Lunar Lander ILQR Final Cost")
            plt.xlabel("Final Cost")
            plt.ylabel("Count")
            plt.show()

    return runs_np, stats

calc_stats(data)