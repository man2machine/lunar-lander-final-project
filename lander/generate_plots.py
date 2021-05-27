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
        plt.title("Lunar Lander ILQR Reward")
        plt.xlabel("Reward")
        plt.ylabel("Count")
        plt.show()

        fuel = runs_np['total_fuel']
        plt.hist(fuel)
        plt.title("Lunar Lander ILQR Fuel")
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

# calc_stats(data)

import matplotlib as mpl

def plot_x_trj(x_trj, end, dest, w, h):
    plt.figure(figsize=(9.5,8))
    # Plot circle
    helipad_x = np.linspace(-1, 1, 50) + dest[0]
    helipad_y = np.zeros(helipad_x.shape) + dest[1]
    plt.plot(helipad_x, helipad_y, linewidth=5)
    ax = plt.gca()

    # Plot resulting trajecotry of car
    plt.plot(x_trj[:end,0], x_trj[:end,2], linewidth=1)
    # Plot rectangles
    for n in range(0, end, 5):
        rect = mpl.patches.Rectangle((-w/2,-h/2), w, h, fill=False)
        t = mpl.transforms.Affine2D().rotate(x_trj[n,4]).translate(x_trj[n,0], x_trj[n,2]) + ax.transData
        rect.set_transform(t)
        ax.add_patch(rect)
    ax.set_aspect(1)
    #plt.ylim((-3, 15))
    #plt.xlim((-5, 5))
    plt.tight_layout()
    plt.show()