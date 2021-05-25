# -*- coding: utf-8 -*-
"""
Created on Sat Apr 24 15:14:26 2021

@author: Shahir
"""

import numpy as np
import matplotlib.pyplot as plt

import pydrake
from pydrake.all import (Variable, SymbolicVectorSystem, DiagramBuilder,
                         LogOutput, Simulator, ConstantVectorSource,
                         MathematicalProgram, Solve, SnoptSolver, PiecewisePolynomial)
import pydrake.symbolic as sym

from lander.lunar_lander import LunarLander, H

def sigmoid(x, m=np):
    return 1 / (1 + m.exp(-x))

def get_discrete_dynamics(env):
    pass

def get_continuous_dynamics(env):
    pass

def rollout(env, x0, u_trj):
    x_trj = np.zeros((u_trj.shape[0]+1, x0.shape[0]))
    # TODO: Define the rollout here and return the state trajectory x_trj: [N, number of states]
    x_trj[0] = x0
    for n in range(x_trj.shape[0] - 1):
        x = x_trj[n]
        x_next = env.discrete_dynamics(x, u_trj[n])
        x_trj[n + 1] = x_next
    return x_trj

eps = 1e-6 # The derivative of sqrt(x) at x=0 is undefined. Avoid by subtle smoothing
def cost_stage(x, u):
    # x = [x position, x velocity, y position, y velocity, angle, angular velocity] 
    # u = [u_l, u_r, u_u]
    m = sym if x.dtype == object else np # Check type for autodiff
    c_dest = 3*(x[0]**2 + (x[2] - H/2)**2)/(x[2] + eps)
    # c_dest += -x[2]
    c_vel = (x[3]**2)*1
    # c_vel = 0
 
    u = sigmoid(u, m) # squashing as suggested by https://github.com/anassinator/ilqr/issues/11
    u_l, u_r, u_u = u[0], u[1], u[2]
    c_test = (1 - u_u) * (x[0]**2)
    c_control =  0.3 * (u_u**2)

    return c_dest + c_vel + c_control

def cost_final(x):
    return 0
    m = sym if x.dtype == object else np # Check type for autodiff
    c_dest = 5 * m.sqrt((x[0])**2 + (x[2])**2 + x[4]**2 + eps)
    c_vel = 1*(x[1]**2 + x[3]**2)
    c_landing = 0
    return 0