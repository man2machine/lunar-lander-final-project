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

N_X = 6
N_U = 3
eps = 1e-6 # The derivative of sqrt(x) at x=0 is undefined. Avoid by subtle smoothing

def sigmoid(x, m=np):
    if m == np:
        output = []
        for x_i in x:
            c = np.clip(-x_i, -100, 100)
            # print("clip", c)
            output.append(1/(1+m.exp(c)))
        return output
    return [1/(1+m.exp(-x_i)) for x_i in x]
    # return 1 / (1 + m.exp(-x))

def rollout(env, x0, u_trj):
    x_trj = np.zeros((u_trj.shape[0]+1, x0.shape[0]))
    # TODO: Define the rollout here and return the state trajectory x_trj: [N, number of states]
    x_trj[0] = x0
    for n in range(x_trj.shape[0] - 1):
        x = x_trj[n]
        x_next = env.discrete_dynamics(x, sigmoid(u_trj[n]))
        x_trj[n + 1] = x_next
    return x_trj

def cost_stage(x, u):
    # x = [x position, x velocity, y position, y velocity, angle, angular velocity] 
    # u = [u_l, u_r, u_u]
    m = sym if x.dtype == object else np # Check type for autodiff
    c_dest = (x[0]**2 + x[2]**2)
    # c_dest += -x[2]
    c_vel = (x[3]**2)*0.1
    # c_vel = 0
 
    u = sigmoid(u, m) # squashing as suggested by https://github.com/anassinator/ilqr/issues/11
    u_l, u_r, u_u = u[0], u[1], u[2]
    c_test = (1 - u_u) * (x[0]**2)
    c_control =  0.3 * (u_u**2)
    c_control = 0

    return (c_dest + c_vel + c_control) * 1e-4

def cost_final(x):
    # return 0
    m = sym if x.dtype == object else np # Check type for autodiff
    # c_dest = 5 * m.sqrt((x[0])**2 + (x[2])**2 + x[4]**2 + eps)
    # c_vel = 1*(x[1]**2 + x[3]**2)
    c_landing = (x[3]**2)*1
    return c_landing * 10**(-3)

def cost_trj(x_trj, u_trj):
    total = 0.0
    for n in range(u_trj.shape[0]):
        x = x_trj[n]
        u = u_trj[n]
        c = cost_stage(x, u)
        total += c
    total += cost_final(x_trj[-1])
    return total

class ILQRDerivatives():
    def __init__(self, env, cost_stage, cost_final, n_x=N_X, n_u=N_U):
        self.x_sym = np.array([sym.Variable("x_{}".format(i)) for i in range(n_x)])
        self.u_sym = np.array([sym.Variable("u_{}".format(i)) for i in range(n_u)])
        x = self.x_sym
        u = self.u_sym
        
        l = cost_stage(x, u)
        self.l_x = sym.Jacobian([l], x).ravel()
        self.l_u = sym.Jacobian([l], u).ravel()
        self.l_xx = sym.Jacobian(self.l_x, x)
        self.l_ux = sym.Jacobian(self.l_u, x)
        self.l_uu = sym.Jacobian(self.l_u, u)
        
        l_final = cost_final(x)
        self.l_final_x = sym.Jacobian([l_final], x).ravel()
        self.l_final_xx = sym.Jacobian(self.l_final_x, x)
        
        f = env.discrete_dynamics(x, sigmoid(u, sym), sym)
        self.f_x = sym.Jacobian(f, x)
        self.f_u = sym.Jacobian(f, u)
    
    def stage(self, x, u):
        env = {self.x_sym[i]: x[i] for i in range(x.shape[0])}
        env.update({self.u_sym[i]: u[i] for i in range(u.shape[0])})
        
        l_x = sym.Evaluate(self.l_x, env).ravel()
        l_u = sym.Evaluate(self.l_u, env).ravel()
        l_xx = sym.Evaluate(self.l_xx, env)
        l_ux = sym.Evaluate(self.l_ux, env)
        l_uu = sym.Evaluate(self.l_uu, env)
        
        f_x = sym.Evaluate(self.f_x, env)
        f_u = sym.Evaluate(self.f_u, env)

        return l_x, l_u, l_xx, l_ux, l_uu, f_x, f_u
    
    def final(self, x):
        env = {self.x_sym[i]: x[i] for i in range(x.shape[0])}
        
        l_final_x = sym.Evaluate(self.l_final_x, env).ravel()
        l_final_xx = sym.Evaluate(self.l_final_xx, env)
        
        return l_final_x, l_final_xx

def Q_terms(l_x, l_u, l_xx, l_ux, l_uu, f_x, f_u, V_x, V_xx):
    Q_x = l_x + np.dot(f_x.T, V_x)
    Q_u = l_u + np.dot(f_u.T, V_x)
    Q_xx = l_xx + np.dot(np.dot(f_x.T, V_xx), f_x)
    Q_ux = l_ux + np.dot(np.dot(f_u.T, V_xx), f_x)
    Q_uu = l_uu + np.dot(np.dot(f_u.T, V_xx), f_u)
    return Q_x, Q_u, Q_xx, Q_ux, Q_uu

def gains(Q_uu, Q_u, Q_ux):
    Q_uu_inv = np.linalg.inv(Q_uu)
    k = -np.dot(Q_uu_inv, Q_u.T)
    K = -np.dot(Q_uu_inv, Q_ux)
    return k, K

def V_terms(Q_x, Q_u, Q_xx, Q_ux, Q_uu, K, k):
    V_x = (Q_x + np.dot(Q_u, K)) + (np.dot(k.T, Q_ux) + np.dot(Q_ux.T, k) + np.dot(k.T, (np.dot(Q_uu.T, K)))) + np.dot(K.T, (np.dot(Q_uu.T, k)))
    V_xx = Q_xx + np.dot(K.T, Q_ux) + np.dot(Q_ux.T, K) + np.dot(K.T, (np.dot(Q_uu.T, K)))
    return V_x, V_xx

def expected_cost_reduction(Q_u, Q_uu, k):
    return -Q_u.T.dot(k) - 0.5 * k.T.dot(Q_uu.dot(k))

def forward_pass(env, x_trj, u_trj, k_trj, K_trj):
    x_trj_new = np.zeros(x_trj.shape)
    x_trj_new[0, :] = x_trj[0, :]
    u_trj_new = np.zeros(u_trj.shape)
    for n in range(u_trj.shape[0]):
        x = x_trj_new[n]
        u_trj_new[n, :] = u_trj[n] + k_trj[n] + np.dot(K_trj[n], x_trj_new[n] - x_trj[n])
        x_trj_new[n+1, :] = env.discrete_dynamics(x_trj_new[n], sigmoid(u_trj_new[n]))
    return x_trj_new, u_trj_new

def backward_pass(derivs, x_trj, u_trj, regu):
    k_trj = np.zeros([u_trj.shape[0], u_trj.shape[1]])
    K_trj = np.zeros([u_trj.shape[0], u_trj.shape[1], x_trj.shape[1]])
    expected_cost_redu = 0
    # Set terminal boundary condition here (V_x, V_xx)
    V_x = np.zeros((x_trj.shape[1],))
    V_xx = np.zeros((x_trj.shape[1],x_trj.shape[1]))
    N = x_trj.shape[0] - 1
    l_final_x, l_final_xx = derivs.final(x_trj[N])
    V_x = l_final_x
    V_xx = l_final_xx
    for n in range(u_trj.shape[0]-1, -1, -1):
        # First compute derivatives, then the Q-terms
        x = x_trj[n]
        u = u_trj[n]
        l_x, l_u, l_xx, l_ux, l_uu, f_x, f_u = derivs.stage(x, u)
        Q_x, Q_u, Q_xx, Q_ux, Q_uu = Q_terms(l_x, l_u, l_xx, l_ux, l_uu, f_x, f_u, V_x, V_xx)
        # We add regularization to ensure that Q_uu is invertible and nicely conditioned
        Q_uu_regu = Q_uu + np.eye(Q_uu.shape[0])*regu
        k, K = gains(Q_uu_regu, Q_u, Q_ux)
        k_trj[n,:] = k
        K_trj[n,:,:] = K
        V_x, V_xx = V_terms(Q_x, Q_u, Q_xx, Q_ux, Q_uu, K, k)
        expected_cost_redu += expected_cost_reduction(Q_u, Q_uu, k)
    return k_trj, K_trj, expected_cost_redu

def run_ilqr(env, N, max_iter=50, regu_init=10):
    # ilqr uses discrete dynamics directly from the env, instead of step
    # this bypasses the normal simulation process, so there is no exit condition checking,
    # to see if there are collissions or the lander is out of bounds

    x0 = env.get_state()
    derivs = ILQRDerivatives(env, cost_stage, cost_final)

    # First forward rollout
    u_trj = np.random.randn(N-1, N_U)*0.1
    x_trj = rollout(env, x0, u_trj)
    total_cost = cost_trj(x_trj, u_trj)
    regu = regu_init
    max_regu = 100
    min_regu = 0.01
    
    # Setup traces
    cost_trace = [total_cost]
    expected_cost_redu_trace = []
    redu_ratio_trace = [1]
    redu_trace = []
    regu_trace = [regu]
    
    # Run main loop
    for it in range(max_iter):
        # Backward and forward pass
        k_trj, K_trj, expected_cost_redu = backward_pass(derivs, x_trj, u_trj, regu)
        x_trj_new, u_trj_new = forward_pass(env, x_trj, u_trj, k_trj, K_trj)
        # Evaluate new trajectory
        total_cost = cost_trj(x_trj_new, u_trj_new)
        cost_redu = cost_trace[-1] - total_cost
        redu_ratio = cost_redu / abs(expected_cost_redu)
        # Accept or reject iteration
        if cost_redu > 0:
            # Improvement! Accept new trajectories and lower regularization
            redu_ratio_trace.append(redu_ratio)
            cost_trace.append(total_cost)
            x_trj = x_trj_new
            u_trj = u_trj_new
            regu *= 0.7
        else:
            # Reject new trajectories and increase regularization
            regu *= 2.0
            cost_trace.append(cost_trace[-1])
            redu_ratio_trace.append(0)
        regu = min(max(regu, min_regu), max_regu)
        regu_trace.append(regu)
        redu_trace.append(cost_redu)

        # Early termination if expected improvement is small
        # if expected_cost_redu <= 1e-10:
        #     break
            
    return x_trj, u_trj, cost_trace, regu_trace, redu_ratio_trace, redu_trace

class ILQRPlaybackPolicy:
    def __init__(self, x_trj, u_trj, i=0):
        self.x_trj = x_trj
        self.u_trj = u_trj
        self.i = i
    
    def get_state(self):
        data = {
            'x': self.x_trj,
            'u': self.u_trj,
            'i': self.i
        }
        return data
    
    @classmethod
    def load_state(cls, data):
        return cls(data['x'], data['u'], data['i'])

    def predict(self, s):
        if self.i >= len(self.u_trj):
            raise StopIteration
        u = self.u_trj[self.i]
        self.i += 1
        return u
    
class ILQRModelPredictivePolicy:
    def __init__(self, env):
        self.env = env
        self._sub_policies = [None]
    
    def get_state(self):
        datas = []
        for p in self._sub_policies:
            if p:
                datas.append(p.get_state())
            else:
                datas.append(None)

    def load_state(self, datas):
        self._sub_policies = []
        for d in datas:
            if d:
                self._sub_policies.append(ILQRPlaybackPolicy.load_state(d))
            else:
                self._sub_policies.append(None)

    def predict(self, s):
        policy = self._sub_policies[-1]
        if policy is None:
            x_trj, u_trj, *_ = run_ilqr(self.env, 500)
            policy = ILQRPlaybackPolicy(x_trj, u_trj)
            self._sub_policies[-1] = policy
        try:
            a = policy.predict(s)
        except StopIteration:
            self._sub_policies.append(None)
            return self.predict(s)
        return a

if __name__ == '__main__':
    env = LunarLander()
    env.seed(0)
    env.reset()

    num_steps = 1000
    x_trj, u_trj, cost_trace, regu_trace, redu_ratio_trace, redu_trace = run_ilqr(env, num_steps)

    plt.subplots(figsize=(10, 6))
    # Plot results
    plt.subplot(2, 2, 1)
    plt.plot(cost_trace)
    plt.xlabel('# Iteration')
    plt.ylabel('Total cost')
    plt.title('Cost trace')

    plt.subplot(2, 2, 2)
    delta_opt = (np.array(cost_trace) - cost_trace[-1])
    plt.plot(delta_opt)
    plt.yscale('log')
    plt.xlabel('# Iteration')
    plt.ylabel('Optimality gap')
    plt.title('Convergence plot')

    plt.subplot(2, 2, 3)
    plt.plot(redu_ratio_trace)
    plt.title('Ratio of actual reduction and expected reduction')
    plt.ylabel('Reduction ratio')
    plt.xlabel('# Iteration')

    plt.subplot(2, 2, 4)
    plt.plot(regu_trace)
    plt.title('Regularization trace')
    plt.ylabel('Regularization')
    plt.xlabel('# Iteration')
    plt.tight_layout()

    policy = ILQRPlaybackPolicy(x_trj, u_trj)
    
    obs = env.get_state()
    for i in range(1000):
        action = policy.predict(obs)
        obs, reward, done, info = env.step(action)
        env.render()
        if done:
            obs = env.reset()

    env.close()

