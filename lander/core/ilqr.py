# -*- coding: utf-8 -*-
"""
Created on Sat Apr 24 15:14:26 2021

@author: Shahir
"""

import numpy as np

u_l, u_r, u_u = 0, 0, 0
eng_force_limits = [1, 1, 10]
mass = 1
w = 5
h = 2

box_poly = [(w/2, h/2),
            (-w/2, h/2),
            (-w/2, -h/2),
            (w/2, -h/2)]
box_poly = np.array(box_poly)

moment_I = (1/12) * w*h*(w*w + h*h) * mass
eng_points = [box_poly[0], box_poly[1], np.mean(box_poly[2:4])]
eng_dirs = [(-1, 0), (1, 0), (0, 1)]
center = (0, 0)

controls = [u_l, u_r, u_u]
eng_forces = [controls[i] * eng_force_limits[i] for i in range(3)]

(x, x_d, y, y_d, t, t_d) = state = (0, 0, 0, 0, 0, 0)

m = np

for i in range(3):
    f_local = eng_forces[i]
    appl_point = eng_points[i]
    appl_dir = eng_dirs[i] # local frame of ref
    s, c = (m.sin(t), m.cos(t))
    appl_dir_rot = (appl_dir[0] * c - appl_dir[1] * s, appl_dir[0] * s + appl_dir[1] * c) # global frame of ref

    force_vec = (appl_dir_rot[0] * f_local, appl_dir_rot[1] * f_local)
    center_to_eng = (appl_point[0] - center[0], appl_point[1] - center[1])
    eng_to_center = (center[0] - appl_point[0], center[1] - appl_point[1])

    proj_scale = force_vec[0] * eng_to_center[0] + force_vec[1] * eng_to_center[1]
    proj_scale /= eng_to_center[0] * eng_to_center[0] + eng_to_center[1] * eng_to_center[1]
    proj = (eng_to_center[0] * proj_scale, eng_to_center[1] * proj_scale)
    cross = center_to_eng[0] * force_vec[1] - center_to_eng[1] * force_vec[0]

    x_d += proj[0]
    y_d += proj[1]
    t_d += cross

# python libraries

import matplotlib.pyplot as plt
import matplotlib as mpl

# pydrake imports
from pydrake.all import (Variable, SymbolicVectorSystem, DiagramBuilder,
                         LogOutput, Simulator, ConstantVectorSource,
                         MathematicalProgram, Solve, SnoptSolver, PiecewisePolynomial)
import pydrake.symbolic as sym

n_x = 5
n_u = 2
def car_continuous_dynamics(x, u):
    # x = [x position, y position, heading, speed, steering angle] 
    # u = [acceleration, steering velocity]
    m = sym if x.dtype == object else np # Check type for autodiff
    heading = x[2]
    v = x[3]
    steer = x[4]
    x_d = np.array([
        v*m.cos(heading),
        v*m.sin(heading),
        v*m.tan(steer),
        u[0],
        u[1]        
    ])
    return x_d

def discrete_dynamics(x, u):
    dt = 0.1
    # TODO: Fill in the Euler integrator below and return the next state
    x_next = x + car_continuous_dynamics(x, u) * dt
    return x_next

def rollout(x0, u_trj):
    x_trj = np.zeros((u_trj.shape[0]+1, x0.shape[0]))
    # TODO: Define the rollout here and return the state trajectory x_trj: [N, number of states]
    x_trj[0] = x0
    for n in range(x_trj.shape[0] - 1):
        x = x_trj[n]
        x_next = discrete_dynamics(x, u_trj[n])
        x_trj[n + 1] = x_next
    return x_trj

# Debug your implementation with this example code
N = 10
x0 = np.array([1, 0, 0, 1, 0])
u_trj = np.zeros((N-1, n_u))
x_trj = rollout(x0, u_trj)

r = 2.0
v_target = 2.0
eps = 1e-6 # The derivative of sqrt(x) at x=0 is undefined. Avoid by subtle smoothing
def cost_stage(x, u):
    m = sym if x.dtype == object else np # Check type for autodiff
    c_circle = (m.sqrt(x[0]**2 + x[1]**2 + eps) - r)**2
    c_speed = (x[3]-v_target)**2
    c_control = (u[0]**2 + u[1]**2)*0.1
    return c_circle + c_speed + c_control

def cost_final(x):
    m = sym if x.dtype == object else np # Check type for autodiff
    c_circle = (m.sqrt(x[0]**2 + x[1]**2 + eps) - r)**2
    c_speed = (x[3]-v_target)**2
    return c_circle + c_speed

def cost_trj(x_trj, u_trj):
    total = 0.0
    for n in range(u_trj.shape[0]):
        x = x_trj[n]
        u = u_trj[n]
        total += cost_stage(x, u)
    total += cost_final(x_trj[-1])
    return total
    
# Debug your code
cost_trj(x_trj, u_trj)

class derivatives():
    def __init__(self, discrete_dynamics, cost_stage, cost_final, n_x, n_u):
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
        
        f = discrete_dynamics(x, u)
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
        
derivs = derivatives(discrete_dynamics, cost_stage, cost_final, n_x, n_u)
# Test the output:
x = np.array([0, 0, 0, 0, 0])
u = np.array([0, 0])
print(derivs.stage(x, u))
print(derivs.final(x))

def Q_terms(l_x, l_u, l_xx, l_ux, l_uu, f_x, f_u, V_x, V_xx):
    # TODO: Define the Q-terms here
    Q_x = l_x + np.dot(f_x.T, V_x)
    Q_u = l_u + np.dot(f_u.T, V_x)
    Q_xx = l_xx + np.dot(np.dot(f_x.T, V_xx), f_x)
    Q_ux = l_ux + np.dot(np.dot(f_u.T, V_xx), f_x)
    Q_uu = l_uu + np.dot(np.dot(f_u.T, V_xx), f_u)
    return Q_x, Q_u, Q_xx, Q_ux, Q_uu

def gains(Q_uu, Q_u, Q_ux):
    Q_uu_inv = np.linalg.inv(Q_uu)
    # TODO: Implement the feedforward gain k and feedback gain K.
    k = -np.dot(Q_uu_inv, Q_u.T)
    K = -np.dot(Q_uu_inv, Q_ux)
    return k, K

def V_terms(Q_x, Q_u, Q_xx, Q_ux, Q_uu, K, k):
    # TODO: Implement V_x and V_xx, hint: use the A.dot(B) function for matrix multiplcation.
    V_x = (Q_x + np.dot(Q_u, K)) + (np.dot(k.T, Q_ux) + np.dot(Q_ux.T, k) + np.dot(k.T, (np.dot(Q_uu.T, K)))) + np.dot(K.T, (np.dot(Q_uu.T, k)))
    V_xx = Q_xx + np.dot(K.T, Q_ux) + np.dot(Q_ux.T, K) + np.dot(K.T, (np.dot(Q_uu.T, K)))
    return V_x, V_xx

def expected_cost_reduction(Q_u, Q_uu, k):
    return -Q_u.T.dot(k) - 0.5 * k.T.dot(Q_uu.dot(k))

def forward_pass(x_trj, u_trj, k_trj, K_trj):
    x_trj_new = np.zeros(x_trj.shape)
    x_trj_new[0, :] = x_trj[0, :]
    u_trj_new = np.zeros(u_trj.shape)
    # TODO: Implement the forward pass here
    for n in range(u_trj.shape[0]):
        u_trj_new[n, :] = u_trj[n] + k_trj[n] + np.dot(K_trj[n], x_trj_new[n] - x_trj[n])
        x_trj_new[n+1, :] = discrete_dynamics(x_trj_new[n], u_trj_new[n])
    return x_trj_new, u_trj_new

def backward_pass(x_trj, u_trj, regu):
    k_trj = np.zeros([u_trj.shape[0], u_trj.shape[1]])
    K_trj = np.zeros([u_trj.shape[0], u_trj.shape[1], x_trj.shape[1]])
    expected_cost_redu = 0
    # TODO: Set terminal boundary condition here (V_x, V_xx)
    V_x = np.zeros((x_trj.shape[1],))
    V_xx = np.zeros((x_trj.shape[1],x_trj.shape[1]))
    N = x_trj.shape[0] - 1
    l_final_x, l_final_xx = derivs.final(x_trj[N])
    V_x = l_final_x
    V_xx = l_final_xx
    for n in range(u_trj.shape[0]-1, -1, -1):
        # TODO: First compute derivatives, then the Q-terms
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

def run_ilqr(x0, N, max_iter=50, regu_init=100):
    # First forward rollout
    u_trj = np.random.randn(N-1, n_u)*0.0001
    x_trj = rollout(x0, u_trj)
    total_cost = cost_trj(x_trj, u_trj)
    regu = regu_init
    max_regu = 10000
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
        k_trj, K_trj, expected_cost_redu = backward_pass(x_trj, u_trj, regu)
        x_trj_new, u_trj_new = forward_pass(x_trj, u_trj, k_trj, K_trj)
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
        if expected_cost_redu <= 1e-6:
            break
            
    return x_trj, u_trj, cost_trace, regu_trace, redu_ratio_trace, redu_trace

# Setup problem and call iLQR
x0 = np.array([-3.0, 1.0, -0.2, 0.0, 0.0])
N = 50
max_iter=50
regu_init=100
x_trj, u_trj, cost_trace, regu_trace, redu_ratio_trace, redu_trace = run_ilqr(x0, N, max_iter, regu_init)


plt.figure(figsize=(9.5,8))
# Plot circle
theta = np.linspace(0, 2*np.pi, 100)
plt.plot(r*np.cos(theta), r*np.sin(theta), linewidth=5)
ax = plt.gca()

# Plot resulting trajecotry of car
plt.plot(x_trj[:,0], x_trj[:,1], linewidth=5)
w = 2.0
h = 1.0

# Plot rectangles
for n in range(x_trj.shape[0]):
    rect = mpl.patches.Rectangle((-w/2,-h/2), w, h, fill=False)
    t = mpl.transforms.Affine2D().rotate_deg_around(0, 0, 
            np.rad2deg(x_trj[n,2])).translate(x_trj[n,0], x_trj[n,1]) + ax.transData
    rect.set_transform(t)
    ax.add_patch(rect)
ax.set_aspect(1)
plt.ylim((-3,3))
plt.xlim((-4.5,3))
plt.tight_layout()

plt.subplots(figsize=(10,6))
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
