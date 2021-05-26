# -*- coding: utf-8 -*-
"""
Created on Sun May 23 02:49:48 2021

@author: Shahir
"""


import sys
import math

import numpy as np

import Box2D
from Box2D.b2 import (edgeShape, circleShape, fixtureDef, polygonShape)

import gym
from gym import spaces
from gym.utils import seeding, EzPickle

SIMULATION_RATE = 50
INITIAL_RANDOM_FORCE = 5000
W, H = 34, 28
GRAVITY = 300
LANDER_MASS = 5
LANDER_POLY = np.array([
    [W/2, H/2],
    [-W/2, H/2],
    [-W/2, -H/2],
    [W/2, -H/2]
])

ENGINE_FORCE_LIMITS = np.array([1.2, 1.2, 10.2]) * LANDER_MASS * SIMULATION_RATE
LANDER_CENTER = np.array([0, 0])
LANDER_MOMENT_I = (1/12) * (W*W + H*H) * LANDER_MASS
ENGINE_POINTS = np.array([
    [W/2, H/2],
    [-W/2, H/2],
    [0, -H/2]
])
ENGINE_DIRS = np.array([
    [-1, 0],
    [1, 0],
    [0, 1]
])

VIEWPORT_W = 600
VIEWPORT_H = 400
HELIPAD_SIZE = 100

class LunarLander(gym.Env):
    def __init__(self):
        self.seed()
        self.viewer = None

        self.world = Box2D.b2World()
        self.moon = None
        self.lander = None
        self.particles = []

        self.prev_reward = None
        self.max_steps_reward = 300
        self.num_steps = None

        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(6,), dtype=np.float32)
        self.action_space = spaces.Box(0, 1, (3,), dtype=np.float32)

        self.reset()

    def seed(self, seed=None):
        self.rng, seed = seeding.np_random(seed)
        return [seed]

    def _destroy(self):
        if not self.moon:
            return
        self.world.contactListener = None
        self._clean_particles(True)
        self.world.DestroyBody(self.moon)
        self.moon = None
        self.world.DestroyBody(self.lander)
        self.lander = None

    def reset(self):
        self._destroy()
        # self.world.contactListener_keepref = ContactDetector(self)
        # self.world.contactListener = self.world.contactListener_keepref
        self.game_over = False
        self.prev_shaping = None
        self.num_steps = 0

        # self.raw_to_pix_scale = np.array([VIEWPORT_H / 2, VIEWPORT_H / 2])
        # self.raw_to_rl_scale = np.array([1, 1])

        self.raw_to_pix_scale = np.array([1, 1])
        self.raw_to_rl_scale = 1 / np.array([VIEWPORT_H / 2, VIEWPORT_H / 2])

        self.helipad_y = VIEWPORT_H / 4
        self.terrain_x = [0, VIEWPORT_W]
        self.terrain_y = [self.helipad_y, self.helipad_y]
        self.world_origin = np.array([VIEWPORT_W/2, self.helipad_y])

        self.helipad_x1 = self.world_origin[0] - HELIPAD_SIZE / 2
        self.helipad_x2 = self.world_origin[0] + HELIPAD_SIZE / 2

        self.moon = self.world.CreateStaticBody(shapes=edgeShape(vertices=[(0, 0), (VIEWPORT_W, 0)]))
        self.sky_polys = []
        for i in range(len(self.terrain_x)-1):
            p1 = (self.terrain_x[i], self.terrain_y[i])
            p2 = (self.terrain_x[i+1], self.terrain_y[i+1])
            self.moon.CreateEdgeFixture(
                vertices=[p1, p2],
                density=0
            )
            self.sky_polys.append([p1, p2, (p2[0], VIEWPORT_H), (p1[0], VIEWPORT_H)])
        self.moon.color1 = (0.0, 0.0, 0.0)
        self.moon.color2 = (0.0, 0.0, 0.0)

        initial_x_world = self.world_origin[0]
        initial_y_world = VIEWPORT_H
        self.lander = self.world.CreateStaticBody(
            position=(initial_x_world, initial_y_world),
            angle=0.0,
            fixtures=fixtureDef(
                shape=polygonShape(vertices=LANDER_POLY.tolist()),
                categoryBits=0x0010,
                maskBits=0x001
            )
        )
        self.lander.color1 = (0.5, 0.4, 0.9)
        self.lander.color2 = (0.3, 0.3, 0.5)

        self.drawlist = [self.lander]

        f = self.rng.uniform(-INITIAL_RANDOM_FORCE, INITIAL_RANDOM_FORCE, 2)
        x_pix = np.array([
            initial_x_world - self.world_origin[0],
            f[0] / SIMULATION_RATE,
            initial_y_world - self.world_origin[1],
            f[1] / SIMULATION_RATE,
            0,
            0
        ])
        self.x = x_pix.copy()
        self.x[0] /= self.raw_to_pix_scale[0]
        self.x[1] /= self.raw_to_pix_scale[0]
        self.x[2] /= self.raw_to_pix_scale[1]
        self.x[3] /= self.raw_to_pix_scale[1]

        return self.x

    def _create_particle(self, mass, x, y, ttl):
        p = self.world.CreateDynamicBody(
            position=(x, y),
            angle=0.0,
            fixtures = fixtureDef(
                shape=circleShape(radius=2, pos=(0, 0)),
                density=mass,
                friction=0.1,
                categoryBits=0x0100,
                maskBits=0x001, # collide only with ground
                restitution=0
            )
        )
        p.ttl = ttl
        self.particles.append(p)
        self._clean_particles(False)
        return p

    def _clean_particles(self, all):
        while self.particles and (all or self.particles[0].ttl < 0):
            self.world.DestroyBody(self.particles.pop(0))

    def continuous_dynamics(self, x, u, m=np, return_info=False):
        # x = [x position, x velocity, y position, y velocity, angle, angular velocity] 
        # u = [left thrust, right thrust, upwards thrust] (each bounded from 0 to 1)
        eng_force_mags = u * ENGINE_FORCE_LIMITS
        x_d = np.array([
            x[1] * self.raw_to_pix_scale[0],
            0,
            x[3] * self.raw_to_pix_scale[1],
            -GRAVITY,
            x[5],
            0
        ])

        s = m.sin(x[4])
        c = m.cos(x[4])
        def rotate_vec(start_vec):
            return np.array([start_vec[0] * c - start_vec[1] * s,
                             start_vec[0] * s + start_vec[1] * c])
        
        forces = []
        force_locs = []
        for i in range(3):
            appl_point = ENGINE_POINTS[i] # point of application of force
            appl_dir = ENGINE_DIRS[i] # local frame of ref
            appl_dir_rot = rotate_vec(appl_dir) # global frame of ref
            center_to_eng_rot = rotate_vec(appl_point - LANDER_CENTER)
            eng_to_center_rot = rotate_vec(LANDER_CENTER - appl_point)
            force_vec = appl_dir_rot * eng_force_mags[i]
            forces.append(force_vec)
            force_locs.append(center_to_eng_rot)

            proj_scale = np.dot(force_vec, eng_to_center_rot) / np.dot(eng_to_center_rot, eng_to_center_rot)
            force_proj = eng_to_center_rot * proj_scale
            force_cross = center_to_eng_rot[0] * force_vec[1] - center_to_eng_rot[1] * force_vec[0]

            x_d[1] += force_proj[0] / LANDER_MASS
            x_d[3] += force_proj[1] / LANDER_MASS
            x_d[5] += force_cross / LANDER_MOMENT_I
        
        x_d[0] /= self.raw_to_pix_scale[0]
        x_d[1] /= self.raw_to_pix_scale[0]
        x_d[2] /= self.raw_to_pix_scale[1]
        x_d[3] /= self.raw_to_pix_scale[1]

        self.num_steps += 1
        
        if return_info: # info is in pix coordinates
            return x_d, forces, force_locs
        return x_d
    
    def discrete_dynamics(self, x, u, m=np):
        dt = 1 / SIMULATION_RATE
        x_next = x + self.continuous_dynamics(x, u, m) * dt
        return x_next
    
    def get_state(self):
        return self.x
    
    def detect_collision(self):
        f = self.lander.fixtures[0]
        trans = f.body.transform
        vertices = [trans*v for v in f.shape.vertices]
        for p in vertices:
            if p[1] < self.helipad_y:
                return True
        return False
    
    def detect_out_of_bounds(self):
        return np.abs(self.x[0]) * self.raw_to_pix_scale[0] > VIEWPORT_W / 2
    
    def detect_inside_helipad(self):
        x_0_pix = self.x[0] * self.raw_to_pix_scale[0] + self.world_origin[0]
        return self.helipad_x1 < x_0_pix < self.helipad_x2

    def detect_land_slowly(self):
        return np.abs(self.x[3]) * self.raw_to_rl_scale[0] < 0.2
    
    def detect_land_upright(self):
        return abs(self.x[4]) < np.pi / 12
    
    def detect_too_long(self):
        return self.num_steps > self.max_steps_reward

    def step(self, action, apply_anim=False):
        action = np.clip(action, 0, 1)
        dt = 1 / SIMULATION_RATE
        x_d, forces, force_locs = self.continuous_dynamics(self.x, action,
            return_info=True)
        x_next = self.x + x_d * dt
        self.x = x_next

        shift = np.array([
            self.x[0] * self.raw_to_pix_scale[0] + self.world_origin[0],
            self.x[2] * self.raw_to_pix_scale[1] + self.world_origin[1]
        ])
        self.lander.position.x = shift[0]
        self.lander.position.y = shift[1]
        self.lander.angle = self.x[4]

        if apply_anim:
            pass
            # for u_i, f, loc, f_lim in zip(action, forces, force_locs, ENGINE_FORCE_LIMITS):
            #     if np.linalg.norm(f) == 0:
            #         continue
            #     force_loc_world = loc + shift
            #     p = self._create_particle(1e-12, force_loc_world[0], force_loc_world[1], u_i * 0.5)
            #     impulse = -f / f_lim * 25 * 1e30
            #     # if np.linalg.norm(impulse) > 50:
            #     #     impulse /= np.linalg.norm(impulse)
            #     # print(u_i)
            #     print(impulse)
            #     #print(impulse, u_i)
            #     #print(p.mass)
            #     p.ApplyForceToCenter(impulse, True)
            #     #p.ApplyLinearImpulse(impulse, force_loc_world, True)
            # self.world.Step(dt, 6*30, 2*30)
        
        state = self.x.tolist()
        state[0] *= self.raw_to_rl_scale[0]
        state[1] *= self.raw_to_rl_scale[0]
        state[2] *= self.raw_to_rl_scale[1]
        state[3] *= self.raw_to_rl_scale[1]

        info = {}
        reward = 0
        shaping = \
            - 100*np.sqrt(state[0]*state[0] + state[2]*state[2]) \
            - 100*np.sqrt(state[1]*state[1] + state[3]*state[3]) \
            - 100*abs(state[4])
        if self.prev_shaping is not None:
            reward = shaping - self.prev_shaping
        self.prev_shaping = shaping

        reward -= action[2] * 0.30  # less fuel spent is better, about -30 for heuristic landing
        reward -= (action[0] + action[1]) * 0.03
        info = {'fuel': np.sum(action * ENGINE_FORCE_LIMITS) / SIMULATION_RATE}

        done = False
        if self.detect_collision() or self.detect_out_of_bounds():
            reward = 0
            done = True
            if self.detect_too_long():
                reward -= 15
            if self.detect_inside_helipad():
                reward += 70
                if self.detect_land_upright():
                    reward += 30
                if self.detect_land_slowly():
                    reward += 30
            else:
                reward -= 100
        
        return state, reward, done, info

    def render(self, mode='human'):
        from gym.envs.classic_control import rendering
        if self.viewer is None:
            self.viewer = rendering.Viewer(VIEWPORT_W, VIEWPORT_H)
            self.viewer.set_bounds(0, VIEWPORT_W, 0, VIEWPORT_H)

        for obj in self.particles:
            obj.ttl -= 0.15
            obj.color1 = (0.7, 0.2, 0.2)
            obj.color2 = (0.7, 0.2, 0.2)

        self._clean_particles(False)

        for p in self.sky_polys:
            self.viewer.draw_polygon(p, color=(0, 0, 0))

        for obj in self.particles + self.drawlist:
            for f in obj.fixtures:
                trans = f.body.transform
                if type(f.shape) is circleShape:
                    t = rendering.Transform(translation=trans*f.shape.pos)
                    self.viewer.draw_circle(f.shape.radius, 20, color=obj.color1).add_attr(t)
                    self.viewer.draw_circle(f.shape.radius, 20, color=obj.color2, filled=False, linewidth=2).add_attr(t)
                else:
                    path = [trans*v for v in f.shape.vertices]
                    self.viewer.draw_polygon(path, color=obj.color1)
                    path.append(path[0])
                    self.viewer.draw_polyline(path, color=obj.color2, linewidth=2)

        for x in [self.helipad_x1, self.helipad_x2]:
            flagy1 = self.helipad_y
            flagy2 = flagy1 + 50
            self.viewer.draw_polyline([(x, flagy1), (x, flagy2)], color=(1, 1, 1))
            self.viewer.draw_polygon([(x, flagy2), (x, flagy2-10), (x + 25, flagy2 - 5)],
                                      color=(0.8, 0.8, 0))
        
        rgb_array = mode == 'rgb_array'
        return self.viewer.render(return_rgb_array=rgb_array)
    
    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

def heuristic(x):
    angle_targ = x[0] * 0.5 + x[1] * 1.0 # angle should point towards center
    angle_targ = np.clip(angle_targ, -0.4, 0.4) # more than 0.4 radians (22 degrees) is bad
    hover_targ = 0.55 * np.abs(x[0]) # target y should be proportional to horizontal offset

    angle_todo = (angle_targ - x[4]) * 0.5 - (x[5]) * 0.5
    hover_todo = (hover_targ - x[2]) * 0.5 - (x[3]) * 0.5
    hover_todo *= 20
    angle_todo *= 10
    u_u = hover_todo
    if angle_todo < 0:
        u_r = abs(angle_todo)
        u_l = 0
    else:
        u_l = abs(angle_todo)
        u_r = 0

    u = np.array([u_l, u_r, u_u])
    u = np.clip(u, 0, 1)

    return u

def demo_heuristic_lander(env, render=False):
    total_reward = 0
    steps = 0
    s = env.reset()
    while True:
        a = heuristic(s)
        s, r, done, info = env.step(a)
        total_reward += r

        if render:
            still_open = env.render()
            if still_open == False:
                break
        
        if steps % 20 == 0 or done:
            print("observations:", " ".join(["{:+0.3f}".format(x) for x in s]))
            print("step {} total_reward {:+0.3f}".format(steps, total_reward))
        steps += 1
        if done:
            break
    return total_reward


if __name__ == '__main__':
    env = LunarLander()
    env.seed(0)
    env.reset()
    demo_heuristic_lander(env, render=True)
    env.reset()
    demo_heuristic_lander(env, render=True)
