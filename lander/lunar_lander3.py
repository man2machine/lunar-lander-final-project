# -*- coding: utf-8 -*-
"""
Created on Sun May 23 02:49:48 2021

@author: Shahir
"""


import sys
import math

import numpy as np

import Box2D
from Box2D.b2 import (edgeShape, circleShape, fixtureDef, polygonShape, revoluteJointDef, weldJointDef, contactListener)

import gym
from gym import spaces
from gym.utils import seeding, EzPickle

SCALE = 30

SIMULATION_RATE = 50
INITIAL_RANDOM_VEL = 30
W, H = 34, 28
GRAVITY = 10
LANDER_MASS = 5.2888
LANDER_POLY = np.array([
    [W/2, H/2],
    [-W/2, H/2],
    [-W/2, -H/2],
    [W/2, -H/2]
])

ENGINE_FORCE_LIMITS = (np.array([6, 6, 52])  / SCALE) * SIMULATION_RATE

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

LANDER_MOMENT_I /= SCALE * SCALE

LANDER_POLY /= SCALE
ENGINE_POINTS /= SCALE


VIEWPORT_W = 600
VIEWPORT_H = 400
HELIPAD_SIZE = 72

class ContactDetector(contactListener):
    def __init__(self, env):
        contactListener.__init__(self)
        self.env = env

    def BeginContact(self, contact):
        if self.env.lander == contact.fixtureA.body or self.env.lander == contact.fixtureB.body:
            self.env.game_over = True

class LunarLander(gym.Env):
    def __init__(self):
        self.seed()
        self.viewer = None

        self.world = Box2D.b2World()
        self.moon = None
        self.lander = None
        self.particles = []

        self.prev_reward = None

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

        self.helipad_y = VIEWPORT_H/4 / SCALE
        terrain_x = [0, VIEWPORT_W / SCALE]
        terrain_y = [self.helipad_y, self.helipad_y]
        self.world_origin = np.array([VIEWPORT_W/2 / SCALE, self.helipad_y])

        self.helipad_x1 = self.world_origin[0] - HELIPAD_SIZE / 2 / SCALE
        self.helipad_x2 = self.world_origin[0] + HELIPAD_SIZE / 2 / SCALE

        self.moon = self.world.CreateStaticBody(shapes=edgeShape(vertices=[(0, 0), (VIEWPORT_W / SCALE, 0)]))
        self.sky_polys = []
        for i in range(len(terrain_x)-1):
            p1 = (terrain_x[i], terrain_y[i])
            p2 = (terrain_x[i+1], terrain_y[i+1])
            self.moon.CreateEdgeFixture(
                vertices=[p1, p2],
                density=0
            )
            self.sky_polys.append([p1, p2, (p2[0], VIEWPORT_H / SCALE), (p1[0], VIEWPORT_H / SCALE)])
        self.moon.color1 = (0.0, 0.0, 0.0)
        self.moon.color2 = (0.0, 0.0, 0.0)

        initial_x_world = self.world_origin[0]
        initial_y_world = VIEWPORT_H / SCALE
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

        self.x = np.array([
            initial_x_world - self.world_origin[0],
            np.random.uniform(-100, 100) / SIMULATION_RATE,
            initial_y_world - self.world_origin[1],
            np.random.uniform(-100, 100) / SIMULATION_RATE,
            0,
            0
        ])

        return self.x

    def _create_particle(self, mass, x, y, ttl):
        p = self.world.CreateDynamicBody(
            position=(x, y),
            angle=0.0,
            fixtures = fixtureDef(
                shape=circleShape(radius=2/SCALE, pos=(0, 0)),
                density=mass,
                friction=0.1,
                categoryBits=0x0100,
                maskBits=0x001, # collide only with ground
                restitution=0.3
            )
        )
        p.ttl = ttl
        self.particles.append(p)
        self._clean_particles(False)
        return p

    def _clean_particles(self, all):
        while self.particles and (all or self.particles[0].ttl < 0):
            self.world.DestroyBody(self.particles.pop(0))
    
    def _sigmoid(x, m):
        return 1 / (1 + m.exp(-x))

    def continuous_dynamics(self, x, u, m, squash_controls=False, return_info=False):
        # x = [x position, x velocity, y position, y velocity, angle, angular velocity] 
        # u = [left thrust, right thrust, upwards thrust]
        if squash_controls:
            # squashing as suggested by https://github.com/anassinator/ilqr/issues/11
            u = self._sigmoid(u, m)
        eng_force_mags = u * ENGINE_FORCE_LIMITS
        x_d = np.array([
            x[1],
            0,
            x[3],
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
        
        if return_info:
            return x_d, forces, force_locs
        return x_d
    
    def discrete_dynamics(self, x, u, m, squash_controls=False):
        dt = 1 / SIMULATION_RATE
        x_next = x + self.continuous_dynamics(x, u, m, squash_controls=squash_controls) * dt
        return x_next
    
    def detect_collision(self):
        f = self.lander.fixtures[0]
        trans = f.body.transform
        vertices = [trans*v for v in f.shape.vertices]
        for p in vertices:
            if p[1] < self.helipad_y:
                return True
        return False

    def step(self, action, apply_anim=True):
        action = np.clip(action, 0, 1)
        #print(action)
        dt = 1 / SIMULATION_RATE
        x_d, forces, force_locs = self.continuous_dynamics(self.x, action, np,
            squash_controls=False, return_info=True)
        x_next = self.x + x_d * dt
        self.x = x_next

        shift = np.array([self.x[0] + self.world_origin[0], self.x[2] + self.world_origin[1]])
        self.lander.position.x = shift[0]
        self.lander.position.y = shift[1]
        self.lander.angle = self.x[4]

        if apply_anim:
            # for u_i, f, loc, f_lim in zip(action, forces, force_locs, ENGINE_FORCE_LIMITS):
            #     if np.linalg.norm(f) == 0:
            #         continue
            #     force_loc_world = loc + shift
            #     p = self._create_particle(1, force_loc_world[0], force_loc_world[1], u_i * 0.5)
            #     impulse = -f / f_lim * 5
            #     if np.linalg.norm(impulse) > 1:
            #         impulse /= np.linalg.norm(impulse)
            #     print(u_i)
            #     print(impulse)
            #     #print(impulse, u_i)
            #     #p.ApplyForceToCenter(impulse, True)
            #     p.ApplyLinearImpulse(impulse, force_loc_world, True)
            # print()
            self.world.Step(dt, 6*30, 2*30)
            pass
        
        state = [
            self.x[0] / (VIEWPORT_W/2/SCALE),
            self.x[1] / (VIEWPORT_W/2/SCALE),
            self.x[2] / (VIEWPORT_H/2/SCALE),
            self.x[3] / (VIEWPORT_H/2/SCALE),
            self.x[4],
            self.x[5],
        ]

        # state = [
        #     self.x[0],
        #     self.x[1],
        #     self.x[2],
        #     self.x[3],
        #     self.x[4],
        #     self.x[5],
        # ]
        #print(state)
        
        reward = 0
        done = False
        if self.detect_collision():
            done = True
        
        return state, reward, done, {}

    def render(self, mode='human'):
        from gym.envs.classic_control import rendering
        if self.viewer is None:
            self.viewer = rendering.Viewer(VIEWPORT_W, VIEWPORT_H)
            self.viewer.set_bounds(0, VIEWPORT_W/SCALE, 0, VIEWPORT_H/SCALE)

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
            flagy2 = flagy1 + 50/SCALE
            self.viewer.draw_polyline([(x, flagy1), (x, flagy2)], color=(1, 1, 1))
            self.viewer.draw_polygon([(x, flagy2), (x, flagy2-10/SCALE), (x + 25/SCALE, flagy2 - 5/SCALE)],
                                      color=(0.8, 0.8, 0))
        
        rgb_array = mode == 'rgb_array'
        return self.viewer.render(return_rgb_array=rgb_array)
    
    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

class LunarLanderContinuous(LunarLander):
    continuous = True

def heuristic(x):
    #return np.array([0, 0, 0])
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

def demo_heuristic_lander(env, seed=None, render=False):
    env.seed(seed)
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
            print("observations:", " ".join(["{:+0.2f}".format(x) for x in s]))
            print("step {} total_reward {:+0.2f}".format(steps, total_reward))
        steps += 1
        if done:
            break
    return total_reward


if __name__ == '__main__':
    env = LunarLanderContinuous()
    demo_heuristic_lander(env, render=True)
