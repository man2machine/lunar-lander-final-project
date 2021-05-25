"""
Rocket trajectory optimization is a classic topic in Optimal Control.

According to Pontryagin's maximum principle it's optimal to fire engine full throttle or
turn it off. That's the reason this environment is OK to have discreet actions (engine on or off).

The landing pad is always at coordinates (0,0). The coordinates are the first two numbers in the state vector.
Reward for moving from the top of the screen to the landing pad and zero speed is about 100..140 points.
If the lander moves away from the landing pad it loses reward. The episode finishes if the lander crashes or
comes to rest, receiving an additional -100 or +100 points. Each leg with ground contact is +10 points.
Firing the main engine is -0.3 points each frame. Firing the side engine is -0.03 points each frame.
Solved is 200 points.

Landing outside the landing pad is possible. Fuel is infinite, so an agent can learn to fly and then land
on its first attempt. Please see the source code for details.

To see a heuristic landing, run:

python gym/envs/box2d/lunar_lander.py

To play yourself, run:

python examples/agents/keyboard_agent.py LunarLander-v2

Created by Oleg Klimov. Licensed on the same terms as the rest of OpenAI Gym.
"""


import sys, math
import numpy as np

import Box2D
from Box2D.b2 import (edgeShape, circleShape, fixtureDef, polygonShape, revoluteJointDef, weldJointDef, contactListener)

import gym
from gym import spaces
from gym.utils import seeding, EzPickle

FPS = 50
SCALE = 30.0   # affects how fast-paced the game is, forces should be adjusted as well
#SCALE = 1

MAIN_ENGINE_POWER = 5.2 * 10
SIDE_ENGINE_POWER = 0.6 * 10

INITIAL_RANDOM_FORCE = 1000.0   # Set 1500 to make game harder

w, h = 34, 28

LANDER_POLY = [(w/2, h/2),
            (-w/2, h/2),
            (-w/2, -h/2),
            (w/2, -h/2)]

LEG_AWAY = 0
LEG_DOWN = 8
LEG_W, LEG_H = 17, 2
LEG_SPRING_TORQUE = 0

SIDE_ENGINE_HEIGHT = h/2
SIDE_ENGINE_AWAY = w/2

VIEWPORT_W = 600
VIEWPORT_H = 400

TERRAIN = False

class ContactDetector(contactListener):
    def __init__(self, env):
        contactListener.__init__(self)
        self.env = env

    def BeginContact(self, contact):
        if self.env.lander == contact.fixtureA.body or self.env.lander == contact.fixtureB.body:
            self.env.game_over = True

    def EndContact(self, contact):
        pass


class LunarLander(gym.Env, EzPickle):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : FPS
    }

    def __init__(self):
        EzPickle.__init__(self)
        self.seed()
        self.viewer = None

        self.world = Box2D.b2World()
        self.moon = None
        self.lander = None
        self.particles = []

        self.prev_reward = None

        # useful range is -1 .. +1, but spikes can be higher
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(8,), dtype=np.float32)

        # Action is two floats [main engine, left-right engines].
        # Main engine: -1..0 off, 0..+1 throttle from 50% to 100% power. Engine can't work with less than 50% power.
        # Left-right:  -1.0..-0.5 fire left engine, +0.5..+1.0 fire right engine, -0.5..0.5 off
        self.action_space = spaces.Box(0, +1, (3,), dtype=np.float32)

        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _destroy(self):
        if not self.moon: return
        self.world.contactListener = None
        self._clean_particles(True)
        self.world.DestroyBody(self.moon)
        self.moon = None
        self.world.DestroyBody(self.lander)
        self.lander = None

    def reset(self):
        self._destroy()
        self.world.contactListener_keepref = ContactDetector(self)
        self.world.contactListener = self.world.contactListener_keepref
        self.game_over = False
        self.prev_shaping = None

        W = VIEWPORT_W/SCALE
        H = VIEWPORT_H/SCALE

        # terrain
        CHUNKS = 11
        if TERRAIN:
            height = self.np_random.uniform(0, H/2, size=(CHUNKS+1,))
        else:
            height = np.zeros(CHUNKS+1) + H/4
        chunk_x = [W/(CHUNKS-1)*i for i in range(CHUNKS)]
        self.helipad_x1 = chunk_x[CHUNKS//2-1]
        self.helipad_x2 = chunk_x[CHUNKS//2+1]
        self.helipad_y = H/4
        height[CHUNKS//2-2] = self.helipad_y
        height[CHUNKS//2-1] = self.helipad_y
        height[CHUNKS//2+0] = self.helipad_y
        height[CHUNKS//2+1] = self.helipad_y
        height[CHUNKS//2+2] = self.helipad_y
        smooth_y = [0.33*(height[i-1] + height[i+0] + height[i+1]) for i in range(CHUNKS)]

        self.moon = self.world.CreateStaticBody(shapes=edgeShape(vertices=[(0, 0), (W, 0)]))
        self.sky_polys = []
        for i in range(CHUNKS-1):
            p1 = (chunk_x[i], smooth_y[i])
            p2 = (chunk_x[i+1], smooth_y[i+1])
            self.moon.CreateEdgeFixture(
                vertices=[p1,p2],
                density=0,
                friction=0.1)
            self.sky_polys.append([p1, p2, (p2[0], H), (p1[0], H)])

        self.moon.color1 = (0.0, 0.0, 0.0)
        self.moon.color2 = (0.0, 0.0, 0.0)

        initial_y = VIEWPORT_H/SCALE
        self.lander = self.world.CreateDynamicBody(
            position=(VIEWPORT_W/SCALE/2, initial_y),
            angle=0.0,
            fixtures = fixtureDef(
                shape=polygonShape(vertices=[(x/SCALE, y/SCALE) for x, y in LANDER_POLY]),
                density=5.0,
                friction=0.1,
                categoryBits=0x0010,
                maskBits=0x001,   # collide only with ground
                restitution=0.0)  # 0.99 bouncy
                )
        self.lander.color1 = (0.5, 0.4, 0.9)
        self.lander.color2 = (0.3, 0.3, 0.5)
        self.lander.ApplyForceToCenter( (
            self.np_random.uniform(-INITIAL_RANDOM_FORCE, INITIAL_RANDOM_FORCE),
            self.np_random.uniform(-INITIAL_RANDOM_FORCE, INITIAL_RANDOM_FORCE)
            ), True)
        
        # self.lander.ApplyForceToCenter( (
        #     -500,
        #     200
        # ), True)
        self.drawlist = [self.lander]

        return self.step(np.array([0, 0, 0]))[0]

    def _create_particle(self, mass, x, y, ttl):
        p = self.world.CreateDynamicBody(
            position = (x, y),
            angle=0.0,
            fixtures = fixtureDef(
                shape=circleShape(radius=2/SCALE, pos=(0, 0)),
                density=mass,
                friction=0.1,
                categoryBits=0x0100,
                maskBits=0x001,  # collide only with ground
                restitution=0.3)
                )
        p.ttl = ttl
        self.particles.append(p)
        self._clean_particles(False)
        return p

    def _clean_particles(self, all):
        while self.particles and (all or self.particles[0].ttl < 0):
            self.world.DestroyBody(self.particles.pop(0))

    def step(self, action):
        # Engines
        tip  = (math.sin(self.lander.angle), math.cos(self.lander.angle))
        side = (-tip[1], tip[0])

        m_power = 0.0
        m_power = np.clip(action[0], 0.0, 1.0)
        ox = tip[0]
        oy = -tip[1]
        impulse_pos = (self.lander.position[0], self.lander.position[1])
        p = self._create_particle(3.5,  # 3.5 is here to make particle speed adequate
                                    impulse_pos[0],
                                    impulse_pos[1],
                                    m_power)  # particles are just a decoration
        impulse = (-ox * MAIN_ENGINE_POWER * m_power / SCALE, -oy * MAIN_ENGINE_POWER * m_power / SCALE)
        #print(np.array(impulse) * FPS)
        p.ApplyLinearImpulse((-impulse[0], -impulse[1]),
                                impulse_pos,
                                True)
        # self.lander.ApplyLinearImpulse(impulse,
        #                                 impulse_pos,
        #                                 True)
        self.lander.ApplyForce((impulse[0] * FPS, impulse[1] * FPS),
                                        impulse_pos,
                                        True)
        
        
        # Orientation engines
        for a, d in zip([action[1], action[2]], [-1, 1]):
            direction = d
            s_power = np.clip(a, 0.0, 1.0)
            ox = side[0] * direction
            oy =  -side[1] * direction
            impulse_pos = (self.lander.position[0] + ox * SIDE_ENGINE_HEIGHT/SCALE - tip[0] * SIDE_ENGINE_AWAY/SCALE,
                            self.lander.position[1] + oy * SIDE_ENGINE_AWAY/SCALE + tip[1] * SIDE_ENGINE_HEIGHT/SCALE)
            impulse = (-ox * SIDE_ENGINE_POWER * s_power / SCALE, -oy * SIDE_ENGINE_POWER * s_power / SCALE)
            if s_power:
                p = self._create_particle(0.2, impulse_pos[0], impulse_pos[1], s_power)
                #print(impulse)
                p.ApplyLinearImpulse((-impulse[0], -impulse[1]),
                                        impulse_pos
                                        , True)
            # self.lander.ApplyLinearImpulse(impulse,
            #                                 impulse_pos,
            #                                 True)
            self.lander.ApplyForce((impulse[0] * FPS, impulse[1] * FPS),
                                        impulse_pos,
                                        True)

        self.world.Step(1.0/FPS, 6*30, 2*30)

        pos = self.lander.position
        vel = self.lander.linearVelocity
        state = [
            (pos.x - VIEWPORT_W/SCALE/2) / (VIEWPORT_W/SCALE/2),
            (pos.y - (self.helipad_y)) / (VIEWPORT_H/SCALE/2),
            vel.x*(VIEWPORT_W/SCALE/2)/FPS,
            vel.y*(VIEWPORT_H/SCALE/2)/FPS,
            self.lander.angle,
            20.0*self.lander.angularVelocity/FPS,
        ]

        state = [
            (pos.x - VIEWPORT_W/SCALE/2) / (VIEWPORT_W/SCALE/2),
            (pos.y - self.helipad_y) / (VIEWPORT_H/SCALE/2),
            vel.x / (VIEWPORT_W/SCALE/2),
            vel.y / (VIEWPORT_H/SCALE/2),
            self.lander.angle,
            self.lander.angularVelocity,
        ]

        reward = 0
        shaping = \
            - 100*np.sqrt(state[0]*state[0] + state[1]*state[1]) \
            - 100*np.sqrt(state[2]*state[2] + state[3]*state[3]) \
            - 100*abs(state[4])
        if self.prev_shaping is not None:
            reward = shaping - self.prev_shaping
        self.prev_shaping = shaping

        reward -= m_power*0.30  # less fuel spent is better, about -30 for heuristic landing
        reward -= s_power*0.03

        done = False
        if self.game_over or abs(state[0]) >= 1.0:
            done = True
            x = state[0] * VIEWPORT_W/SCALE/2 + VIEWPORT_W/SCALE/2
            reward = 0
            if self.helipad_x1 < x < self.helipad_x2:
                reward += 50
            else:
                reward -= 50
            reward += 50 if abs(state[5]) < np.pi/6 else -50
        return np.array(state, dtype=np.float32), reward, done, {}

    def render(self, mode='human'):
        from gym.envs.classic_control import rendering
        if self.viewer is None:
            self.viewer = rendering.Viewer(VIEWPORT_W, VIEWPORT_H)
            self.viewer.set_bounds(0, VIEWPORT_W/SCALE, 0, VIEWPORT_H/SCALE)

        for obj in self.particles:
            obj.ttl -= 0.15
            obj.color1 = (max(0.2, 0.2+obj.ttl), max(0.2, 0.5*obj.ttl), max(0.2, 0.5*obj.ttl))
            obj.color2 = (max(0.2, 0.2+obj.ttl), max(0.2, 0.5*obj.ttl), max(0.2, 0.5*obj.ttl))
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

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')
    
    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None


class LunarLanderContinuous(LunarLander):
    continuous = True

def heuristic(env, s):
    """
    The heuristic for
    1. Testing
    2. Demonstration rollout.

    Args:
        env: The environment
        s (list): The state. Attributes:
                  s[0] is the horizontal coordinate
                  s[1] is the vertical coordinate
                  s[2] is the horizontal speed
                  s[3] is the vertical speed
                  s[4] is the angle
                  s[5] is the angular speed
    returns:
         a: The heuristic to be fed into the step function defined above to determine the next step and reward.
    """

    angle_targ = s[0]*0.5 + s[2]*1.0         # angle should point towards center
    if angle_targ > 0.4: angle_targ = 0.4    # more than 0.4 radians (22 degrees) is bad
    if angle_targ < -0.4: angle_targ = -0.4
    hover_targ = 0.55*np.abs(s[0])           # target y should be proportional to horizontal offset

    angle_todo = (angle_targ - s[4]) * 0.5 - (s[5])*0.5
    hover_todo = (hover_targ - s[1])*0.5 - (s[3])*0.5

    angle_todo *= 10
    u_u = hover_todo * 20
    if angle_todo < 0:
        u_r = abs(angle_todo)
        u_l = 0
    else:
        u_l = abs(angle_todo)
        u_r = 0

    a = np.array([u_u, u_l, u_r])
    a = np.clip(a, 0, 1)
    return a

def demo_heuristic_lander(env, seed=None, render=False):
    env.seed(seed)
    total_reward = 0
    steps = 0
    s = env.reset()
    while True:
        a = heuristic(env, s)
        s, r, done, info = env.step(a)
        total_reward += r
        print(a)

        if render:
            still_open = env.render()
            if still_open == False: break

        if steps % 20 == 0 or done:
            
            print("observations:", " ".join(["{:+0.2f}".format(x) for x in s]))
            print("step {} total_reward {:+0.2f}".format(steps, total_reward))
        steps += 1
        if done: break
    return total_reward


if __name__ == '__main__':
    env = LunarLanderContinuous()
    demo_heuristic_lander(env, render=True)
