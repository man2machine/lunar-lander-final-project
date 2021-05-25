
from lunar_lander_old3 import LunarLanderContinuous
from stable_baselines3 import PPO

env = LunarLanderContinuous()
model = PPO("MlpPolicy", env)
model.learn(total_timesteps=10000)

obs = env.reset()
for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
      obs = env.reset()

env.close()