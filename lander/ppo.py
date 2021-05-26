
from lunar.lunar_lander import LunarLander
from stable_baselines3 import PPO

def get_ppo_model(env, timesteps=10000):
    model = PPO("MlpPolicy", env)
    model.learn(total_timesteps=timesteps)
    return model

if __name__ == '__main__':
    env = LunarLander()
    model = get_ppo_model(env)

    obs = env.reset()
    for i in range(1000):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        env.render()
        if done:
            obs = env.reset()

    env.close()
