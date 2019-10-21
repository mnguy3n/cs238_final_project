import roboschool
import gym

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2

env = gym.make('RoboschoolHalfCheetah-v1')
env = DummyVecEnv([lambda: env]) # The algorithms require a vectorized environment to run

# Train model
model = PPO2(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=10000000)
model.save("ppo2_half_cheetah_model")
del model

model = PPO2.load("ppo2_half_cheetah_model")

# Run trained agent
obs = env.reset()
for i in range((int)(10e5)):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
env.close()
