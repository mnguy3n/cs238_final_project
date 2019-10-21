import roboschool
import gym

env = gym.make('RoboschoolWalker2d-v1')
env.reset()
while True:
  env.step(env.action_space.sample())
  env.render()
