import gym
import d4rl
import minari


print(gym.__version__)
'''
env = minari.DataCollectorV0(gym.make("Ant-v2"))

env.reset()

for _ in range(1000):
    action = env.action_space.sample()
    obs, rew, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        env.reset()

dataset = minari.create_dataset_from_collector_env(dataset_id="Ant-v2-dataset-v0", collector_env=env)
print(dataset)
'''