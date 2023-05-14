import numpy as np

from parameterized_policy import ParameterizedGaussianPolicy
import torch
import gymnasium as gym
from utils import collect_episode, evaluate_agent, reinforce_update

policy = ParameterizedGaussianPolicy(input_size=3,
                                     output_size=1,
                                     action_scale=2.0,
                                     hidden_layers=(128, ))

# params = policy.get_parameters_vector()
# print(params)
# new_params = params + 1
# policy.set_parameters_vector(new_params)
# print(policy.get_parameters_vector())
# assert torch.all(torch.eq(policy.get_parameters_vector(), new_params))


# state = torch.tensor([1., 2., 3.]).unsqueeze(0)
# action = policy.sample_action(state)
# print(policy.get_grad_log_likelihood(state, action))


env = gym.make("Pendulum-v1")
env.reset()
# change gravity for easier


# observation, _ = env.reset()
# for _ in range(500):
#     action = [1]  # agent policy that uses the observation and info
#     observation, reward, terminated, truncated, info = env.step(action)
#
#     if terminated or truncated:
#         observation, info = env.reset()
#
# env.close()

gamma = 0.95

for ep_idx in range(100):
    if ep_idx % 5 == 0:
        print(f"Episode {ep_idx}, evaluating agent...")
        avg_reward = evaluate_agent(policy, env)
        print(f"Average reward: {avg_reward}")
        print()

    states, actions, rewards = collect_episode(policy, env)
    reinforce_update(policy, states, actions, rewards, last_step_to_use=100, gamma=gamma)

env = gym.make("Pendulum-v1", render_mode="human")
evaluate_agent(policy, env, num_episodes=2)

pass
