import gymnasium as gym
import numpy as np
import torch
from pg_tutorial.utils import evaluate_agent
from pg_tutorial.parameterized_policy import ParameterizedGaussianPolicy


def reinforce(env, policy, learning_rate=0.0002, gamma=0.99, n_episodes=1000, last_step_to_use=150):
    for episode in range(n_episodes):
        state, _ = env.reset()
        episode_reward = 0
        episode_states, episode_actions, episode_rewards, actions_log_likelihood_grads = [], [], [], []
        terminated = False
        truncated = False

        while (not terminated) and (not truncated):
            action, log_likelihood_grad = policy.sample_action(state)
            episode_states.append(state)
            episode_actions.append(action)
            actions_log_likelihood_grads.append(log_likelihood_grad)
            state, reward, terminated, truncated, info = env.step(action)
            episode_rewards.append(reward)
            episode_reward += reward

        # compute return for each step, from the end to the beginning:
        episode_returns = []
        current_return = 0
        for reward in episode_rewards[::-1]:
            current_return = reward + gamma * current_return
            episode_returns.insert(0, current_return)

        # normalize episode_returns, for stability:
        episode_returns = np.array(episode_returns)
        episode_returns = (episode_returns - episode_returns.mean()) / (episode_returns.std() + 1e-9)

        policy_parameters = policy.get_parameters_vector()
        # update the parameters according to the policy gradient:
        for i in range(len(episode_returns) - last_step_to_use,):
            current_return = episode_returns[i]
            log_likelihood_grad = actions_log_likelihood_grads[i]

            policy_parameters = policy_parameters + learning_rate * log_likelihood_grad * current_return

        policy.set_parameters_vector(policy_parameters)

        if episode % 20 == 0:
            # evaluate agent:
            mean_reward = evaluate_agent(policy, env, n_episodes=5)
            print(f"Episode {episode}: Evaluation mean accumulated reward = {mean_reward}")


if __name__ == "__main__":
    np.random.seed(2023)
    torch.manual_seed(2023)

    gravity = 1.0

    env = gym.make("Pendulum-v1", g=gravity)
    action_range = 2.
    policy = ParameterizedGaussianPolicy(env.observation_space.shape[0], env.action_space.shape[0], action_range)
    # print number of parameters in the policy:
    print(f"Number of parameters in the policy: {sum([np.prod(p.shape) for p in policy.parameters()])}")
    reinforce(env, policy, gamma=0.97, learning_rate=0.0005, n_episodes=750, last_step_to_use=150)

    for i in range(5):
        env = gym.make("Pendulum-v1", render_mode="human", g=gravity)
        state, _ = env.reset()
        terminated = False
        truncated = False
        while (not terminated) and (not truncated):
            action, _ = policy.sample_action(state)
            state, reward, terminated, truncated, info = env.step([action.item()])
            env.render()
