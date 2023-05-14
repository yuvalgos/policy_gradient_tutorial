import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from utils import evaluate_agent



class Policy(nn.Module):
    def __init__(self, state_dim, action_dim, action_range):
        super(Policy, self).__init__()
        self.hidden1 = nn.Linear(state_dim, 128)
        self.hidden2 = nn.Linear(128, 64)
        self.mu_layer = nn.Linear(64, action_dim)
        self.sigma_layer = nn.Linear(64, action_dim)
        self.action_range = action_range

    def forward(self, state):
        x = torch.relu(self.hidden1(state))
        x = torch.relu(self.hidden2(x))
        mu = torch.tanh(self.mu_layer(x)) * self.action_range
        sp = torch.nn.Softplus()
        sigma = sp(self.sigma_layer(x)) + 1e-5
        return mu, sigma

    def act(self, state):
        mu, sigma = self.forward(state)
        dist = Normal(mu, sigma)
        action = dist.sample()
        return action.clamp(-self.action_range, self.action_range), dist.log_prob(action)


def reinforce(env, policy, optimizer, gamma=0.99, n_episodes=1000, last_step_to_use=150):
    for episode in range(n_episodes):
        state, _ = env.reset()
        episode_reward = 0
        episode_states, episode_actions, episode_rewards, episode_log_probs = [], [], [], []
        terminated = False
        truncated = False

        while (not terminated) and (not truncated):
            action, log_prob = policy.act(torch.FloatTensor(state).unsqueeze(0))
            episode_states.append(state)
            episode_actions.append(action)
            episode_log_probs.append(log_prob)
            state, reward, terminated, truncated, info = env.step([action.item()])
            episode_rewards.append(reward)
            episode_reward += reward

        G = 0
        returns = []
        for r in episode_rewards[::-1]:
            G = r + gamma * G
            returns.insert(0, G)

        returns = torch.FloatTensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)
        policy_loss = []
        for log_prob, G in zip(episode_log_probs[:last_step_to_use], returns[:last_step_to_use]):
            policy_loss.append(-log_prob * G)
        policy_loss = torch.stack(policy_loss).sum()

        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()

        if episode % 10 == 0:
            # evaluate agent:
            print(f"Episode {episode}: Total Reward = {episode_reward}")


if __name__ == "__main__":
    gravity = 2.0

    env = gym.make("Pendulum-v1", render_mode="human", g=gravity)
    action_range = 2.
    policy = Policy(env.observation_space.shape[0], env.action_space.shape[0], action_range)
    # print number of parameters in the policy:
    print(f"Number of parameters in the policy: {sum([np.prod(p.shape) for p in policy.parameters()])}")
    optimizer = optim.SGD(policy.parameters(), lr=2e-4)
    reinforce(env, policy, optimizer, gamma=0.97, n_episodes=1000, last_step_to_use=150)

    for i in range(5):
        env = gym.make("Pendulum-v1", render_mode="human", g=gravity)
        state, _ = env.reset()
        terminated = False
        truncated = False
        while (not terminated) and (not truncated):
            action, _ = policy.act(torch.FloatTensor([state]))
            state, reward, terminated, truncated, info = env.step([action.item()])
            env.render()
