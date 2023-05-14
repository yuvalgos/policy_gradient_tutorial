import io
import base64
from PIL import Image, ImageDraw, ImageFont
from IPython.display import HTML


def plot_video(images, fps=30):
    # Create a list of PIL Image objects
    frames = [Image.fromarray(img) for img in images]

    # Create a buffer to hold the encoded frames
    buffer = io.BytesIO()

    # Add a frame counter to each image
    for i, frame in enumerate(frames):
        draw = ImageDraw.Draw(frame)
        font = ImageFont.truetype("arial.ttf", 16)
        draw.text((10, 10), f'step: {i}', font=font, fill=(125, 125, 125))

    # Save the frames as an animated GIF
    frames[0].save(buffer, format='GIF', append_images=frames[1:], save_all=True, duration=1000 / fps, loop=0)

    # Get the encoded GIF data
    encoded = base64.b64encode(buffer.getvalue()).decode('ascii')

    # Create an HTML tag to display the GIF
    html = f'<img src="data:image/gif;base64,{encoded}" />'

    # Return the HTML tag
    return HTML(html)

########################## old garbage:

import numpy as np
import torch

from parameterized_policy import ParameterizedGaussianPolicy


def normalize_pendulum_obs(obs):
    obs[2] /= 8
    return obs


def collect_episode(policy: ParameterizedGaussianPolicy, env):
    states = []
    actions = []
    rewards = []

    observation, info = env.reset()
    observation = normalize_pendulum_obs(observation)
    terminated = False
    truncated = False
    while not terminated and not truncated:
        states.append(observation)
        action = policy.sample_action(observation)
        actions.append(action)
        observation, reward, terminated, truncated, info = env.step(action)
        observation = normalize_pendulum_obs(observation)
        rewards.append(reward)

    return states, actions, rewards


def evaluate_agent(policy: ParameterizedGaussianPolicy, env, num_episodes=10):
    rewards = []
    for ep_idx in range(num_episodes):
        _, _, ep_rewards = collect_episode(policy, env)
        rewards.append(sum(ep_rewards))
    return sum(rewards) / len(rewards)


def get_discounted_returns(rewards, gamma):
    discounted_returns = [rewards[-1]]
    # compute from the end
    for t in range(len(rewards) - 2, -1, -1):
        next_step_return = discounted_returns[0]
        discounted_returns.insert(0, rewards[t] + gamma * next_step_return)

    return discounted_returns


def normalize_returns(returns):
    returns = np.array(returns)
    min_return = -16.2736044 * 200
    returns = (returns / min_return) * 2 + 1
    return returns.tolist()


def reinforce_update(policy: ParameterizedGaussianPolicy, states, actions, rewards, last_step_to_use=None, gamma=0.9):
    if last_step_to_use is None:
        last_step_to_use = len(states) - 1
    last_step_to_use = min(last_step_to_use, len(states) - 1)

    lr = 0.001

    discounted_returns = get_discounted_returns(rewards, gamma)  # used as value
    discounted_returns = normalize_returns(discounted_returns)
    parameters = policy.get_parameters_vector()
    for t in range(last_step_to_use + 1):
        state = states[t]
        action = actions[t]
        discounted_return = discounted_returns[t]
        reward = rewards[t]

        # zero grad:
        policy.zero_grad()
        # compute gradient of log likelihood
        grad_log_likelihood = policy.get_grad_log_likelihood(state, action)

        # update parameters
        parameters = parameters + lr * reward * grad_log_likelihood

        # print step norm:
        # step = lr * discounted_return * grad_log_likelihood
        # print(torch.norm(step))
        # print(f"grad_log_likelihood_norm: {torch.norm(grad_log_likelihood)}")
    print(f"model_parameters_mean: {torch.mean(parameters)}")
    print(f"model_parameters_std: {torch.std(parameters)}")

    policy.set_parameters_vector(parameters)
