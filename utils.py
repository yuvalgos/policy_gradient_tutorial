import io
import base64
from PIL import Image, ImageDraw, ImageFont
from IPython.display import HTML
import numpy as np
import torch
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from IPython.display import display, clear_output


def plot_video(images, fps=30):
    frames = [Image.fromarray(img) for img in images]

    buffer = io.BytesIO()

    # Add a frame counter to each image
    for i, frame in enumerate(frames):
        draw = ImageDraw.Draw(frame)
        font = ImageFont.truetype("arial.ttf", 16)
        draw.text((10, 10), f'step: {i%200}', font=font, fill=(42, 42, 42))

    frames[0].save(buffer, format='GIF', append_images=frames[1:], save_all=True, duration=1000 / fps, loop=0)
    encoded = base64.b64encode(buffer.getvalue()).decode('ascii')
    html = f'<img src="data:image/gif;base64,{encoded}" style="height:350px"/>'
    return HTML(html)


def plot_gaussian(mean, var):
    x = np.linspace(-2, 2, 100)
    plt.plot(x, stats.norm.pdf(x, mean, np.sqrt(var)))
    plt.show()


def evaluate_agent_episode(policy, env):
    total_reward = 0
    terminated, truncated = False, False
    observation, info = env.reset()
    while not terminated and not truncated:
        action = policy.get_action(observation)
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

    return total_reward


def evaluate_agent(policy, env, n_episodes=10):
    total_rewards = []
    for i in range(n_episodes):
        total_reward = evaluate_agent_episode(policy, env)
        total_rewards.append(total_reward)

    return np.mean(total_rewards)


def visualize_policy(policy, env, n_episodes=1):
    images = []
    for i in range(n_episodes):
        observation, info = env.reset()
        curr_episode_images = []
        terminated = False
        truncated = False
        while not terminated and not truncated:
            curr_episode_images.append(env.render())
            action = policy.get_action(observation)
            observation, reward, terminated, truncated, info = env.step(action)

        images = images + [np.zeros_like(curr_episode_images[0])]*15 + curr_episode_images

    return plot_video(images)


class ActivePlotter:
    def __init__(self, max_iteration, reward_range=(-1900, 0)):
        self.iterations = []
        self.mean_rewards = []
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(1, 1, 1)

        self.ax.set_xlabel('Episode')
        self.ax.set_ylabel('Mean Eval Reward')
        self.ax.set_title('Evaluation Results')
        self.ax.set_xlim([0, max_iteration])
        self.ax.set_ylim(reward_range)
        self.ax.grid(True)

        self.ax.plot(self.iterations, self.mean_rewards, 'b-')

    def update_plot(self, iteration, mean_reward):
        self.iterations.append(iteration)
        self.mean_rewards.append(mean_reward)

        self.ax.plot(self.iterations, self.mean_rewards, 'b-')
        display(self.fig)
        clear_output(wait=True)
        plt.pause(0.0001)

        # self.ax.clear()
        # self.ax.plot(self.iterations, self.mean_rewards, 'b-')
        # self.figure.canvas.draw()
        # plt.show()
        # plt.pause(0.001)  # Pause to display the plot

    def close(self):
        plt.close(self.figure)
