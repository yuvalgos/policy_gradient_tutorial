import numpy as np
from matplotlib import pyplot as plt
from pg_tutorial.utils import evaluate_agent


def visualize_parameter_space(policy, env, thetas_range=20, n_thetas=20):
    theta_1_original = policy.theta_1
    theta_2_original = policy.theta_2

    theta_1_values = np.linspace(theta_1_original - thetas_range, theta_1_original + thetas_range, n_thetas)
    theta_2_values = np.linspace(theta_2_original - thetas_range, theta_2_original + thetas_range, n_thetas)
    theta_3 = policy.theta_3

    rewards = np.zeros((n_thetas, n_thetas))
    for i, theta_1 in enumerate(theta_1_values):
        for j, theta_2 in enumerate(theta_2_values):
            policy.theta_1 = theta_1
            policy.theta_2 = theta_2
            rewards[i, j] = evaluate_agent(policy, env, n_episodes=3)

    # plot the results:
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    theta_1_grid, theta_2_grid = np.meshgrid(theta_1_values, theta_2_values)
    # opacity 0.5
    ax.plot_surface(theta_1_grid, theta_2_grid, rewards, cmap='viridis', alpha=0.7)
    # plot original thetas point, make sure it's visible:
    ax.scatter(theta_1_original, theta_2_original, evaluate_agent(policy, env, n_episodes=3), color='red', s=100)

    ax.set_xlabel(r'$\theta_1}$')
    ax.set_ylabel(r'$\theta_2$')
    ax.set_zlabel('Mean Eval Reward')
    ax.set_title('Parameter Space Visualization')
    plt.show()
