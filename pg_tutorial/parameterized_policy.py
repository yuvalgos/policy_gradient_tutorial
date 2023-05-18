import random
import torch
from torch.distributions import Normal
import torch.nn as nn


class ParameterizedGaussianPolicy(torch.nn.Module):
    def __init__(self, state_dim, action_dim, action_range):
        super(ParameterizedGaussianPolicy, self).__init__()
        self.hidden1 = nn.Linear(state_dim, 128)
        self.hidden2 = nn.Linear(128, 64)
        self.mu_layer = nn.Linear(64, action_dim)
        self.sigma_layer = nn.Linear(64, action_dim)
        self.action_range = action_range
        self.softplus = nn.Softplus()

    def forward(self, state):
        x = torch.relu(self.hidden1(state))
        x = torch.relu(self.hidden2(x))
        mu = torch.tanh(self.mu_layer(x)) * self.action_range

        sigma = self.softplus(self.sigma_layer(x)) + 1e-5
        return mu, sigma

    # none torch methods to abstract the policy for the user:
    def get_action_distribution(self, state):
        """ return the expectation and variance of a gaussian action """
        state = torch.FloatTensor(state)
        with torch.no_grad():
            mu, sigma = self(state)
        return mu.numpy(), sigma.numpy()

    def sample_action(self, state):
        """
        sample an action for a given state, return the action,
         and the gradient of the log probability of the action, as one vector similar to the get_parameters vector
        """
        state = torch.FloatTensor(state)

        mu, sigma = self(state)
        dist = Normal(mu, sigma)
        action = dist.sample()

        log_likelihood = dist.log_prob(action)
        log_likelihood_grad = torch.autograd.grad(log_likelihood, self.parameters())
        log_likelihood_grad = torch.nn.utils.parameters_to_vector(log_likelihood_grad)

        return action.clamp(-self.action_range, self.action_range).numpy(), log_likelihood_grad.numpy()

    def get_action(self, state):
        """ for evaluation only, shouldn't be explicitly used in the tutorial"""
        state = torch.FloatTensor(state)
        with torch.no_grad():
            mu, sigma = self(state)
            dist = Normal(mu, sigma)
            action = dist.sample()
        return action.clamp(-self.action_range, self.action_range).numpy()

    def get_parameters_vector(self):
        """ return the parameters of the policy as one vector """
        return torch.nn.utils.parameters_to_vector(self.parameters()).detach().numpy()

    def set_parameters_vector(self, parameters_vector):
        """ set the parameters of the policy from a vector similar to the get_parameters vector """
        torch.nn.utils.vector_to_parameters(torch.FloatTensor(parameters_vector), self.parameters())
