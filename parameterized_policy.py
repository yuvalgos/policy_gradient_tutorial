import random
import torch


class ParameterizedGaussianPolicy(torch.nn.Module):
    def __init__(self, input_size, output_size, action_scale=2, hidden_layers=(64,)):
        super(ParameterizedGaussianPolicy, self).__init__()

        self.action_scale = action_scale

        self.layer_sizes = (input_size,) + hidden_layers
        self.layers = []
        for i in range(len(self.layer_sizes) - 1):
            self.layers.append(torch.nn.Linear(self.layer_sizes[i], self.layer_sizes[i + 1]))
            self.layers.append(torch.nn.Tanh())
        self.layers = torch.nn.Sequential(*self.layers)

        self.mean_layer = torch.nn.Linear(self.layer_sizes[-1], output_size)
        # self.log_std_layer = torch.nn.Linear(self.layer_sizes[-1], output_size)

        self.mean_tanh = torch.nn.Tanh()

    def forward(self, x):
        z = self.layers(x)
        mean = self.mean_layer(z)
        # log_std = self.log_std_layer(z)
        # log_std = torch.clamp(log_std, min=-10, max=1)
        log_std = torch.tensor([-1])
        mean = self.mean_tanh(mean) * self.action_scale

        # n = random.random()
        # if n > 0.95:
        #     print(f"mean: {mean}, log_std: {log_std}")

        return mean, log_std

    def sample_action(self, state):
        with torch.no_grad():
            state = torch.tensor(state).unsqueeze(0)
            mean, log_std = self(state)
            std = torch.exp(log_std)
            action = torch.normal(mean, std)
        return action.squeeze(0).numpy()

    def get_parameters_vector(self):
        return torch.nn.utils.parameters_to_vector(self.parameters())

    def set_parameters_vector(self, parameters_vector):
        torch.nn.utils.vector_to_parameters(parameters_vector, self.parameters())

    def get_grad_log_likelihood(self, state, action):
        state = torch.tensor(state).unsqueeze(0)
        action = torch.tensor(action).unsqueeze(0)

        mean, log_std = self(state)
        std = torch.exp(log_std)

        log_likelihood = torch.distributions.Normal(mean, std).log_prob(action).sum()
        grad_log_likelihood = torch.autograd.grad(log_likelihood, self.parameters())

        # return grad as one vecotr corresponding to all parameters
        return torch.nn.utils.parameters_to_vector(grad_log_likelihood)
