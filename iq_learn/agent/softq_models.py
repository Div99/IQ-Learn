import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
import torch.nn.functional as F
from torch.autograd import Variable, grad


class SoftQNetwork(nn.Module):
    def __init__(self, obs_dim, action_dim, args, device='cpu'):
        super(SoftQNetwork, self).__init__()
        self.args = args
        self.device = device
        self.tanh = nn.Tanh()

    def _forward(self, x, *args):
        return NotImplementedError

    def forward(self, x, both=False):
        if "DoubleQ" in self.args.q_net._target_:
            out = self._forward(x, both)
        else:
            out = self._forward(x)

        if self.args.method.tanh:
            return self.tanh(out) * 1/(1-self.args.gamma)
        return out

    def jacobian(self, outputs, inputs):
        """Computes the jacobian of outputs with respect to inputs

        :param outputs: tensor for the output of some function
        :param inputs: tensor for the input of some function (probably a vector)
        :returns: a tensor containing the jacobian of outputs with respect to inputs
        """
        batch_size, output_dim = outputs.shape
        jacobian = []
        for i in range(output_dim):
            v = torch.zeros_like(outputs)
            v[:, i] = 1.
            dy_i_dx = grad(outputs,
                           inputs,
                           grad_outputs=v,
                           retain_graph=True,
                           create_graph=True)[0]  # shape [B, N]
            jacobian.append(dy_i_dx)

        jacobian = torch.stack(jacobian, dim=-1).requires_grad_()
        return jacobian

    def grad_pen(self, obs1, action1, obs2, action2, lambda_=1):
        expert_data = obs1
        policy_data = obs2
        batch_size = expert_data.size()[0]

        # Calculate interpolation
        if expert_data.ndim == 4:
            alpha = torch.rand(batch_size, 1, 1, 1)  # B, C, H, W input
        else:
            alpha = torch.rand(batch_size, 1)
        alpha = alpha.expand_as(expert_data).to(expert_data.device)

        interpolated = alpha * expert_data + (1 - alpha) * policy_data
        interpolated = Variable(interpolated, requires_grad=True)
        interpolated = interpolated.to(expert_data.device)

        # Calculate probability of interpolated examples
        prob_interpolated = self.forward(interpolated)
        # Calculate gradients of probabilities with respect to examples
        gradients = self.jacobian(prob_interpolated, interpolated)

        # Gradients have shape (batch_size, input_dim, output_dim)
        out_size = gradients.shape[-1]
        gradients_norm = gradients.reshape([batch_size, -1, out_size]).norm(2, dim=1)

        # Return gradient penalty
        return lambda_ * ((gradients_norm - 1) ** 2).mean()


class SimpleQNetwork(SoftQNetwork):
    def __init__(self, obs_dim, action_dim, args, device='cpu'):
        super(SimpleQNetwork, self).__init__(obs_dim, action_dim, args, device)
        self.args = args
        self.fc1 = nn.Linear(obs_dim, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)

    def _forward(self, x, *args):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class OfflineQNetwork(SoftQNetwork):
    def __init__(self, obs_dim, action_dim, args, device='cpu'):
        super(OfflineQNetwork, self).__init__(obs_dim, action_dim, args, device)
        self.args = args
        self.fc1 = nn.Linear(obs_dim, 64)
        self.elu = nn.ELU()
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_dim)

    def _forward(self, x, *args):
        x = self.elu(self.fc1(x))
        x = self.elu(self.fc2(x))
        x = self.fc3(x)
        return x


class DoubleQNetwork(SoftQNetwork):
    def __init__(self, obs_dim, action_dim, args, device='cpu'):
        super(DoubleQNetwork, self).__init__(obs_dim, action_dim, args, device)
        self.args = args
        self.net1 = AtariQNetwork(obs_dim, action_dim, args, device)
        self.net2 = AtariQNetwork(obs_dim, action_dim, args, device)

    def _forward(self, x, both=False):
        q1 = self.net1.forward(x)
        q2 = self.net2.forward(x)

        if both:
            return q1, q2
        else:
            return torch.minimum(q1, q2)

class AtariQNetwork(SoftQNetwork):
    def __init__(self, obs_dim, action_dim, args, device='cpu', input_dim=(84, 84)):
        super(AtariQNetwork, self).__init__(obs_dim, action_dim, args, device)
        self.frames = 4
        self.n_outputs = action_dim

        # CNN modeled off of Mnih et al.
        self.cnn = nn.Sequential(
            nn.Conv2d(self.frames, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        self.fc_layer_inputs = self.cnn_out_dim(input_dim)

        self.fully_connected = nn.Sequential(
            nn.Linear(self.fc_layer_inputs, 512, bias=True),
            nn.ReLU(),
            nn.Linear(512, self.n_outputs))

    def cnn_out_dim(self, input_dim):
        return self.cnn(torch.zeros(1, self.frames, *input_dim)
                        ).flatten().shape[0]

    def _forward(self, x, *args):
        cnn_out = self.cnn(x).reshape(-1, self.fc_layer_inputs)
        return self.fully_connected(cnn_out)


class SimpleVNetwork(SoftQNetwork):
    def __init__(self, obs_dim, action_dim, args, device='cpu'):
        super(SimpleVNetwork, self).__init__(obs_dim, action_dim, args, device)
        self.args = args
        self.fc1 = nn.Linear(obs_dim, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 1)

    def _forward(self, x, *args):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x
