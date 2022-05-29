import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch import distributions as pyd
from torch.autograd import Variable, grad

import utils.utils as utils

# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


def orthogonal_init_(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)


class DoubleQCritic(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim, hidden_depth, args):
        super(DoubleQCritic, self).__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.args = args

        # Q1 architecture
        self.Q1 = utils.mlp(obs_dim + action_dim, hidden_dim, 1, hidden_depth)

        # Q2 architecture
        self.Q2 = utils.mlp(obs_dim + action_dim, hidden_dim, 1, hidden_depth)

        self.apply(orthogonal_init_)

    def forward(self, obs, action, both=False):
        assert obs.size(0) == action.size(0)

        obs_action = torch.cat([obs, action], dim=-1)
        q1 = self.Q1(obs_action)
        q2 = self.Q2(obs_action)

        if self.args.method.tanh:
            q1 = torch.tanh(q1) * 1/(1-self.args.gamma)
            q2 = torch.tanh(q2) * 1/(1-self.args.gamma)

        if both:
            return q1, q2
        else:
            return torch.min(q1, q2)

    def grad_pen(self, obs1, action1, obs2, action2, lambda_=1):
        expert_data = torch.cat([obs1, action1], 1)
        policy_data = torch.cat([obs2, action2], 1)

        alpha = torch.rand(expert_data.size()[0], 1)
        alpha = alpha.expand_as(expert_data).to(expert_data.device)

        interpolated = alpha * expert_data + (1 - alpha) * policy_data
        interpolated = Variable(interpolated, requires_grad=True)

        interpolated_state, interpolated_action = torch.split(
            interpolated, [self.obs_dim, self.action_dim], dim=1)
        q = self.forward(interpolated_state, interpolated_action, both=True)
        ones = torch.ones(q[0].size()).to(policy_data.device)
        gradient = grad(
            outputs=q,
            inputs=interpolated,
            grad_outputs=[ones, ones],
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        grad_pen = lambda_ * (gradient.norm(2, dim=1) - 1).pow(2).mean()
        return grad_pen


class DoubleQCriticMax(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim, hidden_depth, args):
        super(DoubleQCriticMax, self).__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.args = args

        # Q1 architecture
        self.Q1 = utils.mlp(obs_dim + action_dim, hidden_dim, 1, hidden_depth)

        # Q2 architecture
        self.Q2 = utils.mlp(obs_dim + action_dim, hidden_dim, 1, hidden_depth)

        self.apply(orthogonal_init_)

    def forward(self, obs, action, both=False):
        assert obs.size(0) == action.size(0)

        obs_action = torch.cat([obs, action], dim=-1)
        q1 = self.Q1(obs_action)
        q2 = self.Q2(obs_action)

        if self.args.method.tanh:
            q1 = torch.tanh(q1) * 1/(1-self.args.gamma)
            q2 = torch.tanh(q2) * 1/(1-self.args.gamma)

        if both:
            return q1, q2
        else:
            return torch.max(q1, q2)


class SingleQCritic(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim, hidden_depth, args):
        super(SingleQCritic, self).__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.args = args

        # Q architecture
        self.Q = utils.mlp(obs_dim + action_dim, hidden_dim, 1, hidden_depth)

        self.apply(orthogonal_init_)

    def forward(self, obs, action):
        assert obs.size(0) == action.size(0)

        obs_action = torch.cat([obs, action], dim=-1)
        q = self.Q(obs_action)

        if self.args.method.tanh:
            q = torch.tanh(q) * 1/(1-self.args.gamma)

        return q

    def grad_pen(self, obs1, action1, obs2, action2, lambda_=1):
        expert_data = torch.cat([obs1, action1], 1)
        policy_data = torch.cat([obs2, action2], 1)

        alpha = torch.rand(expert_data.size()[0], 1)
        alpha = alpha.expand_as(expert_data).to(expert_data.device)

        interpolated = alpha * expert_data + (1 - alpha) * policy_data
        interpolated = Variable(interpolated, requires_grad=True)

        interpolated_state, interpolated_action = torch.split(
            interpolated, [self.obs_dim, self.action_dim], dim=1)
        q = self.forward(interpolated_state, interpolated_action)
        ones = torch.ones(q.size()).to(policy_data.device)
        gradient = grad(
            outputs=q,
            inputs=interpolated,
            grad_outputs=ones,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        grad_pen = lambda_ * (gradient.norm(2, dim=1) - 1).pow(2).mean()
        return grad_pen


class DoubleQCriticState(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim, hidden_depth, args):
        super(DoubleQCritic, self).__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.args = args

        # Q1 architecture
        self.Q1 = utils.mlp(obs_dim, hidden_dim, 1, hidden_depth)

        # Q2 architecture
        self.Q2 = utils.mlp(obs_dim, hidden_dim, 1, hidden_depth)

        self.apply(orthogonal_init_)

    def forward(self, obs, action, both=False):
        assert obs.size(0) == action.size(0)

        q1 = self.Q1(obs)
        q2 = self.Q2(obs)

        if self.args.method.tanh:
            q1 = torch.tanh(q1) * 1/(1-self.args.gamma)
            q2 = torch.tanh(q2) * 1/(1-self.args.gamma)

        if both:
            return q1, q2
        else:
            return torch.min(q1, q2)

    def grad_pen(self, obs1, action1, obs2, action2, lambda_=1):
        expert_data = obs1
        policy_data = obs2

        alpha = torch.rand(expert_data.size()[0], 1)
        alpha = alpha.expand_as(expert_data).to(expert_data.device)

        interpolated = alpha * expert_data + (1 - alpha) * policy_data
        interpolated = Variable(interpolated, requires_grad=True)

        interpolated_state, interpolated_action = torch.split(
            interpolated, [self.obs_dim, self.action_dim], dim=1)
        q = self.forward(interpolated_state, interpolated_action)
        ones = torch.ones(q[0].size()).to(policy_data.device)
        gradient = grad(
            outputs=q,
            inputs=interpolated,
            grad_outputs=[ones, ones],
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        grad_pen = lambda_ * (gradient.norm(2, dim=1) - 1).pow(2).mean()
        return grad_pen

class TanhTransform(pyd.transforms.Transform):
    domain = pyd.constraints.real
    codomain = pyd.constraints.interval(-1.0, 1.0)
    bijective = True
    sign = +1

    def __init__(self, cache_size=1):
        super().__init__(cache_size=cache_size)

    @staticmethod
    def atanh(x):
        return 0.5 * (x.log1p() - (-x).log1p())

    def __eq__(self, other):
        return isinstance(other, TanhTransform)

    def _call(self, x):
        return x.tanh()

    def _inverse(self, y):
        # We do not clamp to the boundary here as it may degrade the performance of certain algorithms.
        # one should use `cache_size=1` instead
        return self.atanh(y)

    def log_abs_det_jacobian(self, x, y):
        # We use a formula that is more numerically stable, see details in the following link
        # https://github.com/tensorflow/probability/commit/ef6bb176e0ebd1cf6e25c6b5cecdd2428c22963f#diff-e120f70e92e6741bca649f04fcd907b7
        return 2. * (math.log(2.) - x - F.softplus(-2. * x))


class SquashedNormal(pyd.transformed_distribution.TransformedDistribution):
    def __init__(self, loc, scale):
        self.loc = loc
        self.scale = scale

        self.base_dist = pyd.Normal(loc, scale)
        transforms = [TanhTransform()]
        super().__init__(self.base_dist, transforms)

    @property
    def mean(self):
        mu = self.loc
        for tr in self.transforms:
            mu = tr(mu)
        return mu


class DiagGaussianActor(nn.Module):
    """torch.distributions implementation of an diagonal Gaussian policy."""

    def __init__(self, obs_dim, action_dim, hidden_dim, hidden_depth,
                 log_std_bounds):
        super().__init__()

        self.log_std_bounds = log_std_bounds
        self.trunk = utils.mlp(obs_dim, hidden_dim, 2 * action_dim,
                               hidden_depth)

        self.outputs = dict()
        self.apply(orthogonal_init_)

    def forward(self, obs):
        mu, log_std = self.trunk(obs).chunk(2, dim=-1)

        # constrain log_std inside [log_std_min, log_std_max]
        log_std = torch.tanh(log_std)
        log_std_min, log_std_max = self.log_std_bounds
        log_std = log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std + 1)

        std = log_std.exp()

        # self.outputs['mu'] = mu
        # self.outputs['std'] = std

        dist = SquashedNormal(mu, std)
        return dist

    def sample(self, obs):
        dist = self.forward(obs)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)

        return action, log_prob, dist.mean
