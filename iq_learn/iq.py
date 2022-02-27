"""
Copyright 2022 Div Garg. All rights reserved.
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

# Full IQ-Learn objective with other divergences and options
def iq_loss(agent, current_Q, current_v, next_v, batch):
    args = agent.args
    gamma = agent.gamma
    obs, next_obs, action, env_reward, done, is_expert = batch

    loss_dict = {}
    # keep track of v0
    v0 = agent.getV(obs[is_expert.squeeze(1), ...]).mean()
    loss_dict['v0'] = v0.item()

    if args.method.type == "sqil":
        target_Q = env_reward + (1 - done) * gamma * next_v
        bell_error = F.mse_loss(current_Q, target_Q, reduction='none')

        loss = (bell_error[is_expert]).mean() + \
            args.method.sqil_lmbda * (bell_error[~is_expert]).mean()
        loss_dict['sqil_loss'] = loss.item()

    elif args.method.type == "iq":
        # our method, calculate 1st term of loss
        #  -E_(ρ_expert)[Q(s, a) - γV(s')]
        y = (1 - done) * gamma * next_v
        reward = (current_Q - y)[is_expert]

        with torch.no_grad():
            if args.method.div == "hellinger":
                phi_grad = 1/(1+reward)**2
            elif args.method.div == "kl":
                phi_grad = torch.exp(-reward-1)
            elif args.method.div == "kl2":
                phi_grad = F.softmax(-reward, dim=0) * reward.shape[0]
            elif args.method.div == "kl_fix":
                phi_grad = torch.exp(-reward)
            elif args.method.div == "js":
                phi_grad = torch.exp(-reward)/(2 - torch.exp(-reward))
            else:
                phi_grad = 1
        loss = -(phi_grad * reward).mean()
        loss_dict['softq_loss'] = loss.item()

        if args.method.loss == "v0":
            # calculate 2nd term for our loss
            # (1-γ)E_(ρ0)[V(s0)]
            v0_loss = (1 - gamma) * v0
            loss += v0_loss
            loss_dict['v0_loss'] = v0_loss.item()

        elif args.method.loss == "value":
            # alternative 2nd term for our loss (use expert and policy states)
            # E_(ρ)[V(s) - γV(s')]
            value_loss = (current_v - y).mean()
            loss += value_loss
            loss_dict['value_loss'] = value_loss.item()

        elif args.method.loss == "value_policy":
            # alternative 2nd term for our loss (use only policy states)
            # E_(ρ)[V(s) - γV(s')]
            value_loss = (current_v - y)[~is_expert].mean()
            loss += value_loss
            loss_dict['value_policy_loss'] = value_loss.item()

        elif args.method.loss == "value_expert":
            # alternative 2nd term for our loss (use only expert states)
            # E_(ρ)[Q(s,a) - γV(s')]
            value_loss = (current_v - y)[is_expert].mean()
            loss += value_loss
            loss_dict['value_loss'] = value_loss.item()

        elif args.method.loss == "value_mix":
            # alternative 2nd term for our loss (use expert and policy states)
            # E_(ρ)[Q(s,a) - γV(s')]
            w = args.method.mix_coeff
            value_loss = (w * (current_v - y)[is_expert] +
                        (1-w) * (current_v - y)[~is_expert]).mean()
            loss += value_loss
            loss_dict['value_loss'] = value_loss.item()

        elif args.method.loss == "skip":
            # No loss
            pass
    else:
        raise ValueError(f'This method is not implemented: {args.method.type}')

    if args.method.grad_pen:
        # add a gradient penalty to loss (W1 metric)
        gp_loss = agent.critic_net.grad_pen(obs[is_expert.squeeze(1), ...],
                                        action[is_expert.squeeze(1), ...],
                                        obs[~is_expert.squeeze(1), ...],
                                        action[~is_expert.squeeze(1), ...],
                                        args.method.lambda_gp)
        loss_dict['gp_loss'] = gp_loss.item()
        loss += gp_loss

    if args.method.div == "chi" or args.method.chi:  # TODO: Deprecate method.chi argument for method.div
        # Use χ2 divergence (adds a extra term to the loss)
        y = (1 - done) * gamma * next_v

        reward = current_Q - y
        chi2_loss = 1/(4 * args.method.alpha) * (reward**2)[is_expert].mean()
        loss += chi2_loss
        loss_dict['chi2_loss'] = chi2_loss.item()

    if args.method.regularize:
        y = (1 - done) * gamma * next_v

        reward = current_Q - y
        chi2_loss = 1/(4 * args.method.alpha) * (reward**2).mean()
        loss += chi2_loss
        loss_dict['regularize_loss'] = chi2_loss.item()

    loss_dict['total_loss'] = loss.item()
    return loss, loss_dict
