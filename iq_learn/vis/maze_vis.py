from omegaconf import DictConfig, OmegaConf
import torch
import hydra
import gym
from itertools import count
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import matplotlib
import os
import wandb

from utils.utils import evaluate
from agent import make_agent
from make_envs import make_env

matplotlib.use('Agg')


def get_args(cfg: DictConfig):
    cfg.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(OmegaConf.to_yaml(cfg))
    return cfg


@hydra.main(config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    args = get_args(cfg)

    GAMMA = args.gamma

    env = make_env(args)
    agent = make_agent(env, args)

    if args.method.type == "sqil":
        name = f'sqil'
    else:
        name = f'iq'

    policy_file = f'results'

    if args.eval.policy:
        policy_file = f'{args.eval.policy}'
    print(f'Loading policy from: {policy_file}')

    if args.eval.transfer:
        agent.load(hydra.utils.to_absolute_path(policy_file),
                   f'_{name}_{args.eval.expert_env}')
    else:
        agent.load(hydra.utils.to_absolute_path(policy_file), f'_{name}_{args.env.name}')

    # eval_returns, eval_timesteps = evaluate(agent, env, num_episodes=10)
    # print(f'Avg. eval returns: {eval_returns}, timesteps: {eval_timesteps}')

    visualize_reward(agent, env, args)

    for epoch in count():
        state = env.reset()
        episode_reward = 0
        episode_irl_reward = 0

        for time_steps in count():
            # env.render()
            action = agent.choose_action(state, sample=False)
            next_state, reward, done, _ = env.step(action)

            # Get sqil reward
            with torch.no_grad():
                q = agent.infer_q(state, action)
                next_v = agent.infer_v(next_state)
                y = (1 - done) * GAMMA * next_v
                irl_reward = q - y

            episode_irl_reward += irl_reward.item()
            episode_reward += reward
            if done:
                break
            state = next_state

        print('Ep {}\tMoving average score: {:.2f}\t'.format(epoch, episode_reward))
        print('Ep {}\tMoving Soft Q average score: {:.2f}\t'.format(epoch, episode_irl_reward))


def visualize_reward(agent, env, args, use_wandb=False):

    env = make_env(args, monitor=False)  # Using monitoring breaks setting the states

    grid_size = 0.005
    rescale = 1./grid_size
    boundary_low = -0.1
    boundary_high = 0.6
    barrier_range = [0.2, 0.6]
    barrier_y = 0.3

    for itr in range(5):
        state = env.reset()
        target = state[3:]

        obs_batch = []
        obs_action = []
        next_obs_batch = []

        num_y = 0
        for pos_y in np.arange(boundary_low, boundary_high, grid_size):
            num_y += 1
            num_x = 0
            for pos_x in np.arange(boundary_low, boundary_high, grid_size):
                num_x += 1
                obs_batch.append([pos_x, pos_y, 0.])

                state = np.concatenate(
                    [[pos_x, pos_y, 0.], target], axis=0)
                env.make_state(state)
                action = agent.choose_action(state, sample=False)
                next_state, reward, done, _ = env.step(action)

                obs_action.append(action)
                next_obs_batch.append(next_state[:3])

        obs_batch = np.array(obs_batch)
        next_obs_batch = np.array(next_obs_batch)
        obs_action = np.array(obs_action)

        target_batch = np.repeat([target], obs_batch.shape[0], axis=0)
        obs_batch = np.concatenate([obs_batch, target_batch], axis=1)
        next_obs_batch = np.concatenate([next_obs_batch, target_batch], axis=1)

        # Get sqil reward
        with torch.no_grad():
            state = torch.FloatTensor(obs_batch).to(agent.device)
            action = torch.FloatTensor(obs_action).to(agent.device)
            next_state = torch.FloatTensor(next_obs_batch).to(agent.device)

        done = False

        with torch.no_grad():
            q = agent.critic(state, action)

            next_v = agent.getV(next_state)
            y = (1 - done) * args.gamma * next_v
            irl_reward = q - y

            irl_reward = -irl_reward.cpu().numpy()
            print(irl_reward.shape)

        score = irl_reward
        ax = sns.heatmap(score.reshape([num_x, num_y]), cmap="YlGnBu_r")
        ax.scatter((target[0]-boundary_low)*rescale, (target[1]-boundary_low)
                   * rescale, marker='*', s=150, c='r', edgecolors='k', linewidths=0.5)
        ax.scatter((0.3-boundary_low + np.random.uniform(low=-0.05, high=0.05))*rescale, (0.-boundary_low +
                                                                                          np.random.uniform(low=-0.05, high=0.05))*rescale, marker='o', s=120, c='white', linewidths=0.5, edgecolors='k')
        ax.plot([(barrier_range[0] - boundary_low)*rescale, (barrier_range[1] - boundary_low)*rescale], [(barrier_y - boundary_low)*rescale, (barrier_y - boundary_low)*rescale],
                color='k', linewidth=10)
        ax.invert_yaxis()
        plt.axis('off')
        # plt.show()

        if use_wandb:
            wandb.log({f"rewards_{itr}": wandb.Image(plt)})

        savedir = hydra.utils.to_absolute_path('vis/outputs/transfer')
        if not os.path.exists(savedir):
            os.makedirs(savedir)

        plt.savefig(savedir + '/%s.png' % itr)
        print('Save Itr', itr)
        plt.close()


if __name__ == '__main__':
    main()
