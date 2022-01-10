from itertools import count

import hydra
import torch
import numpy as np
import os
from omegaconf import DictConfig, OmegaConf
from scipy.stats import spearmanr, pearsonr
import matplotlib.pyplot as plt
import seaborn as sns
import wandb

from make_envs import make_env
from agent import make_agent
from utils.utils import evaluate

def get_args(cfg: DictConfig):
    cfg.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(OmegaConf.to_yaml(cfg))
    return cfg


@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig):
    args = get_args(cfg)

    env = make_env(args)
    agent = make_agent(env, args)

    if args.method.type == "sqil":
        name = f'sqil'
    else:
        name = f'iq'

    policy_file = f'results/{args.method.type}.para'
    if args.eval.policy:
        policy_file = f'{args.eval.policy}'
    print(f'Loading policy from: {policy_file}')

    if args.eval.transfer:
        agent.load(hydra.utils.to_absolute_path(policy_file),
                   f'_{name}_{args.eval.expert_env}')
    else:
        agent.load(hydra.utils.to_absolute_path(policy_file), f'_{name}_{args.env.name}')

    eval_returns, eval_timesteps = evaluate(agent, env, num_episodes=args.eval.eps)
    print(f'Avg. eval returns: {np.mean(eval_returns)}, timesteps: {np.mean(eval_timesteps)}')
    if args.eval_only:
        exit()

    measure_correlations(agent, env, args, log=True)


def measure_correlations(agent, env, args, log=False, use_wandb=False):
    GAMMA = args.gamma

    env_rewards = []
    learnt_rewards = []

    for epoch in range(100):

        part_env_rewards = []
        part_learnt_rewards = []

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
                irl_reward = (q - y)

            episode_irl_reward += irl_reward.item()
            episode_reward += reward
            part_learnt_rewards.append(irl_reward.item())
            part_env_rewards.append(reward)

            if done:
                break
            state = next_state

        if log:
            print('Ep {}\tEpisode env rewards: {:.2f}\t'.format(epoch, episode_reward))
            print('Ep {}\tEpisode learnt rewards {:.2f}\t'.format(epoch, episode_irl_reward))

        learnt_rewards.append(part_learnt_rewards)
        env_rewards.append(part_env_rewards)

    # mask = [sum(x) < -5 for x in env_rewards]  # skip outliers
    # env_rewards = [env_rewards[i] for i in range(len(env_rewards)) if mask[i]]
    # learnt_rewards = [learnt_rewards[i] for i in range(len(learnt_rewards)) if mask[i]]

    print(f'Spearman correlation {spearmanr(eps(learnt_rewards), eps(env_rewards))}')
    print(f'Pearson correlation: {pearsonr(eps(learnt_rewards), eps(env_rewards))}')

    # plt.show()
    savedir = hydra.utils.to_absolute_path(f'vis/{args.env.name}/correlation')
    if not os.path.exists(savedir):
        os.makedirs(savedir)

    sns.set()
    plt.figure(dpi=150)
    plt.scatter(eps(env_rewards), eps(learnt_rewards), s=10, alpha=0.8)
    plt.xlabel('Env rewards')
    plt.ylabel('Recovered rewards')
    if use_wandb:
        wandb.log({f"Episode rewards": wandb.Image(plt)})
    plt.savefig(savedir + '/%s.png' % 'Episode rewards')
    plt.close()

    sns.set()
    plt.figure(dpi=150)
    for i in range(20):
        plt.scatter(part_eps(env_rewards)[i], part_eps(learnt_rewards)[i], s=5, alpha=0.6)
    plt.xlabel('Env rewards')
    plt.ylabel('Recovered rewards')
    if use_wandb:
        wandb.log({f"Partial rewards": wandb.Image(plt)})
    plt.savefig(savedir + '/%s.png' % 'Partial rewards')
    plt.close()

    sns.set()
    plt.figure(dpi=150)
    for i in range(20):
        plt.plot(part_eps(env_rewards)[i], part_eps(learnt_rewards)[i], markersize=1, alpha=0.8)
    plt.xlabel('Env rewards')
    plt.ylabel('Recovered rewards')
    if use_wandb:
        wandb.log({f"Partial rewards - Interplolate": wandb.Image(plt)})
    plt.savefig(savedir + '/%s.png' % 'Partial rewards - Interplolate')
    plt.close()

    sns.set()
    plt.figure(dpi=150)
    for i in range(5):
        plt.scatter(env_rewards[i], learnt_rewards[i], s=5, alpha=0.5)
    plt.xlabel('Env rewards')
    plt.ylabel('Recovered rewards')
    if use_wandb:
        wandb.log({f"Step rewards": wandb.Image(plt)})
    plt.savefig(savedir + '/%s.png' % 'Step rewards')
    plt.close()


def eps(rewards):
    return [sum(x) for x in rewards]


def part_eps(rewards):
    return [np.cumsum(x) for x in rewards]


if __name__ == '__main__':
    main()
