import datetime
import os
import random
import time
from collections import deque
from itertools import count
import hydra
import numpy as np
import torch
import torch.nn.functional as F
import wandb
from omegaconf import DictConfig, OmegaConf
from tensorboardX import SummaryWriter

from logger import Logger
from make_envs import make_env
from memory import Memory
from agent import make_agent
from utils import evaluate, eval_mode

torch.set_num_threads(2)


def get_args(cfg: DictConfig):
    cfg.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    cfg.hydra_base_dir = os.getcwd()
    print(OmegaConf.to_yaml(cfg))
    return cfg


@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig):
    args = get_args(cfg)
    wandb.init(project=args.env.name + '_rl', entity='iq-learn', config=args)
    wandb.tensorboard.patch(save=False, pytorch=True)

    # set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device)
    if device.type == 'cuda' and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    env_args = args.env

    env = make_env(args)
    eval_env = make_env(args)
    # Seed envs
    env.seed(args.seed)
    eval_env.seed(args.seed + 1)

    REPLAY_MEMORY = int(env_args.replay_mem)
    INITIAL_MEMORY = int(env_args.initial_mem)
    UPDATE_STEPS = int(env_args.update_steps)
    EPISODE_STEPS = int(env_args.eps_steps)
    EPISODE_WINDOW = int(env_args.eps_window)
    LEARN_STEPS = int(env_args.learn_steps)

    GAMMA = args.gamma
    BATCH = args.train.batch

    agent = make_agent(env, args)

    memory_replay = Memory(REPLAY_MEMORY, args.seed)

    ts_str = datetime.datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.join(args.log_dir, args.env.name, args.exp_name, ts_str)
    writer = SummaryWriter(log_dir=log_dir)
    print(f'--> Saving logs at: {log_dir}')
    # TODO: Fix logging
    logger = Logger(args.log_dir)

    steps = 0
    learn_steps = 0
    begin_learn = False

    # track avg. reward and scores
    rewards_window = deque(maxlen=EPISODE_WINDOW)  # last N rewards
    best_eval_returns = -np.inf

    for epoch in count():
        state = env.reset()
        episode_reward = 0
        done = False
        for episode_step in range(EPISODE_STEPS):
            if steps < args.num_seed_steps:
                action = env.action_space.sample()  # Sample random action
            else:
                with eval_mode(agent):
                    action = agent.choose_action(state, sample=True)
            next_state, reward, done, _ = env.step(action)
            episode_reward += reward
            steps += 1

            if learn_steps % args.env.eval_interval == 0:
                eval_returns, eval_timesteps = evaluate(agent, eval_env)
                returns = np.mean(eval_returns)
                learn_steps += 1  # To prevent repeated eval at timestep 0
                writer.add_scalar('Rewards/eval_rewards', returns,
                                  global_step=learn_steps)
                print('EVAL\tEp {}\tAverage reward: {:.2f}\t'.format(epoch, returns))

                if returns > best_eval_returns:
                    best_eval_returns = returns
                    wandb.run.summary["best_returns"] = best_eval_returns
                    save(agent, epoch, args, output_dir='results_best')

            # allow infinite bootstrap
            done_no_lim = done
            if str(env.__class__.__name__).find('TimeLimit') >= 0 and episode_step + 1 == env._max_episode_steps:
                done_no_lim = 0
            memory_replay.add((state, next_state, action, reward, done_no_lim))

            if memory_replay.size() > INITIAL_MEMORY:
                if begin_learn is False:
                    print('learn begin!')
                    begin_learn = True
                learn_steps += 1

                losses = agent.update(memory_replay, logger, learn_steps)

                if learn_steps % args.log_interval == 0:
                    for key, loss in losses.items():
                        writer.add_scalar(key, loss, global_step=learn_steps)

            if done:
                break

            state = next_state

        rewards_window.append(episode_reward)
        writer.add_scalar('Rewards/train_reward', np.mean(rewards_window), global_step=epoch)
        print('TRAIN\tEp {}\tAverage reward: {:.2f}\t'.format(epoch, np.mean(rewards_window)))
        save(agent, epoch, args, output_dir='results')


def save(agent, epoch, args, output_dir='results'):
    if epoch % args.save_interval == 0:
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        agent.save(f'{output_dir}/{args.agent.name}_{args.env.name}')


if __name__ == "__main__":
    main()
