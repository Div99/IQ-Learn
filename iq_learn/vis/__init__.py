from vis import grid_vis
from vis import maze_vis


def visualize_reward(agent, env, args, use_wandb=True):
    if args.env.name in ["Grid", "GridDet"]:
        return grid_vis.visualize_reward(agent, env, args, use_wandb)

    if args.env.name in ["PointMazeLeft-v0", "PointMazeRight-v0"]:
        return maze_vis.visualize_reward(agent, env, args, use_wandb)
