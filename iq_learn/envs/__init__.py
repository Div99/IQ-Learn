from gym.envs.registration import register


def register_custom_envs():
    register(id='PointMazeRight-v0', entry_point='envs.point_maze_env:PointMazeEnv',
             kwargs={'sparse_reward': False, 'direction': 1, 'discrete': True},
             max_episode_steps=100)
    register(id='PointMazeLeft-v0', entry_point='envs.point_maze_env:PointMazeEnv',
                kwargs={'sparse_reward': False, 'direction': 0, 'discrete': True},
                max_episode_steps=100)
    register(id='PointMazeRightCont-v0', entry_point='envs.point_maze_env:PointMazeEnv',
                kwargs={'sparse_reward': False, 'direction': 1, 'discrete': False},
                max_episode_steps=100)
    register(id='PointMazeLeftCont-v0', entry_point='envs.point_maze_env:PointMazeEnv',
                kwargs={'sparse_reward': False, 'direction': 0, 'discrete': False},
                max_episode_steps=100)
