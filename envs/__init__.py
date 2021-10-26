import numpy as np
import argparse
import random

# import envs.create_maze_env


def get_goal_sample_fn(env_name, evaluate, fix_goal=False, manual_goal=None):
    if env_name in ['AntMaze', 'PointMaze-v0', 'AntMaze-v0']:
        if evaluate:
            return lambda: np.array([0., 16.])
        else:
            return lambda: np.random.uniform((-4, -4), (20, 20))
    elif env_name == 'AntMazeSparse' or env_name == "AntMazeW-v2":
        return lambda: np.array([2., 9.])
    elif env_name == "PointMaze":
        if evaluate:
            return lambda: np.array([0., 5.])
        else:
            return lambda: np.random.uniform((0, 0), (5, 5))
    elif env_name == "AntMazeSmall":
        if evaluate:
            return lambda: np.array([0., 8.])
        else:
            return lambda: np.random.uniform((-2, -2), (10, 10))
    elif env_name == "PointMaze-v1" or env_name == "AntMaze-v1":
        if evaluate:
            return lambda: np.array([0., 8.])
        else:
            if manual_goal is not None:
                goal = random.sample(manual_goal, 1)[0]
                return lambda: np.array(goal)
            elif fix_goal:
                return lambda: np.array([0., 8.])
            else:
                return lambda: np.random.uniform((-2, -2), (10, 10))
    else:
        assert False, 'Unknown env'


def get_reward_fn(env_name, goal_dim, step_style=False):
    if step_style:
        # if env_name in ['AntMaze-v1']:
        #    return lambda obs, goal: -(np.linalg.norm(obs[:2] - goal, axis=-1) > 5).astype(np.float32)
        if env_name in ['AntMaze-v1', 'AntMazeL-v1', 'PointMaze-v1']:
            return lambda obs, goal: -(np.linalg.norm(obs[:goal_dim] - goal, axis=-1) > 2.5).astype(np.float32)
        elif env_name in ["PointMaze-v0", "AntMaze-v0"]:
            return lambda obs, goal: -(np.linalg.norm(obs[:goal_dim] - goal, axis=-1) > 5.0).astype(np.float32)
        elif env_name in ['AntMazeW-v2']:
            return lambda obs, goal: -(np.linalg.norm(obs[:goal_dim] - goal, axis=-1) > 1.0).astype(np.float32)
        else:
            assert False, 'Unknown env'
    else:
        if env_name in ['AntMaze', 'PointMaze', "AntMazeSmall", "AntMazeL", "PointMaze-v1", "AntMaze-v1", "AntMaze-v0",
                        "AntMazeL-v1", "PointMaze-v0",
                        "AntMazeW-v2"]:
            return lambda obs, goal: -np.sum(np.square(obs[:goal_dim] - goal)) ** 0.5
        elif env_name == 'AntMazeSparse':
            return lambda obs, goal: float(np.sum(np.square(obs[:goal_dim] - goal)) ** 0.5 < 1)
        else:
            assert False, 'Unknown env'


def get_success_fn(env_name, step_style=False):
    if step_style:
        if env_name in ['AntMaze-v1', 'AntMazeL-v1', 'PointMaze-v1',
                         "PointMaze-v0", "AntMazeW-v2", "AntMaze-v0"]:
            return lambda reward: reward == 0
        else:
            assert False, 'Unknown env'
    else:
        if env_name in ['AntMaze', 'AntMazeSmall', "AntMazeL", "PointMaze-v0", "AntMaze-v0"]:
            return lambda reward: reward > -5.0
        elif env_name in ["PointMaze-v1", "AntMaze-v1", "AntMazeL-v1"]:
            return lambda reward: reward > -2.5
        elif env_name in ["AntMazeW-v2"]:
            return lambda reward: reward > -1.0
        elif env_name == 'AntMazeSparse':
            return lambda reward: reward > 1e-6
        elif env_name == 'PointMaze':
            return lambda reward: reward > -0.1
        else:
            assert False, 'Unknown env'


class GatherEnv(object):

    def __init__(self, base_env, env_name):
        self.base_env = base_env
        self.env_name = env_name
        self.evaluate = False
        self.count = 0

    def seed(self, seed):
        self.base_env.seed(seed)

    def reset(self):
        obs = self.base_env.reset()
        self.count = 0
        return {
            'observation': obs.copy(),
            'achieved_goal': obs[:2],
            'desired_goal': None,
        }

    def step(self, a):
        obs, reward, done, info = self.base_env.step(a)
        self.count += 1
        next_obs = {
            'observation': obs.copy(),
            'achieved_goal': obs[:2],
            'desired_goal': None,
        }
        return next_obs, reward, done or self.count >= 500, info

    @property
    def action_space(self):
        return self.base_env.action_space


class EnvWithGoal(object):

    def __init__(self, base_env, env_name, fix_goal=False, manual_goals=None, step_style=False,
                 stochastic_xy=False, stochastic_sigma=0.):
        if env_name in ['AntMaze-v1', "PointMaze-v0", "AntMazeSparse", "AntMazeW-v2",
                        "PointMaze-v1", "AntMaze-v0"]:
            self.goal_dim = 2
        else:
            raise NotImplementedError

        self.base_env = base_env
        self.env_name = env_name
        self.evaluate = False
        self.reward_fn = get_reward_fn(env_name, self.goal_dim, step_style)
        self.success_fn = get_success_fn(env_name, step_style)
        self.goal = None
        # self.distance_threshold = 5 if (env_name in ['AntMaze', 'AntMazeSmall', "AntMazeL", "PointMaze-v1", "AntMaze-v1", "AntMazeL-v1"]) else 1
        self.count = 0
        self.early_stop = False if (env_name in ['AntMaze', 'AntMazeSmall', "AntMazeL", "PointMaze-v1",
                                                 "AntMaze-v1", "AntMazeL-v1", "AntMaze-v0"]) else True
        self.early_stop_flag = False
        self.fix_goal = fix_goal
        self.manual_goals = manual_goals

        self.stochastic_xy = stochastic_xy
        self.stochastic_sigma = stochastic_sigma

    def seed(self, seed):
        self.base_env.seed(seed)

    def reset(self):
        # self.viewer_setup()
        self.early_stop_flag = False
        self.goal_sample_fn = get_goal_sample_fn(self.env_name, self.evaluate, self.fix_goal, self.manual_goals)
        obs = self.base_env.reset()
        self.count = 0
        self.goal = self.goal_sample_fn()
        self.desired_goal = None if 'Sparse' in self.env_name else self.goal

        return {
            'observation': obs.copy(),
            'achieved_goal': obs[:self.goal_dim],
            'desired_goal': self.desired_goal,
        }

    def step(self, a):
        obs, _, done, info = self.base_env.step(a)

        reward = self.reward_fn(obs, self.goal)
        if self.early_stop and self.success_fn(reward):
            self.early_stop_flag = True
        self.count += 1
        done = self.early_stop_flag and self.count % 10 == 0
        next_obs = {
            'observation': obs.copy(),
            'achieved_goal': obs[:self.goal_dim],
            'desired_goal': self.desired_goal,
        }
        info['is_success'] = self.success_fn(reward)
        return next_obs, reward, done or self.count >= 500, info

    def render(self, mode='rgb_array'):
        self.base_env.render(mode=mode)

    def get_image(self):
        self.render()
        data = self.base_env.viewer.get_image()

        img_data = data[0]
        width = data[1]
        height = data[2]

        tmp = np.fromstring(img_data, dtype=np.uint8)
        image_obs = np.reshape(tmp, [height, width, 3])
        image_obs = np.flipud(image_obs)

        return image_obs

    @property
    def action_space(self):
        return self.base_env.action_space


# def run_environment(env_name, episode_length, num_episodes):
#     env = EnvWithGoal(create_maze_env.create_maze_env(env_name), env_name)
#
#     def action_fn(obs):
#         action_space = env.action_space
#         action_space_mean = (action_space.low + action_space.high) / 2.0
#         action_space_magn = (action_space.high - action_space.low) / 2.0
#         random_action = (action_space_mean +
#             action_space_magn *
#             np.random.uniform(low=-1.0, high=1.0,
#             size=action_space.shape))
#
#         return random_action
#
#     rewards = []
#     successes = []
#     for ep in range(num_episodes):
#         rewards.append(0.0)
#         successes.append(False)
#         obs = env.reset()
#         for _ in range(episode_length):
#             env.render()
#             print(env.get_image().shape)
#             obs, reward, done, _ = env.step(action_fn(obs))
#             rewards[-1] += reward
#             successes[-1] = env.success_fn(reward)
#             if done:
#                 break
#
#         print('Episode {} reward: {}, Success: {}'.format(ep + 1, rewards[-1], successes[-1]))
#
#     print('Average Reward over {} episodes: {}'.format(num_episodes, np.mean(rewards)))
#     print('Average Success over {} episodes: {}'.format(num_episodes, np.mean(successes)))
