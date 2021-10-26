import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data

import numpy as np
import pandas as pd

import random

from functools import total_ordering
# from higl.higl import var

from itertools import compress

# Code based on: 
# https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Simple replay buffer
class ReplayBuffer(object):
    def __init__(self, maxsize=1e6, batch_size=100, reward_func=None, reward_scale=None):
        self.storage = [[] for _ in range(11)]
        self.maxsize = maxsize
        self.next_idx = 0
        self.batch_size = batch_size
        self.reward_func = reward_func
        self.reward_scale = reward_scale

    # Expects tuples of (x, x', ag, g, u, r, d, x_seq, a_seq, ag_seq)
    def add(self, odict):
        assert list(odict.keys()) == ['state', 'next_state', 'achieved_goal', 'next_achieved_goal',
                                      'goal', 'action', 'reward',
                                      'done', 'state_seq', 'actions_seq', 'achieved_goal_seq']
        data = tuple(odict.values())
        self.next_idx = int(self.next_idx)
        if self.next_idx >= len(self.storage[0]):
            [array.append(datapoint) for array, datapoint in zip(self.storage, data)]
        else:
            [array.__setitem__(self.next_idx, datapoint) for array, datapoint in zip(self.storage, data)]

        self.next_idx = (self.next_idx + 1) % self.maxsize

    def sample(self, batch_size):
        ind = np.random.randint(0, len(self.storage[0]), size=batch_size)

        x, y, ag, ag_next, g, u, r, d, x_seq, a_seq, ag_seq = [], [], [], [], [], [], [], [], [], [], []

        for i in ind:
            X, Y, AG, AG_NEXT, G, U, R, D, obs_seq, acts, AG_seq = (array[i] for array in self.storage)
            x.append(np.array(X, copy=False))
            y.append(np.array(Y, copy=False))
            ag.append(np.array(AG, copy=False))
            ag_next.append(np.array(AG_NEXT, copy=False))
            g.append(np.array(G, copy=False))
            u.append(np.array(U, copy=False))
            r.append(np.array(R, copy=False))
            d.append(np.array(D, copy=False))

            # For off-policy goal correction
            x_seq.append(np.array(obs_seq, copy=False))
            a_seq.append(np.array(acts, copy=False))
            ag_seq.append(np.array(AG_seq, copy=False))

        return np.array(x), np.array(y), np.array(ag), np.array(ag_next), np.array(g), \
               np.array(u), np.array(r).reshape(-1, 1), np.array(d).reshape(-1, 1), \
               x_seq, a_seq, ag_seq

    def save(self, file):
        np.savez_compressed(file, idx=np.array([self.next_idx]), x=self.storage[0],
                            y=self.storage[1], g=self.storage[2], u=self.storage[3],
                            r=self.storage[4], d=self.storage[5], xseq=self.storage[6],
                            aseq=self.storage[7], ld=self.storage[8])

    def load(self, file):
        with np.load(file) as data:
            self.next_idx = int(data['idx'][0])
            self.storage = [data['x'], data['y'], data['g'], data['u'], data['r'],
                            data['d'], data['xseq'], data['aseq'], data['ld']]
            self.storage = [list(l) for l in self.storage]

    def __len__(self):
        return len(self.storage[0])


class TrajectoryBuffer:

    def __init__(self, capacity):
        self._capacity = capacity
        self.reset()

    def reset(self):
        self._num_traj = 0  # number of trajectories
        self._size = 0    # number of game frames
        self.trajectory = []

    def __len__(self):
        return self._num_traj

    def size(self):
        return self._size

    def get_traj_num(self):
        return self._num_traj

    def full(self):
        return self._size >= self._capacity

    def create_new_trajectory(self):
        self.trajectory.append([])
        self._num_traj += 1

    def append(self, s):
        self.trajectory[self._num_traj-1].append(s)
        self._size += 1

    def get_trajectory(self):
        return self.trajectory

    def set_capacity(self, new_capacity):
        assert self._size <= new_capacity
        self._capacity = new_capacity


class NormalNoise(object):
    def __init__(self, sigma):
        self.sigma = sigma

    def perturb_action(self, action, min_action=-np.inf, max_action=np.inf):
        action = (action + np.random.normal(0, self.sigma,
            size=action.shape[0])).clip(min_action, max_action)
        return action

class OUNoise(object):
    def __init__(self, action_dim, mu=0, theta=0.15, sigma=0.3):
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.action_dim = action_dim
        self.X = np.ones(self.action_dim) * self.mu

    def reset(self):
        self.X = np.ones(self.action_dim) * self.mu

    def perturb_action(self, action, min_action=-np.inf, max_action=np.inf):
        dx = self.theta * (self.mu - self.X)
        dx = dx + self.sigma * np.random.randn(len(self.X))
        self.X = self.X + dx
        return (self.X + action).clip(min_action, max_action)


def train_adj_net(a_net, states, adj_mat, optimizer, margin_pos, margin_neg,
                  n_epochs=100, batch_size=64, device='cpu', verbose=False):
    if verbose:
        print('Generating training data...')
    dataset = MetricDataset(states, adj_mat)
    if verbose:
        print('Totally {} training pairs.'.format(len(dataset)))
    dataloader = Data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=False)
    n_batches = len(dataloader)

    loss_func = ContrastiveLoss(margin_pos, margin_neg)

    for i in range(n_epochs):
        epoch_loss = []
        for j, data in enumerate(dataloader):
            x, y, label = data
            x = x.float().to(device)
            y = y.float().to(device)
            label = label.long().to(device)
            x = a_net(x)
            y = a_net(y)
            loss = loss_func(x, y, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if verbose and (j % 50 == 0 or j == n_batches - 1):
                print('Training metric network: epoch {}/{}, batch {}/{}'.format(i+1, n_epochs, j+1, n_batches))

            epoch_loss.append(loss.item())

        if verbose:
            print('Mean loss: {:.4f}'.format(np.mean(epoch_loss)))


class ContrastiveLoss(nn.Module):

    def __init__(self, margin_pos, margin_neg):
        super().__init__()
        assert margin_pos <= margin_neg
        self.margin_pos = margin_pos
        self.margin_neg = margin_neg

    def forward(self, x, y, label):
        # mutually reachable states correspond to label = 1
        dist = torch.sqrt(torch.pow(x - y, 2).sum(dim=1) + 1e-12)
        loss = (label * (dist - self.margin_pos).clamp(min=0)).mean() + ((1 - label) * (self.margin_neg - dist).clamp(min=0)).mean()
        return loss


class MetricDataset(Data.Dataset):

    def __init__(self, states, adj_mat):
        super().__init__()
        n_samples = adj_mat.shape[0]
        self.x = []
        self.y = []
        self.label = []
        for i in range(n_samples - 1):
            for j in range(i + 1, n_samples):
                self.x.append(states[i])
                self.y.append(states[j])
                self.label.append(adj_mat[i, j])
        self.x = np.array(self.x)
        self.y = np.array(self.y)
        self.label = np.array(self.label)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.label[idx]


def get_reward_function(env, env_name, absolute_goal=False, binary_reward=False):
    # distance_threshold = env.distance_threshold
    if env_name in ["AntMaze-v1", "PointMaze-v1"]:
        distance_threshold = 2.5
    elif env_name == "AntMazeW-v2":
        distance_threshold = 1
    if absolute_goal and binary_reward:
        def controller_reward(ag, subgoal, next_ag, scale, action=None):
            reward = -float(np.linalg.norm(subgoal - next_ag, axis=-1) > distance_threshold) * scale
            return reward
    elif absolute_goal:
        def controller_reward(ag, subgoal, next_ag, scale, action=None):
            reward = -np.linalg.norm(subgoal - next_ag, axis=-1) * scale
            return reward
    elif binary_reward:
        def controller_reward(ag, subgoal, next_ag, scale, action=None):
            reward = -float(np.linalg.norm(ag + subgoal - next_ag, axis=-1) > distance_threshold) * scale
            return reward
    else:
        def controller_reward(ag, subgoal, next_ag, scale, action=None):
            reward = -np.linalg.norm(ag + subgoal - next_ag, axis=-1) * scale
            return reward

    return controller_reward


def get_mbrl_fetch_reward_function(env, env_name, binary_reward, absolute_goal):
    action_penalty_coeff = 0.0001
    distance_threshold = 0.25
    if env_name in ["Reacher3D-v0", "Pusher-v0"]:
        if absolute_goal and not binary_reward:
            def controller_reward(ag, subgoal, next_ag, scale, action):
                reward = -np.sum(np.square(ag - subgoal))
                reward -= action_penalty_coeff * np.square(action).sum()
                return reward * scale

        elif absolute_goal and binary_reward:
            def controller_reward(ag, subgoal, next_ag, scale, action):
                reward_ctrl = action_penalty_coeff * -np.square(action).sum()
                fail = True
                if np.sqrt(np.sum(np.square(ag - subgoal))) <= distance_threshold:
                    fail = False
                reward = reward_ctrl - float(fail)
                return reward * scale

        elif not absolute_goal and not binary_reward:
            def controller_reward(ag, subgoal, next_ag, scale, action):
                reward = -np.sum(np.square(ag + subgoal - next_ag))
                reward -= action_penalty_coeff * np.square(action).sum()
                return reward * scale

        elif not absolute_goal and binary_reward:
            def controller_reward(ag, subgoal, next_ag, scale, action):
                reward_ctrl = action_penalty_coeff * -np.square(action).sum()
                fail = True
                if np.sqrt(np.sum(np.square(ag - subgoal))) <= distance_threshold:
                    fail = False
                reward = reward_ctrl - float(fail)
                return reward * scale
    else:
        raise NotImplementedError
    return controller_reward


def is_goal_unreachable(a_net, state, goal, goal_dim, margin, device, absolute_goal=False):
    state = torch.from_numpy(state[:goal_dim]).float().to(device)
    goal = torch.from_numpy(goal).float().to(device)
    if not absolute_goal:
        goal = state + goal
    inputs = torch.stack((state, goal), dim=0)
    outputs = a_net(inputs)
    s_embedding = outputs[0]
    g_embedding = outputs[1]
    dist = F.pairwise_distance(s_embedding.unsqueeze(0), g_embedding.unsqueeze(0)).squeeze()
    return dist > margin


@total_ordering
class StorageElement:
    def __init__(self, state, achieved_goal, score):
        self.state = state
        self.achieved_goal = achieved_goal
        self.score = score

    def __eq__(self, other):
        return np.isclose(self.score, other.score)

    def __lt__(self, other):
        return self.score < other.score

    def __hash__(self):
        return hash(tuple(self.state))


def unravel_elems(elems):
    return tuple(map(list, zip(*[(elem.state, elem.score) for elem in elems])))


class PriorityQueue:
    def __init__(self, top_k, close_thr=0.1, discard_by_anet=False):
        self.elems = []
        self.elems_state_tensor = None
        self.elems_achieved_goal_tensor = None

        self.top_k = top_k
        self.close_thr = close_thr
        self.discard_by_anet = discard_by_anet

    def __len__(self):
        return len(self.elems)

    def add_list(self, state_list, achieved_goal_list, score_list, a_net=None):
        if self.discard_by_anet:
            self.discard_out_of_date_by_anet(achieved_goal_list, a_net)
        else:
            self.discard_out_of_date(achieved_goal_list)
        # total_timesteps = len(state_list)
        # Fill inf in the future observation | achieved goal
        new_elems = [StorageElement(state=state, achieved_goal=achieved_goal, score=score)
                     for timestep, (state, achieved_goal, score)
                     in enumerate(zip(state_list, achieved_goal_list, score_list))]
        self.elems.extend(new_elems)
        self.elems = list(set(self.elems))
        self.update_tensors()

    def update_tensors(self):
        self.elems_state_tensor = torch.FloatTensor([elems.state for elems in self.elems]).to(device)
        self.elems_achieved_goal_tensor = torch.FloatTensor([elems.achieved_goal for elems in self.elems]).to(device)

    # update novelty of similar states existing in storage to the newly encountered one.
    def discard_out_of_date(self, achieved_goal_list):
        if len(self.elems) == 0:
            return

        achieved_goals = torch.FloatTensor(np.array(achieved_goal_list)).to(device)
        dist = torch.cdist(self.elems_achieved_goal_tensor, achieved_goals)
        close = dist < self.close_thr
        keep = close.sum(dim=1) == 0
        self.elems = list(compress(self.elems, keep))
        self.update_tensors()

    def discard_out_of_date_by_anet(self, achieved_goal_list, a_net):
        assert a_net is not None
        if len(self.elems) == 0:
            return

        with torch.no_grad():
            achieved_goals = torch.FloatTensor(np.array(achieved_goal_list)).to(device)
            dist1 = torch.cdist(a_net(achieved_goals), a_net(self.elems_achieved_goal_tensor)).T
            dist2 = torch.cdist(a_net(self.elems_achieved_goal_tensor), a_net(achieved_goals))
            dist = (dist1 + dist2)/2

            close = dist < self.close_thr
            keep = close.sum(dim=1) == 0
            self.elems = list(compress(self.elems, keep))
            self.update_tensors()

    def get_elems(self):
        return unravel_elems(self.elems[:self.top_k])

    def get_states(self):
        return self.elems_state_tensor[:self.top_k]

    def get_landmarks(self):
        return self.elems_achieved_goal_tensor[:self.top_k]

    def squeeze_by_kth(self, k):
        k = min(k, len(self.elems))
        self.elems = sorted(self.elems, reverse=True)[:k]
        self.update_tensors()
        return self.elems[-1].score

    def squeeze_by_thr(self, thr):
        self.elems = sorted(self.elems, reverse=True)
        k = next((i for i, elem in enumerate(self.elems) if elem.score < thr), len(self.elems))

        self.elems = self.elems[:k]
        self.update_tensors()
        return unravel_elems(self.elems)

    def sample_batch(self, batch_size):
        sampled_elems = random.choices(population=self.elems, k=batch_size)
        return unravel_elems(sampled_elems)

    def save_log(self, timesteps, log_file):
        elems = self.get_elems()
        output_df = pd.DataFrame((timesteps, score, state) for state, score in zip(elems[0], elems[1]))
        output_df.to_csv(log_file, mode='a', header=False)

    def sample_by_novelty_weight(self):
        raise NotImplementedError


class RunningMeanStd(object):
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * (self.count)
        m_b = batch_var * (batch_count)
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / (self.count + batch_count)
        new_var = M2 / (self.count + batch_count)

        new_count = batch_count + self.count

        self.mean = new_mean
        self.var = new_var
        self.count = new_count
