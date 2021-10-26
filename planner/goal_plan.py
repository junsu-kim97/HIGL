import torch
import numpy as np
from .sample import farthest_point_sample

"""
Planner module is derived from Map-planner (https://github.com/FangchenLiu/map_planner)
"""

class Planner:
    def __init__(self,
                 heat=0.9,
                 n_landmark_cov=200,
                 initial_sample=1000,
                 landmark_cov_sampling="fps",
                 clip_v=-4,
                 goal_thr=-5.,
                 ):
        self.n_landmark_cov = n_landmark_cov
        self.initial_sample = initial_sample
        self.landmark_cov_sampling = landmark_cov_sampling
        self.clip_v = clip_v
        self.heat = heat
        self.goal_thr = goal_thr

    def clip_dist(self, dists, reserve=True):
        v = self.clip_v
        if reserve:
            mm = torch.min((dists - 1000 * torch.eye(len(dists)).to(dists.device)).max(dim=0)[0], dists[0]*0 + v)
            dists = dists - (dists < mm[None, :]).float() * 1000000
        else:
            dists = dists - (dists < v).float() * 1000000
        return dists

    def _value_iteration(self, A, B):
        A = A[:, :, None] + B[None, :, :]
        d = torch.softmax(A * self.heat, dim=1)
        return (A*d).sum(dim=1), d

    def value_iteration(self, dists):
        cc = dists * (1.-torch.eye(len(dists))).to(dists.device)
        ans = cc
        for i in range(20):
            ans = self._value_iteration(ans, ans)[0]
        to = self._value_iteration(cc, ans)[1]
        return ans, to

    def make_obs(self, init, goal):
        a = init[None, :].expand(len(goal), *init.shape)
        a = torch.cat((goal, a), dim=1)
        return a

    def pairwise_dists(self, states, ag, landmarks):
        with torch.no_grad():
            dists = []
            for i in landmarks:
                ld = i[None, :].expand(len(states), *i.shape)
                dists.append(self.agent.pairwise_value(states, ag, ld))
        return torch.stack(dists, dim=1)

    def pairwise_dists_batch(self, states, ag, landmarks):
        with torch.no_grad():
            states_repeat = states.repeat(len(landmarks), 1)
            ag_repeat = ag.repeat(len(landmarks), 1)
            landmarks_repeat = torch.repeat_interleave(landmarks, len(states), dim=0)
            dists = self.agent.pairwise_value(states_repeat, ag_repeat, landmarks_repeat)
            dists_list = list(torch.split(dists, len(states)))
        return torch.stack(dists_list, dim=1)

    def build_landmark_graph(self, final_goal):
        if isinstance(final_goal, torch.Tensor):
            final_goal = final_goal.detach().cpu().numpy()

        x, _, ag, _, g, _, _, _, _, _, _ = self.replay_buffer.sample(self.initial_sample)
        landmarks = ag.copy()
        state = x.copy()
        achieved_goal = ag.copy()

        # Sample Coverage-based landmarks
        if self.landmark_cov_sampling == "fps":
            # build a pool of states
            random_idx = np.random.choice(len(landmarks), self.initial_sample)
            state = state[random_idx]
            landmarks = landmarks[random_idx]
            achieved_goal = achieved_goal[random_idx]

            idx = farthest_point_sample(landmarks, self.n_landmark_cov, device=self.agent.device)
            state = state[idx]
            landmarks = landmarks[idx]
            achieved_goal = achieved_goal[idx]

            state = torch.Tensor(state).to(self.agent.device)
            landmarks = torch.Tensor(landmarks).to(self.agent.device)
            achieved_goal = torch.Tensor(achieved_goal).to(self.agent.device)

            if state.ndim == 1:
                print("Warning: coverage based landmark num: 1")
                state = state.unsqueeze(dim=0)
                landmarks = landmarks.unsqueeze(dim=0)
                achieved_goal = achieved_goal.unsqueeze(dim=0)

        elif self.landmark_cov_sampling == 'none':
            pass
        else:
            raise NotImplementedError

        # Sample Novelty-based landmarks
        if self.novelty_pq is not None:
            state_novelty = self.novelty_pq.get_states()
            landmarks_novelty = self.novelty_pq.get_landmarks()
            if self.landmark_cov_sampling == 'none':
                state = state_novelty
                achieved_goal = landmarks_novelty
                landmarks = landmarks_novelty
            else:
                state = torch.cat((state, state_novelty), dim=0)
                achieved_goal = torch.cat((achieved_goal, landmarks_novelty), dim=0)
                landmarks = torch.cat((landmarks, landmarks_novelty), dim=0)

        fg = torch.Tensor(final_goal).to(self.agent.device)
        self.num_landmark_cov_nov = len(landmarks)
        self.landmark_cov_nov = landmarks.clone()
        self.landmarks_cov_nov_fg = torch.cat((landmarks, fg), dim=0)

        dists = self.pairwise_dists_batch(state, achieved_goal, self.landmarks_cov_nov_fg)
        dists = torch.min(dists, dists*0)
        dists = torch.cat((dists, torch.zeros(len(final_goal), dists.shape[1], device=self.agent.device)-100000), dim=0)
        dists = self.clip_dist(dists)
        dists, to = self.value_iteration(dists)
        self.dists_ld2goal = dists[:, -len(fg):]
        self.to = to[:, -len(fg):]

        return self.landmarks_cov_nov_fg, self.dists_ld2goal

    def __call__(self, cur_obs, cur_ag, final_goal, agent, replay_buffer, novelty_pq):
        self.agent = agent
        self.replay_buffer = replay_buffer
        self.novelty_pq = novelty_pq

        if isinstance(cur_obs, np.ndarray):
            cur_obs = torch.Tensor(cur_obs).to(self.agent.device)
        if isinstance(cur_ag, np.ndarray):
            cur_ag = torch.Tensor(cur_ag).to(self.agent.device)
        if isinstance(final_goal, np.ndarray):
            final_goal = torch.Tensor(final_goal).to(self.agent.device)

        landmarks_cov_nov_fg, dists_ld2goal = self.build_landmark_graph(final_goal)

        dists_cur2ld = self.pairwise_dists(cur_obs, cur_ag, landmarks_cov_nov_fg)
        dists_cur2ld = torch.min(dists_cur2ld, dists_cur2ld * 0)
        dists_cur2ld = self.clip_dist(dists_cur2ld, reserve=False)

        dist = dists_cur2ld + dists_ld2goal.T

        goal_idx = list(range(len(cur_obs)))
        goal_idx_offset = list(range(self.num_landmark_cov_nov, len(landmarks_cov_nov_fg), 1))

        dist_through_ld = dist[:, :self.num_landmark_cov_nov]
        dist_direct_goal = dist[goal_idx, goal_idx_offset]

        ld = self.landmark_cov_nov[torch.argmax(dist_through_ld, dim=1)]
        if torch.any(dist_direct_goal > self.goal_thr):
            ld[dist_direct_goal > self.goal_thr] = final_goal[dist_direct_goal > self.goal_thr]

        return ld
