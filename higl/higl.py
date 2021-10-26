import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from higl.models import ControllerActor, ControllerCritic, ManagerActor, ManagerCritic, RndPredictor
# from higl.utils import RunningMeanStd
from planner.goal_plan import Planner

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def var(tensor):
    return tensor.to(device)


def get_tensor(z):
    if z is None:
        return None
    if z[0].dtype == np.dtype("O"):
        return None
    if len(z.shape) == 1:
        return var(torch.FloatTensor(z.copy())).unsqueeze(0)
    else:
        return var(torch.FloatTensor(z.copy()))


class Manager(object):
    def __init__(self,
                 state_dim,
                 goal_dim,
                 action_dim,
                 actor_lr,
                 critic_lr,
                 candidate_goals,
                 correction=True,
                 scale=10,
                 actions_norm_reg=0,
                 policy_noise=0.2,
                 noise_clip=0.5,
                 goal_loss_coeff=0,
                 absolute_goal=False,
                 absolute_goal_scale=8.,
                 landmark_loss_coeff=0.,
                 delta=2.0,
                 no_pseudo_landmark=False,
                 automatic_delta_pseudo=False,
                 planner_start_step=50000,
                 planner_cov_sampling='fps',
                 planner_clip_v=-38.,
                 n_landmark_cov=20,
                 planner_initial_sample=1000,
                 planner_goal_thr=-10.,
                 ):
        self.scale = scale
        self.actor = ManagerActor(state_dim,
                                  goal_dim,
                                  action_dim,
                                  scale=scale,
                                  absolute_goal=absolute_goal,
                                  absolute_goal_scale=absolute_goal_scale)
        self.actor_target = ManagerActor(state_dim,
                                         goal_dim,
                                         action_dim,
                                         scale=scale)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)

        self.critic = ManagerCritic(state_dim, goal_dim, action_dim)
        self.critic_target = ManagerCritic(state_dim, goal_dim, action_dim)

        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=critic_lr,
                                                 weight_decay=0.0001)

        self.action_norm_reg = 0

        if torch.cuda.is_available():
            self.actor = self.actor.to(device)
            self.actor_target = self.actor_target.to(device)
            self.critic = self.critic.to(device)
            self.critic_target = self.critic_target.to(device)

        self.criterion = nn.SmoothL1Loss()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.candidate_goals = candidate_goals
        self.correction = correction
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.goal_loss_coeff = goal_loss_coeff
        self.absolute_goal = absolute_goal

        self.landmark_loss_coeff = landmark_loss_coeff
        self.delta = delta
        self.device = device
        self.planner = None
        self.no_pseudo_landmark = no_pseudo_landmark

        self.automatic_delta_pseudo = automatic_delta_pseudo
        if self.automatic_delta_pseudo:
            self.delta = 0.

        self.planner_start_step = planner_start_step
        self.planner_cov_sampling = planner_cov_sampling
        self.planner_clip_v = planner_clip_v
        self.n_landmark_cov = n_landmark_cov
        self.planner_initial_sample = planner_initial_sample
        self.planner_goal_thr = planner_goal_thr

    def init_planner(self):
        self.planner = Planner(landmark_cov_sampling=self.planner_cov_sampling,
                               clip_v=self.planner_clip_v,
                               n_landmark_cov=self.n_landmark_cov,
                               initial_sample=self.planner_initial_sample,
                               goal_thr=self.planner_goal_thr)

    def set_delta(self, data, alpha=0.9):
        assert self.automatic_delta_pseudo
        if self.delta == 0:
            self.delta = data
        else:
            self.delta = alpha * data + (1 - alpha) * self.delta

    def set_eval(self):
        self.actor.set_eval()
        self.actor_target.set_eval()

    def set_train(self):
        self.actor.set_train()
        self.actor_target.set_train()

    def sample_goal(self, state, goal, to_numpy=True):
        state = get_tensor(state)
        goal = get_tensor(goal)

        if to_numpy:
            return self.actor(state, goal).cpu().data.numpy().squeeze()
        else:
            return self.actor(state, goal).squeeze()

    def value_estimate(self, state, goal, subgoal):
        return self.critic(state, goal, subgoal)

    def get_pseudo_landmark(self, ag, planned_ld):
        direction = planned_ld - ag
        norm_direction = F.normalize(direction)
        scaled_norm_direction = norm_direction * self.delta
        pseudo_landmarks = ag.clone()
        pseudo_landmarks[~torch.isnan(scaled_norm_direction)] = pseudo_landmarks[~torch.isnan(scaled_norm_direction)] +\
                                                            scaled_norm_direction[~torch.isnan(scaled_norm_direction)]

        scaled_norm_direction[scaled_norm_direction != scaled_norm_direction] = 0
        return pseudo_landmarks, scaled_norm_direction.mean(dim=0)

    def actor_loss(self, state, achieved_goal, goal, a_net, r_margin, selected_landmark=None, no_pseudo_landmark=False):
        actions = self.actor(state, goal)
        eval = -self.critic.Q1(state, goal, actions).mean()
        norm = torch.norm(actions)*self.action_norm_reg
        if a_net is None:
            return eval + norm  # HIRO

        scaled_norm_direction = var(torch.FloatTensor([0.] * self.action_dim))
        gen_subgoal = actions if self.absolute_goal else achieved_goal + actions
        goal_loss = torch.clamp(F.pairwise_distance(a_net(achieved_goal), a_net(gen_subgoal)) - r_margin, min=0.).mean()
        if selected_landmark is None:
            return eval + norm, goal_loss, None, scaled_norm_direction  # HRAC

        if no_pseudo_landmark:
            selected_landmark[selected_landmark == float('inf')] = achieved_goal[selected_landmark == float('inf')]
            batch_landmarks = selected_landmark.clone()
        else:
            batch_landmarks, scaled_norm_direction = self.get_pseudo_landmark(achieved_goal, selected_landmark)
        ld_loss = torch.clamp(F.pairwise_distance(a_net(batch_landmarks), a_net(gen_subgoal)) - r_margin, min=0.).mean()

        return eval + norm, goal_loss, ld_loss, scaled_norm_direction  # HIGL

    def off_policy_corrections(self, controller_policy, batch_size, subgoals, x_seq, a_seq, ag_seq):
        first_ag = [x[0] for x in ag_seq]
        last_ag = [x[-1] for x in ag_seq]

        # Shape: (batchsz, 1, subgoal_dim)
        diff_goal = (np.array(last_ag) - np.array(first_ag))[:, np.newaxis, ]

        # Shape: (batchsz, 1, subgoal_dim)
        original_goal = np.array(subgoals)[:, np.newaxis, :]
        random_goals = np.random.normal(loc=diff_goal,
                                        scale=.5*self.scale[None, None, :self.action_dim],
                                        size=(batch_size, self.candidate_goals, original_goal.shape[-1]))
        random_goals = random_goals.clip(-self.scale[:self.action_dim], self.scale[:self.action_dim])

        # Shape: (batchsz, 10, subgoal_dim)
        candidates = np.concatenate([original_goal, diff_goal, random_goals], axis=1)
        x_seq = np.array(x_seq)[:, :-1, :]
        a_seq = np.array(a_seq)
        seq_len = len(x_seq[0])

        # For ease
        new_batch_sz = seq_len * batch_size
        action_dim = a_seq[0][0].shape
        obs_dim = x_seq[0][0].shape
        ncands = candidates.shape[1]

        true_actions = a_seq.reshape((new_batch_sz,) + action_dim)
        observations = x_seq.reshape((new_batch_sz,) + obs_dim)
        goal_shape = (new_batch_sz, self.action_dim)

        policy_actions = np.zeros((ncands, new_batch_sz) + action_dim)

        for c in range(ncands):
            candidate = controller_policy.multi_subgoal_transition(np.array(ag_seq)[:, :-1, :], candidates[:, c])
            candidate = candidate.reshape(*goal_shape)
            policy_actions[c] = controller_policy.select_action(observations, candidate)

        difference = (policy_actions - true_actions)
        difference = np.where(difference != -np.inf, difference, 0)
        difference = difference.reshape((ncands, batch_size, seq_len) + action_dim).transpose(1, 0, 2, 3)

        logprob = -0.5*np.sum(np.linalg.norm(difference, axis=-1)**2, axis=-1)
        max_indices = np.argmax(logprob, axis=-1)

        return candidates[np.arange(batch_size), max_indices]

    def train(self,
              algo,
              controller_policy,
              replay_buffer,
              controller_replay_buffer,
              iterations,
              batch_size=100,
              discount=0.99,
              tau=0.005,
              a_net=None,
              r_margin=None,
              total_timesteps=None,
              novelty_pq=None,
              ):
        self.manager_buffer = replay_buffer
        avg_act_loss, avg_crit_loss, avg_goal_loss, avg_ld_loss, avg_bonus, avg_norm_sel = 0., 0., 0., 0., 0., 0.
        avg_scaled_norm_direction = get_tensor(np.array([0.] * self.action_dim)).squeeze()

        if algo == 'higl' and self.planner is None and total_timesteps >= self.planner_start_step:
            self.init_planner()

        for it in range(iterations):
            # Sample replay buffer
            x, y, ag, ag_next, g, sgorig, r, d, xobs_seq, a_seq, ag_seq = replay_buffer.sample(batch_size)

            if self.correction and not self.absolute_goal:
                sg = self.off_policy_corrections(controller_policy, batch_size, sgorig, xobs_seq, a_seq, ag_seq)
            else:
                sg = sgorig

            state = get_tensor(x)
            next_state = get_tensor(y)
            achieved_goal = get_tensor(ag)
            goal = get_tensor(g)
            subgoal = get_tensor(sg)

            reward = get_tensor(r)
            done = get_tensor(1 - d)

            noise = torch.FloatTensor(sgorig).data.normal_(0, self.policy_noise).to(device)
            noise = noise.clamp(-self.noise_clip, self.noise_clip)
            next_action = (self.actor_target(next_state, goal) + noise)
            next_action = torch.min(next_action, self.actor.scale)
            next_action = torch.max(next_action, -self.actor.scale)

            target_Q1, target_Q2 = self.critic_target(next_state, goal, next_action)

            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + (done * discount * target_Q)
            target_Q_no_grad = target_Q.detach()

            # Get current Q estimate
            current_Q1, current_Q2 = self.value_estimate(state, goal, subgoal)

            # Compute critic loss
            critic_loss = self.criterion(current_Q1, target_Q_no_grad) + \
                          self.criterion(current_Q2, target_Q_no_grad)

            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            if algo == "hiro":
                actor_loss = self.actor_loss(state, achieved_goal, goal, a_net, r_margin)

            elif algo == "hrac":
                assert a_net is not None

                actor_loss, goal_loss, _, _ = \
                    self.actor_loss(state, achieved_goal, goal, a_net, r_margin, selected_landmark=None)
                actor_loss = actor_loss + self.goal_loss_coeff * goal_loss
                avg_goal_loss += goal_loss

            elif algo == "higl":
                assert a_net is not None

                if self.planner is None:  # If planner is not ready
                    selected_landmark = torch.ones(len(state), self.action_dim).to(device)
                    selected_landmark *= float("inf")  # Build dummy selected landmark
                else:  # Select a landmark by a planner
                    selected_landmark = self.planner(cur_obs=x,
                                                     cur_ag=ag,
                                                     final_goal=g,
                                                     agent=controller_policy,
                                                     replay_buffer=controller_replay_buffer,
                                                     novelty_pq=novelty_pq)
                    if self.automatic_delta_pseudo:
                        ag2sel = np.linalg.norm(selected_landmark.cpu().numpy() - ag, axis=1).mean()
                        self.set_delta(ag2sel)

                actor_loss, goal_loss, ld_loss, scaled_norm_direction = self.actor_loss(state, achieved_goal, goal,
                                                                                        a_net, r_margin,
                                                                                        selected_landmark,
                                                                                        self.no_pseudo_landmark)
                actor_loss = actor_loss + self.landmark_loss_coeff * ld_loss
                avg_goal_loss += goal_loss
                avg_ld_loss += ld_loss
                avg_scaled_norm_direction += scaled_norm_direction
            else:
                raise NotImplementedError

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            avg_act_loss += actor_loss
            avg_crit_loss += critic_loss

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        return avg_act_loss / iterations, \
               avg_crit_loss / iterations, \
               avg_goal_loss / iterations, \
               avg_ld_loss / iterations,\
               avg_scaled_norm_direction / iterations

    def load_pretrained_weights(self, filename):
        state = torch.load(filename)
        self.actor.encoder.load_state_dict(state)
        self.actor_target.encoder.load_state_dict(state)
        print("Successfully loaded Manager encoder.")

    def save(self, dir, env_name, algo, version, seed):
        torch.save(self.actor.state_dict(),
                   "{}/{}_{}_{}_{}_ManagerActor.pth".format(dir, env_name, algo, version, seed))
        torch.save(self.critic.state_dict(),
                   "{}/{}_{}_{}_{}_ManagerCritic.pth".format(dir, env_name, algo, version, seed))
        torch.save(self.actor_target.state_dict(),
                   "{}/{}_{}_{}_{}_ManagerActorTarget.pth".format(dir, env_name, algo, version, seed))
        torch.save(self.critic_target.state_dict(),
                   "{}/{}_{}_{}_{}_ManagerCriticTarget.pth".format(dir, env_name, algo, version, seed))
        # torch.save(self.actor_optimizer.state_dict(), "{}/{}_{}_ManagerActorOptim.pth".format(dir, env_name, algo))
        # torch.save(self.critic_optimizer.state_dict(), "{}/{}_{}_ManagerCriticOptim.pth".format(dir, env_name, algo))

    def load(self, dir, env_name, algo, version, seed):
        self.actor.load_state_dict(
            torch.load("{}/{}_{}_{}_{}_ManagerActor.pth".format(dir, env_name, algo, version, seed)))
        self.critic.load_state_dict(
            torch.load("{}/{}_{}_{}_{}_ManagerCritic.pth".format(dir, env_name, algo, version, seed)))
        self.actor_target.load_state_dict(
            torch.load("{}/{}_{}_{}_{}_ManagerActorTarget.pth".format(dir, env_name, algo, version, seed)))
        self.critic_target.load_state_dict(
            torch.load("{}/{}_{}_{}_{}_ManagerCriticTarget.pth".format(dir, env_name, algo, version, seed)))
        # self.actor_optimizer.load_state_dict(torch.load("{}/{}_{}_ManagerActorOptim.pth".format(dir, env_name, algo)))
        # self.critic_optimizer.load_state_dict(torch.load("{}/{}_{}_ManagerCriticOptim.pth".format(dir, env_name, algo)))


class Controller(object):
    def __init__(self,
                 state_dim,
                 goal_dim,
                 action_dim,
                 max_action,
                 actor_lr,
                 critic_lr,
                 no_xy=True,
                 policy_noise=0.2,
                 noise_clip=0.5,
                 absolute_goal=False,
    ):
        self.actor = ControllerActor(state_dim, goal_dim, action_dim, scale=max_action)
        self.actor_target = ControllerActor(state_dim, goal_dim, action_dim, scale=max_action)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)

        self.critic = ControllerCritic(state_dim, goal_dim, action_dim)
        self.critic_target = ControllerCritic(state_dim, goal_dim, action_dim)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr, weight_decay=0.0001)

        self.no_xy = no_xy

        self.subgoal_transition = self.subgoal_transition

        if torch.cuda.is_available():
            self.actor = self.actor.to(device)
            self.actor_target = self.actor_target.to(device)
            self.critic = self.critic.to(device)
            self.critic_target = self.critic_target.to(device)

        self.criterion = nn.SmoothL1Loss()
        self.state_dim = state_dim
        self.goal_dim = goal_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.absolute_goal = absolute_goal

        self.device = device

    def clean_obs(self, state, dims=2):
        if self.no_xy:
            with torch.no_grad():
                mask = torch.ones_like(state)
                if len(state.shape) == 3:
                    mask[:, :, :dims] = 0
                elif len(state.shape) == 2:
                    mask[:, :dims] = 0
                elif len(state.shape) == 1:
                    mask[:dims] = 0

                return state*mask
        else:
            return state

    def select_action(self, state, sg, to_numpy=True):
        state = get_tensor(state)
        sg = get_tensor(sg)
        state = self.clean_obs(state)

        if to_numpy:
            return self.actor(state, sg).cpu().data.numpy().squeeze()
        else:
            return self.actor(state, sg).squeeze()

    def value_estimate(self, state, sg, action):
        state = self.clean_obs(get_tensor(state))
        sg = get_tensor(sg)
        action = get_tensor(action)
        return self.critic(state, sg, action)

    def actor_loss(self, state, sg):
        return -self.critic.Q1(state, sg, self.actor(state, sg)).mean()

    def subgoal_transition(self, achieved_goal, subgoal, next_achieved_goal):
        if self.absolute_goal:
            return subgoal
        else:
            if len(achieved_goal.shape) == 1:  # check if batched
                return achieved_goal + subgoal - next_achieved_goal
            else:
                return achieved_goal[:, ] + subgoal - next_achieved_goal[:, ]

    def multi_subgoal_transition(self, achieved_goal, subgoal):
        subgoals = (subgoal + achieved_goal[:, 0, ])[:, None] - achieved_goal[:, :, ]
        return subgoals

    def train(self, replay_buffer, iterations, batch_size=100, discount=0.99, tau=0.005):

        avg_act_loss, avg_crit_loss = 0., 0.

        for it in range(iterations):
            # Sample replay buffer
            x, y, ag, ag_next, sg, u, r, d, _, _, _ = replay_buffer.sample(batch_size)

            next_g = get_tensor(self.subgoal_transition(ag, sg, ag_next))
            state = self.clean_obs(get_tensor(x))
            action = get_tensor(u)
            sg = get_tensor(sg)
            done = get_tensor(1 - d)
            reward = get_tensor(r)
            next_state = self.clean_obs(get_tensor(y))

            noise = torch.FloatTensor(u).data.normal_(0, self.policy_noise).to(device)
            noise = noise.clamp(-self.noise_clip, self.noise_clip)
            next_action = (self.actor_target(next_state, next_g) + noise)
            next_action = torch.min(next_action, self.actor.scale)
            next_action = torch.max(next_action, -self.actor.scale)

            target_Q1, target_Q2 = self.critic_target(next_state, next_g, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + (done * discount * target_Q)
            target_Q_no_grad = target_Q.detach()

            # Get current Q estimate
            current_Q1, current_Q2 = self.critic(state, sg, action)

            # Compute critic loss
            critic_loss = self.criterion(current_Q1, target_Q_no_grad) +\
                          self.criterion(current_Q2, target_Q_no_grad)

            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Compute actor loss
            actor_loss = self.actor_loss(state, sg)

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            avg_act_loss += actor_loss
            avg_crit_loss += critic_loss

            # Update the target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        return avg_act_loss / iterations, avg_crit_loss / iterations,

    def save(self, dir, env_name, algo, version, seed):
        torch.save(self.actor.state_dict(), "{}/{}_{}_{}_{}_ControllerActor.pth".format(dir, env_name, algo, version, seed))
        torch.save(self.critic.state_dict(), "{}/{}_{}_{}_{}_ControllerCritic.pth".format(dir, env_name, algo, version, seed))
        torch.save(self.actor_target.state_dict(), "{}/{}_{}_{}_{}_ControllerActorTarget.pth".format(dir, env_name, algo, version, seed))
        torch.save(self.critic_target.state_dict(), "{}/{}_{}_{}_{}_ControllerCriticTarget.pth".format(dir, env_name, algo, version, seed))

    def load(self, dir, env_name, algo, version, seed):
        self.actor.load_state_dict(torch.load("{}/{}_{}_{}_{}_ControllerActor.pth".format(dir, env_name, algo, version, seed)))
        self.critic.load_state_dict(torch.load("{}/{}_{}_{}_{}_ControllerCritic.pth".format(dir, env_name, algo, version, seed)))
        self.actor_target.load_state_dict(torch.load("{}/{}_{}_{}_{}_ControllerActorTarget.pth".format(dir, env_name, algo, version, seed)))
        self.critic_target.load_state_dict(torch.load("{}/{}_{}_{}_{}_ControllerCriticTarget.pth".format(dir, env_name, algo, version, seed)))

    def pairwise_value(self, obs, ag, goal):
        assert ag.shape[0] == goal.shape[0]
        with torch.no_grad():
            if not self.absolute_goal:
                relative_goal = goal - ag
                cleaned_obs = self.clean_obs(obs)
                actions = self.actor(cleaned_obs, relative_goal)
                dist1, dist2 = self.critic(cleaned_obs, relative_goal, actions)
                dist = torch.min(dist1, dist2)
                return dist.squeeze(-1)
            else:
                cleaned_obs = self.clean_obs(obs)
                actions = self.actor(cleaned_obs, goal)
                dist1, dist2 = self.critic(cleaned_obs, goal, actions)
                dist = torch.min(dist1, dist2)
                return dist.squeeze(-1)


class RandomNetworkDistillation(object):
    def __init__(self, input_dim, output_dim, lr, use_ag_as_input=False):
        self.predictor = RndPredictor(input_dim, output_dim)
        self.predictor_target = RndPredictor(input_dim, output_dim)

        if torch.cuda.is_available():
            self.predictor = self.predictor.to(device)
            self.predictor_target = self.predictor_target.to(device)

        self.optimizer = torch.optim.Adam(self.predictor.parameters(), lr=lr)
        self.use_ag_as_input = use_ag_as_input

    def get_novelty(self, obs):
        obs = get_tensor(obs)
        with torch.no_grad():
            target_feature = self.predictor_target(obs)
            feature = self.predictor(obs)
            novelty = (feature - target_feature).pow(2).sum(1).unsqueeze(1) / 2
        return novelty

    def train(self, replay_buffer, iterations, batch_size=100):
        for it in range(iterations):
            # Sample replay buffer
            x, _, ag, _, _, _, _, _, _, _, _ = replay_buffer.sample(batch_size)

            input = x if not self.use_ag_as_input else ag
            input = get_tensor(input)

            with torch.no_grad():
                target_feature = self.predictor_target(input)
            feature = self.predictor(input)
            loss = (feature - target_feature).pow(2).mean()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return loss
