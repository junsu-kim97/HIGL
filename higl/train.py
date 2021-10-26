import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import os
import numpy as np
import pandas as pd
from math import ceil
from collections import OrderedDict

import higl.utils as utils
import higl.higl as higl
from higl.models import ANet

import gym
from goal_env import *
from goal_env.mujoco import *

from envs import EnvWithGoal


def evaluate_policy(env,
                    env_name,
                    manager_policy,
                    controller_policy,
                    calculate_controller_reward,
                    ctrl_rew_scale,
                    manager_propose_frequency=10,
                    eval_idx=0,
                    eval_episodes=5,
                    ):
    print("Starting evaluation number {}...".format(eval_idx))
    env.evaluate = True

    with torch.no_grad():
        avg_reward = 0.
        avg_controller_rew = 0.
        global_steps = 0
        goals_achieved = 0
        for eval_ep in range(eval_episodes):
            obs = env.reset()

            goal = obs["desired_goal"]
            achieved_goal = obs["achieved_goal"]
            state = obs["observation"]

            done = False
            step_count = 0
            env_goals_achieved = 0
            while not done:
                if step_count % manager_propose_frequency == 0:
                    subgoal = manager_policy.sample_goal(state, goal)

                step_count += 1
                global_steps += 1
                action = controller_policy.select_action(state, subgoal)
                new_obs, reward, done, info = env.step(action)
                is_success = info['is_success']
                if is_success:
                    env_goals_achieved += 1
                    goals_achieved += 1
                    done = True

                goal = new_obs["desired_goal"]
                new_achieved_goal = new_obs['achieved_goal']
                new_state = new_obs["observation"]

                subgoal = controller_policy.subgoal_transition(achieved_goal, subgoal, new_achieved_goal)

                avg_reward += reward
                avg_controller_rew += calculate_controller_reward(achieved_goal, subgoal, new_achieved_goal,
                                                                  ctrl_rew_scale, action)
                state = new_state
                achieved_goal = new_achieved_goal

        avg_reward /= eval_episodes
        avg_controller_rew /= global_steps
        avg_step_count = global_steps / eval_episodes
        avg_env_finish = goals_achieved / eval_episodes

        print("---------------------------------------")
        print("Evaluation over {} episodes:\nAvg Ctrl Reward: {:.3f}".format(eval_episodes, avg_controller_rew))
        if "Gather" in env_name:
            print("Avg reward: {:.1f}".format(avg_reward))
        else:
            print("Goals achieved: {:.1f}%".format(100*avg_env_finish))
        print("Avg Steps to finish: {:.1f}".format(avg_step_count))
        print("---------------------------------------")

        env.evaluate = False

        final_x = new_obs['achieved_goal'][0]
        final_y = new_obs['achieved_goal'][1]

        final_subgoal_x = subgoal[0]
        final_subgoal_y = subgoal[1]
        try:
            final_z = new_obs['achieved_goal'][2]
            final_subgoal_z = subgoal[2]
        except IndexError:
            final_z = 0
            final_subgoal_z = 0

        return avg_reward, avg_controller_rew, avg_step_count, avg_env_finish, \
               final_x, final_y, final_z, \
               final_subgoal_x, final_subgoal_y, final_subgoal_z


def run_higl(args):
    if not os.path.exists("./results"):
        os.makedirs("./results")
    if args.save_models and not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    if not os.path.exists(os.path.join(args.log_dir, args.algo)):
        os.makedirs(os.path.join(args.log_dir, args.algo))
    output_dir = os.path.join(args.log_dir, args.algo)
    print("Logging in {}".format(output_dir))

    if args.save_models:
        import pickle
        with open("opts.pkl", "wb") as f:
            pickle.dump(args, f)

    if "Ant" in args.env_name:
        step_style = args.reward_shaping == 'sparse'
        env = EnvWithGoal(gym.make(args.env_name,
                                   stochastic_xy=args.stochastic_xy,
                                   stochastic_sigma=args.stochastic_sigma),
                          env_name=args.env_name, step_style=step_style)
    elif "Point" in args.env_name:
        assert not args.stochastic_xy
        step_style = args.reward_shaping == 'sparse'
        env = EnvWithGoal(gym.make(args.env_name), env_name=args.env_name, step_style=step_style)
    else:
        env = gym.make(args.env_name, reward_shaping=args.reward_shaping)
        # env_test = gym.make(args.env_name_test, reward_shaping=args.reward_shaping)

    max_action = float(env.action_space.high[0])
    train_ctrl_policy_noise = args.train_ctrl_policy_noise
    train_ctrl_noise_clip = args.train_ctrl_noise_clip

    train_man_policy_noise = args.train_man_policy_noise
    train_man_noise_clip = args.train_man_noise_clip

    if args.env_name in ["Reacher3D-v0"]:
        high = np.array([1., 1., 1., ])
        low = - high
    elif args.env_name in ["Pusher-v0"]:
        high = np.array([2., 2., 2., 2., 2., 2.])
        low = - high
    elif "AntMaze" in args.env_name or "PointMaze" in args.env_name:
        high = np.array((10., 10.))
        low = - high
    else:
        raise NotImplementedError

    man_scale = (high - low) / 2
    absolute_goal_scale = 0

    if args.absolute_goal:
        no_xy = False
    else:
        if "Point" in args.env_name or "Ant" in args.env_name:
            no_xy = True
        else:
            no_xy = False

    obs = env.reset()
    print("obs: ", obs)

    goal = obs["desired_goal"]
    state = obs["observation"]

    controller_goal_dim = obs["achieved_goal"].shape[0]

    tb_path = "{}/{}/{}/{}".format(args.env_name, args.algo, args.version, args.seed)
    writer = SummaryWriter(log_dir=os.path.join(args.log_dir, tb_path))

    # Write Hyperparameters to file
    print("---------------------------------------")
    print("Current Arguments:")
    with open(os.path.join(args.log_dir, tb_path, "hps.txt"), 'w') as f:
        for arg in vars(args):
            print("{}: {}".format(arg, getattr(args, arg)))
            f.write("{}: {}\n".format(arg, getattr(args, arg)))
    print("---------------------------------------\n")

    torch.cuda.set_device(args.gid)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    file_name = "{}_{}_{}".format(args.env_name, args.algo, args.seed)
    output_data = {"frames": [], "reward": [], "dist": []}

    env.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    state_dim = state.shape[0]
    goal_dim = goal.shape[0]
    action_dim = env.action_space.shape[0]

    if args.env_name in ["Reacher3D-v0", "Pusher-v0"]:
        calculate_controller_reward = utils.get_mbrl_fetch_reward_function(env, args.env_name,
                                                                           binary_reward=args.binary_int_reward,
                                                                           absolute_goal=args.absolute_goal)
    elif "Point" in args.env_name or "Ant" in args.env_name:
        calculate_controller_reward = utils.get_reward_function(env, args.env_name,
                                                                absolute_goal=args.absolute_goal,
                                                                binary_reward=args.binary_int_reward)
    else:
        raise NotImplementedError

    controller_buffer = utils.ReplayBuffer(maxsize=args.ctrl_buffer_size,
                                           reward_func=calculate_controller_reward,
                                           reward_scale=args.ctrl_rew_scale)
    manager_buffer = utils.ReplayBuffer(maxsize=args.man_buffer_size)

    controller_policy = higl.Controller(
        state_dim=state_dim,
        goal_dim=controller_goal_dim,
        action_dim=action_dim,
        max_action=max_action,
        actor_lr=args.ctrl_act_lr,
        critic_lr=args.ctrl_crit_lr,
        no_xy=no_xy,
        absolute_goal=args.absolute_goal,
        policy_noise=train_ctrl_policy_noise,
        noise_clip=train_ctrl_noise_clip,
    )

    manager_policy = higl.Manager(
        state_dim=state_dim,
        goal_dim=goal_dim,
        action_dim=controller_goal_dim,
        actor_lr=args.man_act_lr,
        critic_lr=args.man_crit_lr,
        candidate_goals=args.candidate_goals,
        correction=not args.no_correction,
        scale=man_scale,
        goal_loss_coeff=args.goal_loss_coeff,
        absolute_goal=args.absolute_goal,
        absolute_goal_scale=absolute_goal_scale,
        landmark_loss_coeff=args.landmark_loss_coeff,
        delta=args.delta,
        policy_noise=train_man_policy_noise,
        noise_clip=train_man_noise_clip,
        no_pseudo_landmark=args.no_pseudo_landmark,
        automatic_delta_pseudo=args.automatic_delta_pseudo,
        planner_start_step=args.planner_start_step,
        planner_cov_sampling=args.landmark_sampling,
        planner_clip_v=args.clip_v,
        n_landmark_cov=args.n_landmark_coverage,
        planner_initial_sample=args.initial_sample,
        planner_goal_thr=args.goal_thr,
    )

    if args.noise_type == "ou":
        man_noise = utils.OUNoise(goal_dim, sigma=args.man_noise_sigma)
        ctrl_noise = utils.OUNoise(action_dim, sigma=args.ctrl_noise_sigma)

    elif args.noise_type == "normal":
        man_noise = utils.NormalNoise(sigma=args.man_noise_sigma)
        ctrl_noise = utils.NormalNoise(sigma=args.ctrl_noise_sigma)

    if args.load_replay_buffer is not None:
        manager_buffer.load(args.load_replay_buffer + "_manager.npz")
        controller_buffer.load(args.load_replay_buffer + "_controller.npz")
        print("Replay buffers loaded")

    # Initialize adjacency matrix and adjacency network
    n_states = 0
    state_list = []
    state_dict = {}
    adj_mat_size = 1000
    adj_mat = np.diag(np.ones(adj_mat_size, dtype=np.uint8))
    traj_buffer = utils.TrajectoryBuffer(capacity=args.traj_buffer_size)
    if args.algo in ['higl', 'hrac']:
        a_net = ANet(controller_goal_dim, args.r_hidden_dim, args.r_embedding_dim)
        if args.load_adj_net:
            print("Loading adjacency network...")
            a_net.load_state_dict(torch.load("{}/{}_{}_{}_{}_a_network.pth".format(args.load_dir,
                                                                                   args.env_name,
                                                                                   args.algo,
                                                                                   args.version,
                                                                                   args.seed)))
        a_net.to(device)
        optimizer_r = optim.Adam(a_net.parameters(), lr=args.lr_r)
    else:
        a_net = None

    if args.load:
        try:
            manager_policy.load(args.load_dir, args.env_name, args.load_algo, args.version, args.seed)
            controller_policy.load(args.load_dir, args.env_name, args.load_algo, args.version, args.seed)
            print("Loaded successfully.")
            just_loaded = True
        except Exception as e:
            just_loaded = False
            print(e, "Loading failed.")
    else:
        just_loaded = False

    # Logging Parameters
    total_timesteps = 0
    timesteps_since_eval = 0
    timesteps_since_manager = 0
    episode_timesteps = 0
    timesteps_since_subgoal = 0
    episode_num = 0
    done = True
    evaluations = []

    ep_obs_seq = None
    ep_ac_seq = None

    # Novelty PQ and novelty algorithm
    if args.algo == 'higl' and args.use_novelty_landmark:
        if args.novelty_algo == 'rnd':
            novelty_pq = utils.PriorityQueue(args.n_landmark_novelty,
                                             close_thr=args.close_thr,
                                             discard_by_anet=args.discard_by_anet)
            rnd_input_dim = state_dim if not args.use_ag_as_input else controller_goal_dim
            RND = higl.RandomNetworkDistillation(rnd_input_dim, args.rnd_output_dim, args.rnd_lr, args.use_ag_as_input)
            print("Novelty PQ is generated")
        else:
            raise NotImplementedError
    else:
        novelty_pq = None
        RND = None

    while total_timesteps < args.max_timesteps:
        if done:
            # Update Novelty Priority Queue
            if ep_obs_seq is not None:
                assert ep_ac_seq is not None
                if args.algo == 'higl' and args.use_novelty_landmark:
                    if args.novelty_algo == 'rnd':
                        if args.use_ag_as_input:
                            novelty = RND.get_novelty(np.array(ep_ac_seq).copy())
                        else:
                            novelty = RND.get_novelty(np.array(ep_obs_seq).copy())
                        novelty_pq.add_list(ep_obs_seq, ep_ac_seq, list(novelty), a_net=a_net)
                        novelty_pq.squeeze_by_kth(k=args.n_landmark_novelty)
                    else:
                        raise NotImplementedError

            if total_timesteps != 0 and not just_loaded:
                if episode_num % 10 == 0:
                    print("Episode {}".format(episode_num))
                # Train controller
                ctrl_act_loss, ctrl_crit_loss = controller_policy.train(controller_buffer,
                                                                        episode_timesteps,
                                                                        batch_size=args.ctrl_batch_size,
                                                                        discount=args.ctrl_discount,
                                                                        tau=args.ctrl_tau)
                if episode_num % 10 == 0:
                    print("Controller actor loss: {:.3f}".format(ctrl_act_loss))
                    print("Controller critic loss: {:.3f}".format(ctrl_crit_loss))

                writer.add_scalar("data/controller_actor_loss", ctrl_act_loss, total_timesteps)
                writer.add_scalar("data/controller_critic_loss", ctrl_crit_loss, total_timesteps)
                writer.add_scalar("data/controller_ep_rew", episode_reward, total_timesteps)

                # Train manager
                if timesteps_since_manager >= args.train_manager_freq:
                    timesteps_since_manager = 0
                    r_margin = (args.r_margin_pos + args.r_margin_neg) / 2

                    man_act_loss, man_crit_loss, man_goal_loss, man_ld_loss, avg_scaled_norm_direction = \
                        manager_policy.train(args.algo,
                                             controller_policy,
                                             manager_buffer,
                                             controller_buffer,
                                             ceil(episode_timesteps/args.train_manager_freq),
                                             batch_size=args.man_batch_size,
                                             discount=args.discount,
                                             tau=args.man_tau,
                                             a_net=a_net,
                                             r_margin=r_margin,
                                             novelty_pq=novelty_pq,
                                             total_timesteps=total_timesteps
                                             )

                    writer.add_scalar("data/manager_actor_loss", man_act_loss, total_timesteps)
                    writer.add_scalar("data/manager_critic_loss", man_crit_loss, total_timesteps)
                    writer.add_scalar("data/manager_goal_loss", man_goal_loss, total_timesteps)
                    writer.add_scalar("data/manager_landmark_loss", man_ld_loss, total_timesteps)

                    if episode_num % 10 == 0:
                        print("Manager actor loss: {:.3f}".format(man_act_loss))
                        print("Manager critic loss: {:.3f}".format(man_crit_loss))
                        print("Manager goal loss: {:.3f}".format(man_goal_loss))
                        print("Manager landmark loss: {:.3f}".format(man_ld_loss))

                # Evaluate
                if timesteps_since_eval >= args.eval_freq:
                    timesteps_since_eval = 0
                    avg_ep_rew, avg_controller_rew, avg_steps, avg_env_finish, \
                    final_x, final_y, final_z, final_subgoal_x, final_subgoal_y, final_subgoal_z = \
                        evaluate_policy(env, args.env_name, manager_policy, controller_policy,
                                        calculate_controller_reward, args.ctrl_rew_scale,
                                        args.manager_propose_freq, len(evaluations))

                    writer.add_scalar("eval/avg_ep_rew", avg_ep_rew, total_timesteps)
                    writer.add_scalar("eval/avg_controller_rew", avg_controller_rew, total_timesteps)

                    if "Maze" in args.env_name or args.env_name in ["Reacher3D-v0", "Pusher-v0"]:
                        writer.add_scalar("eval/avg_steps_to_finish", avg_steps, total_timesteps)
                        writer.add_scalar("eval/perc_env_goal_achieved", avg_env_finish, total_timesteps)

                    evaluations.append([avg_ep_rew, avg_controller_rew, avg_steps])
                    output_data["frames"].append(total_timesteps)
                    if "Maze" in args.env_name or args.env_name in ["Reacher3D-v0", "Pusher-v0"]:
                        output_data["reward"].append(avg_env_finish)
                    else:
                        output_data["reward"].append(avg_ep_rew)
                    output_data["dist"].append(-avg_controller_rew)

                    if args.save_models:
                        controller_policy.save(args.save_dir, args.env_name, args.algo, args.version, args.seed)
                        manager_policy.save(args.save_dir, args.env_name, args.algo, args.version, args.seed)

                    if args.save_replay_buffer is not None:
                        manager_buffer.save(args.save_replay_buffer + "_manager")
                        controller_buffer.save(args.save_replay_buffer + "_controller")

                # Train adjacency network
                if args.algo in ["higl", "hrac"]:
                    if traj_buffer.full():
                        for traj in traj_buffer.get_trajectory():
                            for i in range(len(traj)):
                                adj_factor = args.adj_factor if args.algo == "higl" else 1
                                for j in range(1, min(int(args.manager_propose_freq * adj_factor), len(traj) - i)):
                                    s1 = tuple(np.round(traj[i]).astype(np.int32))
                                    s2 = tuple(np.round(traj[i + j]).astype(np.int32))
                                    if s1 not in state_list:
                                        state_list.append(s1)
                                        state_dict[s1] = n_states
                                        n_states += 1
                                    if s2 not in state_list:
                                        state_list.append(s2)
                                        state_dict[s2] = n_states
                                        n_states += 1
                                    adj_mat[state_dict[s1], state_dict[s2]] = 1
                                    adj_mat[state_dict[s2], state_dict[s1]] = 1

                        print("Explored states: {}".format(n_states))
                        flags = np.ones((25, 25))
                        for s in state_list:
                            flags[int(s[0]), int(s[1])] = 0
                        print(flags)
                        if not args.load_adj_net:
                            print("Training adjacency network...")
                            utils.train_adj_net(a_net, state_list, adj_mat[:n_states, :n_states],
                                                optimizer_r, args.r_margin_pos, args.r_margin_neg,
                                                n_epochs=args.r_training_epochs, batch_size=args.r_batch_size,
                                                device=device, verbose=True)

                            if args.save_models:
                                r_filename = os.path.join(args.save_dir,
                                                          "{}_{}_{}_{}_a_network.pth".format(args.env_name,
                                                                                             args.algo,
                                                                                             args.version,
                                                                                             args.seed))
                                torch.save(a_net.state_dict(), r_filename)
                                print("----- Adjacency network {} saved. -----".format(episode_num))

                        traj_buffer.reset()

                # Update RND module
                if RND is not None:
                    rnd_loss = RND.train(controller_buffer, episode_timesteps, args.rnd_batch_size)
                    writer.add_scalar("data/rnd_loss", rnd_loss, total_timesteps)

                if len(manager_transition['state_seq']) != 1:
                    manager_transition['next_state'] = state
                    manager_transition['done'] = float(True)
                    manager_buffer.add(manager_transition)

            # Reset environment
            obs = env.reset()
            goal = obs["desired_goal"]
            achieved_goal = obs["achieved_goal"]
            state = obs["observation"]

            ep_obs_seq = [state]  # For Novelty PQ
            ep_ac_seq = [achieved_goal]
            traj_buffer.create_new_trajectory()
            traj_buffer.append(achieved_goal)

            done = False
            episode_reward = 0
            episode_timesteps = 0
            just_loaded = False
            episode_num += 1

            subgoal = manager_policy.sample_goal(state, goal)
            timesteps_since_subgoal = 0
            manager_transition = OrderedDict({
                'state': state,
                'next_state': None,
                'achieved_goal': achieved_goal,
                'next_achieved_goal': None,
                'goal': goal,
                'action': subgoal,
                'reward': 0,
                'done': False,
                'state_seq': [state],
                'actions_seq': [],
                'achieved_goal_seq': [achieved_goal]
            })

        action = controller_policy.select_action(state, subgoal)
        action = ctrl_noise.perturb_action(action, -max_action, max_action)
        action_copy = action.copy()

        next_tup, manager_reward, env_done, _ = env.step(action_copy)

        # Update cumulative reward for the manager
        manager_transition['reward'] += manager_reward * args.man_rew_scale

        next_goal = next_tup["desired_goal"]
        next_achieved_goal = next_tup['achieved_goal']
        next_state = next_tup["observation"]

        traj_buffer.append(next_achieved_goal)
        ep_obs_seq.append(next_state)
        ep_ac_seq.append(next_achieved_goal)

        # Append low level sequence for off policy correction
        manager_transition['actions_seq'].append(action)
        manager_transition['state_seq'].append(next_state)
        manager_transition['achieved_goal_seq'].append(next_achieved_goal)

        controller_reward = calculate_controller_reward(achieved_goal, subgoal, next_achieved_goal,
                                                        args.ctrl_rew_scale, action)
        subgoal = controller_policy.subgoal_transition(achieved_goal, subgoal, next_achieved_goal)

        controller_goal = subgoal
        if env_done:
            done = True

        episode_reward += controller_reward

        # Store low level transition
        if args.inner_dones:
            ctrl_done = done or timesteps_since_subgoal % args.manager_propose_freq == 0
        else:
            ctrl_done = done

        controller_transition = OrderedDict({
            'state': state,
            'next_state': next_state,
            'achieved_goal': achieved_goal,
            'next_achieved_goal': next_achieved_goal,
            'goal': controller_goal,
            'action': action,
            'reward': controller_reward,
            'done': float(ctrl_done),
            'state_seq': [],
            'actions_seq': [],
            'achieved_goal_seq': [],
        })
        controller_buffer.add(controller_transition)

        state = next_state
        goal = next_goal
        achieved_goal = next_achieved_goal

        episode_timesteps += 1
        total_timesteps += 1
        timesteps_since_eval += 1
        timesteps_since_manager += 1
        timesteps_since_subgoal += 1

        if timesteps_since_subgoal % args.manager_propose_freq == 0:
            manager_transition['next_state'] = state
            manager_transition['next_achieved_goal'] = achieved_goal
            manager_transition['done'] = float(done)
            manager_buffer.add(manager_transition)

            subgoal = manager_policy.sample_goal(state, goal)

            if not args.absolute_goal:
                subgoal = man_noise.perturb_action(subgoal, min_action=-man_scale, max_action=man_scale)
            else:
                subgoal = man_noise.perturb_action(subgoal, min_action=-man_scale, max_action=man_scale)

            # Reset number of timesteps since we sampled a subgoal
            writer.add_scalar("data/manager_ep_rew", manager_transition['reward'], total_timesteps)
            timesteps_since_subgoal = 0

            # Create a high level transition
            manager_transition = OrderedDict({
                'state': state,
                'next_state': None,
                'achieved_goal': achieved_goal,
                'next_achieved_goal': None,
                'goal': goal,
                'action': subgoal,
                'reward': 0,
                'done': False,
                'state_seq': [state],
                'actions_seq': [],
                'achieved_goal_seq': [achieved_goal]
            })

    # Final evaluation
    avg_ep_rew, avg_controller_rew, avg_steps, avg_env_finish, \
    final_x, final_y, final_z, final_subgoal_x, final_subgoal_y, final_subgoal_z = \
        evaluate_policy(env, args.env_name, manager_policy, controller_policy, calculate_controller_reward,
                        args.ctrl_rew_scale, args.manager_propose_freq, len(evaluations))

    writer.add_scalar("eval/avg_ep_rew", avg_ep_rew, total_timesteps)
    writer.add_scalar("eval/avg_controller_rew", avg_controller_rew, total_timesteps)

    if "Maze" in args.env_name or args.env_name in ["Reacher3D-v0", "Pusher-v0"]:
        writer.add_scalar("eval/avg_steps_to_finish", avg_steps, total_timesteps)
        writer.add_scalar("eval/perc_env_goal_achieved", avg_env_finish, total_timesteps)

    evaluations.append([avg_ep_rew, avg_controller_rew, avg_steps])
    output_data["frames"].append(total_timesteps)
    if "Maze" in args.env_name or args.env_name in ["Reacher3D-v0", "Pusher-v0"]:
        output_data["reward"].append(avg_env_finish)
    else:
        output_data["reward"].append(avg_ep_rew)
    output_data["dist"].append(-avg_controller_rew)

    if args.save_models:
        controller_policy.save(args.save_dir, args.env_name, args.algo, args.version, args.seed)
        manager_policy.save(args.save_dir, args.env_name, args.algo, args.version, args.seed)

    output_df = pd.DataFrame(output_data)
    output_df.to_csv(os.path.join("./results", file_name+".csv"), float_format="%.4f", index=False)
    print("Training finished.")
