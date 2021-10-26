import argparse

from higl.train import run_higl

parser = argparse.ArgumentParser()

# Environment
parser.add_argument("--env_name", required=True, type=str)
parser.add_argument("--reward_shaping", type=str, default="dense", choices=["dense", "sparse"])
parser.add_argument("--stochastic_xy", action="store_true")
parser.add_argument("--stochastic_sigma", default=0., type=float)
parser.add_argument("--gid", type=int, default=0)
parser.add_argument("--algo", default="higl", type=str, choices=["higl", "hrac", "hiro"])
parser.add_argument("--seed", default=2, type=int)
parser.add_argument("--eval_freq", default=5e3, type=float)
parser.add_argument("--max_timesteps", default=5e6, type=float)

# Off-policy correction (from HIRO)
parser.add_argument("--no_correction", action="store_true")
parser.add_argument("--inner_dones", action="store_true")
parser.add_argument("--absolute_goal", action="store_true")
parser.add_argument("--binary_int_reward", action="store_true")

# Manager Parameters
parser.add_argument("--man_tau", default=0.005, type=float)
parser.add_argument("--man_batch_size", default=128, type=int)
parser.add_argument("--man_buffer_size", default=2e5, type=int)
parser.add_argument("--man_rew_scale", default=0.1, type=float)
parser.add_argument("--man_act_lr", default=1e-4, type=float)
parser.add_argument("--man_crit_lr", default=1e-3, type=float)
parser.add_argument("--candidate_goals", default=10, type=int)
parser.add_argument("--manager_propose_freq", "-k", default=10, type=int)
parser.add_argument("--train_manager_freq", default=10, type=int)
parser.add_argument("--discount", default=0.99, type=float)

# Controller Parameters
parser.add_argument("--ctrl_tau", default=0.005, type=float)
parser.add_argument("--ctrl_batch_size", default=128, type=int)
parser.add_argument("--ctrl_buffer_size", default=2e5, type=int)
parser.add_argument("--ctrl_rew_scale", default=1.0, type=float)
parser.add_argument("--ctrl_act_lr", default=1e-4, type=float)
parser.add_argument("--ctrl_crit_lr", default=1e-3, type=float)
parser.add_argument("--ctrl_discount", default=0.95, type=float)

# Noise Parameters
parser.add_argument("--noise_type", default="normal", type=str, choices=["normal", "ou"])
parser.add_argument("--ctrl_noise_sigma", default=1., type=float)
parser.add_argument("--man_noise_sigma", default=1., type=float)
parser.add_argument("--train_ctrl_policy_noise", default=0.2, type=float)
parser.add_argument("--train_ctrl_noise_clip", default=0.5, type=float)
parser.add_argument("--train_man_policy_noise", default=0.2, type=float)
parser.add_argument("--train_man_noise_clip", default=0.5, type=float)

# Adjacency Network (from HRAC)
parser.add_argument("--traj_buffer_size", type=int, default=50000)
parser.add_argument("--lr_r", type=float, default=2e-4)
parser.add_argument("--r_margin_pos", type=float, default=1.0)
parser.add_argument("--r_margin_neg", type=float, default=1.2)
parser.add_argument("--r_training_epochs", type=int, default=25)
parser.add_argument("--r_batch_size", type=int, default=64)
parser.add_argument("--r_hidden_dim", type=int, default=128)
parser.add_argument("--r_embedding_dim", type=int, default=32)
parser.add_argument("--goal_loss_coeff", type=float, default=20.)

# HIGL
parser.add_argument("--landmark_loss_coeff", default=20., type=float)
parser.add_argument("--delta", type=float, default=2)
parser.add_argument("--adj_factor", default=0.5, type=float)

# HIGL: Planner, Coverage
parser.add_argument("--landmark_sampling", type=str, choices=["fps", "none"])
parser.add_argument('--clip_v', type=float, default=-38., help="clip bound for the planner")
parser.add_argument("--n_landmark_coverage", type=int, default=20)
parser.add_argument("--initial_sample", type=int, default=1000)
parser.add_argument("--goal_thr", type=float, default=-10.)
parser.add_argument("--planner_start_step", type=int, default=60000)

# HIGL: Novelty
parser.add_argument("--novelty_algo", type=str, default="none", choices=["rnd", "none"])
parser.add_argument("--use_novelty_landmark", action="store_true")
parser.add_argument("--close_thr", type=float, default=0.2)
parser.add_argument("--n_landmark_novelty", type=int, default=20)
parser.add_argument("--rnd_output_dim", type=int, default=128)
parser.add_argument("--rnd_lr", type=float, default=1e-3)
parser.add_argument("--rnd_batch_size", default=128, type=int)
parser.add_argument("--use_ag_as_input", action="store_true")

# Ablation
parser.add_argument("--no_pseudo_landmark", action="store_true")
parser.add_argument("--discard_by_anet", action="store_true")
parser.add_argument("--automatic_delta_pseudo", action="store_true")

# Save
parser.add_argument("--save_models", action="store_true")
parser.add_argument("--save_dir", default="./models", type=str)
parser.add_argument("--save_replay_buffer", type=str)

# Load
parser.add_argument("--load", action="store_true")
parser.add_argument("--load_dir", default="./models", type=str)
parser.add_argument("--load_algo", type=str)
parser.add_argument("--log_dir", default="./logs", type=str)
parser.add_argument("--load_replay_buffer", type=str)
parser.add_argument("--load_adj_net", default=False, action="store_true")

parser.add_argument("--version", type=str, default='v0')

args = parser.parse_args()

if args.load_algo is None:
    args.load_algo = args.algo

# Run the algorithm
run_higl(args)
