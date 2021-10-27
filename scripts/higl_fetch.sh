ENV=$1
TIMESTEPS=$2
GPU=$2
SEED=$3

CUDA_VISIBLE_DEVICES=${GPU} python main.py \
--absolute_goal \
--delta 1.0 \
--env_name ${ENV} \
--reward_shaping "sparse" \
--algo higl \
--seed ${SEED} \
--max_timesteps  ${TIMESTEPS} \
--manager_propose_freq 5 \
--landmark_sampling fps \
--n_landmark_coverage 20 \
--use_novelty_landmark \
--novelty_algo rnd \
--n_landmark_novelty 20 \
--ctrl_noise_sigma 0.1 \
--man_noise_sigma 0.2 \
--train_ctrl_policy_noise 0.1 \
--train_man_policy_noise 0.2 \
--ctrl_rew_scale 0.1 \
--r_margin_pos 0.01 \
--r_margin_neg 0.012 \
--close_thr 0.02 \
--clip_v -15 \
--goal_thr -5 \
--version "sparse"
