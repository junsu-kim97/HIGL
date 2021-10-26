# HIGL
This is a PyTorch implementation for our paper: Landmark-Guided Subgoal Generation in Hierarchical Reinforcement Learning (NeurIPS 2021).

Our code is based on official implementation of [HRAC](https://github.com/trzhang0116/HRAC) (NeurIPS 2020) and
[Map-planner](https://github.com/FangchenLiu/map_planner) (NeurIPS 2019)
## Installation
```
conda create -n higl python=3.6
conda activate higl
./install_all.sh
```

Also, to run the MuJoCo experiments, a license is required (see [here](https://www.roboti.us/license.html)).

## Usage
### Training & Evaluation
- Point Maze
```
./scripts/higl_point_maze.sh ${reward_shaping} ${timesteps} ${gpu} ${seed}
./scripts/higl_point_maze.sh dense 5e5 0 2
./scripts/higl_point_maze.sh sparse 5e5 0 2
```

- Ant Maze (U-shape)
```
./scripts/higl_ant_maze_u.sh ${reward_shaping} ${timesteps} ${gpu} ${seed}
./scripts/higl_ant_maze_u.sh dense 10e5 0 2
./scripts/higl_ant_maze_u.sh sparse 10e5 0 2
```

- Ant Maze (W-shape)
```
./scripts/higl_ant_maze_w.sh ${reward_shaping} ${timesteps} ${gpu} ${seed}
./scripts/higl_ant_maze_w.sh dense 10e5 0 2
./scripts/higl_ant_maze_w.sh sparse 10e5 0 2
```

- Reacher & Pusher
```
./scripts/higl_fetch.sh ${env} ${gpu} ${seed}
./scripts/higl_fetch.sh Reacher3D-v0 0 2
./scripts/higl_fetch.sh Pusher-v0 0 2
```

- Stochastic Ant Maze (U-shape)
```
./scripts/higl_ant_maze_u_stoch.sh ${reward_shaping} ${timesteps} ${gpu} ${seed}
./scripts/higl_ant_maze_u_stoch.sh dense 10e5 0 2
./scripts/higl_ant_maze_u_stoch.sh sparse 10e5 0 2
```

