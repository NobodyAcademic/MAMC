import argparse


parser = argparse.ArgumentParser()


# General settings
parser.add_argument("--algorithm", type = str, default = "MAMC", help = "Algorithm name")


# Saving settings
parser.add_argument("--save_result", action = "store_true", help = "Whether to save result")
parser.add_argument("--output_path", type = str, default = "Result", help = "Output file path")


# Experiment settings
parser.add_argument("--env_name", type = str, default = "HalfCheetah-v5", help = "Task name")
parser.add_argument("--device", type = str, default = "cuda:0", help = "Device used for the experiment")
parser.add_argument("--seed", type = int, default = 0, help = "Random seed")
parser.add_argument("--max_steps", type = int, default = int(3e5), help = "Maximum number of environment steps")


# Performance evaluation settings
parser.add_argument("--test_performance_freq", type = int, default = 1000, help = "Evaluate actor performance every N environment steps")
parser.add_argument("--test_n", type = int, default = 20, help = "Number of episodes per evaluation")


# Reinforcement Learning settings
parser.add_argument("--replay_buffer_size", type = int, default = int(1e6), help = "Maximum capacity of the replay buffer")
parser.add_argument("--batch_size", type = int, default = 256, help = "Random mini-batch size")
parser.add_argument("--start_steps", type = int, default = 5000, help = "Use random actions to initialize the replay buffer")
parser.add_argument("--gamma", type = float, default = 0.99, help = "Discount factor for Temporal Difference Learning")


# MAMC settings
parser.add_argument("--actor_size", type = int, default = 10, help = "Number of actors")
parser.add_argument("--critic_size", type = int, default = 10, help = "Number of critics")
parser.add_argument("--smr_ratio", type = int, default = 10)
parser.add_argument("--actor_learning_rate", type = float, default = 1e-4, help = "Learning rate of the actor")
parser.add_argument("--critic_learning_rate", type = float, default = 3e-4, help = "Learning rate of the critic")
parser.add_argument("--exploration_noise", type = float, default = 0.1, help = "Add noise to action during interaction to enhance diversity")
parser.add_argument("--tau", type = float, default = 0.005, help = "Soft update rate for target networks")
parser.add_argument("--quantile_q", type = float, default = 0.2, help = "Quantile q value used for the ensemble critic")


args = parser.parse_args()