import gymnasium as gym
import torch
import numpy as np
from copy import deepcopy


from config import args
from model import Actor , Critic
from replay_buffer import ReplayBuffer
from learning_curve import LearningCurve
from ground_truth_curve import GroundTruthCurve
from train import train_actors, train_critics
from evaluate import evaluate_actors
from select_survey_corps import select_survey_corps


###### Create the environment ######
env = gym.make(args.env_name)


###### Set the random seed ######
np.random.seed(args.seed)
torch.manual_seed(args.seed)
env.reset(seed=args.seed)
env.action_space.seed(args.seed)


###### Determine the dimensions ######
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = env.action_space.high


###### Initialize actors, critics, critic_targets ######
actors: list[Actor] = []
for i in range(args.actor_size):
    actor = Actor(state_dim, action_dim, max_action).to(args.device)
    actors.append(actor)

critics: list[Critic] = []
for i in range(args.critic_size):
    critic = Critic(state_dim, action_dim).to(args.device)
    critics.append(critic)

critic_targets = deepcopy(critics)


###### Initialize actor_optimizers, critic_optimizers ######
actor_optimizers: list[torch.optim.Adam] = []
for actor in actors:
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=args.actor_learning_rate)
    actor_optimizers.append(actor_optimizer)

critic_optimizers: list[torch.optim.Adam] = []
for critic in critics:
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=args.critic_learning_rate)
    critic_optimizers.append(critic_optimizer)


###### Initialize the replay buffer ######
replay_buffer = ReplayBuffer()


###### Initialize the learning curve ######
learning_curve = LearningCurve(actors)
ground_truth_curve = GroundTruthCurve(actors, critics, replay_buffer)


###### Initialize certain parameters ######
survey_corps_size = int(np.ceil(np.sqrt(args.actor_size)))
exploration_noise = args.exploration_noise * max_action


###### Reinitialize the environment ######
state , _ = env.reset()


###### Start training ######
for steps in range(args.max_steps):

    if steps < args.start_steps:

        action = env.action_space.sample()  # Purely random action

    else:

        train_critics(critics, critic_targets, actors, critic_optimizers, replay_buffer)
        train_actors(actors, critics, actor_optimizers, replay_buffer)
        evaluation_table = evaluate_actors(actors, critics, replay_buffer)
        survey_corps = select_survey_corps(actors, survey_corps_size)
        actor = survey_corps[np.random.randint(survey_corps_size)]
        with torch.no_grad():
            state_ = torch.tensor(state, dtype=torch.float32, device=args.device)
            action = actor(state_)
            action = action.cpu().numpy()
        noise = np.random.normal(0, exploration_noise, size=action_dim)
        action = (action + noise).clip(-max_action , max_action)


    state_info = (deepcopy(env.unwrapped.data.qpos), deepcopy(env.unwrapped.data.qvel))

    next_state , reward , done , reach_step_limit , _ = env.step(action)

    replay_buffer.push(state, action, next_state, reward, not done, state_info)


    if done or reach_step_limit:
        state , _ = env.reset()
    else:
        state = next_state


    if steps >= args.start_steps:
        learning_curve.update(evaluation_table)
    learning_curve.add_step()


    if learning_curve.steps >= args.start_steps:
        if learning_curve.steps % args.test_performance_freq == 0:
            print(f"steps={learning_curve.steps}  score={learning_curve.LC_scores[-1]:.3f}")
        elif learning_curve.steps % 100 == 0:
            print(f"steps={learning_curve.steps}")



    ground_truth_curve.add_step()
    if ground_truth_curve.steps >= args.start_steps:
        if ground_truth_curve.steps % 5000 == 0:
            print(f"GTR={ground_truth_curve.GTC_ground_truth_return[-1]:.3f} ER={ground_truth_curve.GTC_estimate_return[-1]:.3f} RD={ground_truth_curve.GTC_return_difference[-1]:.3f}")




###### Save the result ######
if args.save_result == True:
    learning_curve.save()
    ground_truth_curve.save()


print("Finish")












