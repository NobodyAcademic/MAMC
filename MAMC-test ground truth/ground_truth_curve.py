import gymnasium as gym
import torch
import numpy as np

import os
import json
import datetime
import random
import string

from config import args
from model import Actor, Critic
from replay_buffer import ReplayBuffer



class GroundTruthCurve:

    def __init__(self, actors: list[Actor], critics: list[Critic], replay_buffer: ReplayBuffer):

        self.steps = 0

        self.actors = actors
        self.critics = critics
        self.replay_buffer = replay_buffer


        self.GTC_steps = []
        self.GTC_steps.append(0)

        self.GTC_ground_truth_return = []
        self.GTC_ground_truth_return.append(0)

        self.GTC_estimate_return = []
        self.GTC_estimate_return.append(0)

        self.GTC_return_difference = []
        self.GTC_return_difference.append(0)


        self.GTC_estimate_return_distribution = np.zeros(shape=(3,20), dtype=np.int32)
        self.GTC_ground_truth_return_distribution = np.zeros(shape=(3,20), dtype=np.int32)



    def add_step(self):

        self.steps += 1

        if self.steps % 5000 == 0:

            if self.steps >= args.start_steps:

                self.GTC_steps.append(self.steps)

                with torch.no_grad():

                    batch_size = 30

                    ##################################################################################################

                    replays = self.replay_buffer.sample(batch_size=batch_size)
                    states = torch.stack([replay.state for replay in replays])

                    multi_actions = []
                    for actor in self.actors:

                        action = actor(states)

                        multi_actions.append(action)

                    multi_states = torch.cat([states] * args.actor_size, dim=0)  # shape = (actor_size * batch_size , state_dim)
                    multi_actions = torch.cat(multi_actions, dim=0)              # shape = (actor_size * batch_size , action_dim)


                    evaluation_table = []
                    for critic in self.critics:

                        multi_Qs = critic(multi_states , multi_actions)          # shape = (actor_size * batch_size , 1)
                        multi_Qs = multi_Qs.view(args.actor_size , batch_size)   # shape = (actor_size , batch_size)

                        evaluation_table.append(multi_Qs)

                    evaluation_table = torch.stack(evaluation_table)  # shape = (critic_size , actor_size , batch_size)

                    ##################################################################################################

                    env = gym.make(args.env_name)

                    multi_ground_truth_returns = []

                    for i in range(batch_size):

                        discount_returns = []

                        for actor in self.actors:

                            env.reset()
                            qpos, qvel = replays[i].state_info
                            env.unwrapped.set_state(qpos, qvel)

                            state = replays[i].state
                            state = state.cpu().numpy()
                            done = False
                            reach_step_limit = False

                            discount_return = 0.
                            discount = 1.
                            while (not done) and (not reach_step_limit):

                                state = torch.tensor(state, dtype=torch.float32, device=args.device)
                                action = actor(state)
                                action = action.cpu().numpy()

                                state , reward , done , reach_step_limit , _ = env.step(action)

                                discount_return += discount * reward
                                discount *= args.gamma

                            discount_returns.append(discount_return)

                        discount_returns = torch.tensor(discount_returns, dtype=torch.float32, device=args.device)  # shape = (actor_size)
                        multi_ground_truth_returns.append(discount_returns)

                    ##################################################################################################

                    multi_ground_truth_returns = torch.stack(multi_ground_truth_returns, dim=1)  # shape = (actor_size , batch_size)
                    ground_truth_returns = multi_ground_truth_returns.quantile(q=0.5, dim=0)     # shape = (batch_size)

                    evaluation_table_mean = evaluation_table.mean(dim=0)                         # shape = (actor_size , batch_size)
                    estimate_returns = evaluation_table_mean.quantile(q=0.5, dim=0)              # shape = (batch_size)

                    self.GTC_ground_truth_return.append(ground_truth_returns.mean().cpu().item())
                    self.GTC_estimate_return.append(estimate_returns.mean().cpu().item())
                    self.GTC_return_difference.append((estimate_returns - ground_truth_returns).abs().mean().cpu().item())

                    ##################################################################################################

                    if self.steps <= (args.max_steps * (1 / 3)):
                        phase = 0
                    elif self.steps <= (args.max_steps * (2 / 3)):
                        phase = 1
                    else:
                        phase = 2

                    for k in range(batch_size):

                        for j in range(args.actor_size):

                            Qs = evaluation_table[ : , j , k]                 # shape = (critic_size)
                            ground_truth = multi_ground_truth_returns[j , k]  # shape = (1)

                            Q_min = Qs.min()
                            Q_max = Qs.max()

                            if (Q_max - Q_min) > 1e-18:

                                slice_length = (Q_max - Q_min) / 20
                                offset = Q_min

                                for i in range(19):

                                    self.GTC_estimate_return_distribution[phase][i] += ((Qs >= offset) & (Qs < (offset + slice_length))).sum().item()
                                    
                                    if i == 0:
                                        if ground_truth < (offset + slice_length):
                                            self.GTC_ground_truth_return_distribution[phase][i] += 1
                                    else:
                                        if (ground_truth >= offset) and (ground_truth < (offset + slice_length)):
                                            self.GTC_ground_truth_return_distribution[phase][i] += 1

                                    offset += slice_length

                                self.GTC_estimate_return_distribution[phase][-1] += (Qs >= offset).sum().item()
                                if ground_truth >= offset:
                                    self.GTC_ground_truth_return_distribution[phase][-1] += 1



    def save(self):

        if (not os.path.exists(args.output_path)):
            os.makedirs(args.output_path)

        file_name = f"[{args.algorithm}][{args.env_name}][{args.seed}][{datetime.date.today()}][Ground Truth Curve][{''.join(random.choices(string.ascii_uppercase, k=6))}].json"
        path = os.path.join(args.output_path, file_name)

        result = {
            "Config": vars(args),
            "Ground Truth Curve": {
                "Steps": self.GTC_steps,
                "Ground Truth Return": self.GTC_ground_truth_return,
                "Estimate Return": self.GTC_estimate_return,
                "Return Difference": self.GTC_return_difference,

                "Estimate Return Distribution": self.GTC_estimate_return_distribution.tolist(),
                "Ground Truth Return Distribution": self.GTC_ground_truth_return_distribution.tolist()
            }
        }

        with open(path, mode="w") as file:

            json.dump(result, file)


