import gymnasium as gym
import torch

import os
import json
import datetime
import random
import string

from config import args
from model import Actor


class LearningCurve:

    def __init__(self, actors: list[Actor]):

        self.steps = 0
        self.LC_steps = []
        self.LC_scores = []
        self.LC_Qs = []
        self.LC_Qs_difference = []

        self.actors = actors
        self.evaluation_table = None

        self.test_initial_performance()


    def test_initial_performance(self):

        actor_representative = self.actors[0]

        self.LC_steps.append(0)
        self.LC_scores.append(self.test_performance(actor_representative))
        self.LC_Qs.append(0)
        self.LC_Qs_difference.append(0)


    def update(self, evaluation_table: torch.Tensor):

        self.evaluation_table = evaluation_table


    def add_step(self):

        self.steps += 1

        if self.steps % args.test_performance_freq == 0:

            self.LC_steps.append(self.steps)

            if self.steps > args.start_steps:

                actor_representative = max(self.actors, key=lambda actor: actor.skill)

                self.LC_scores.append(self.test_performance(actor_representative))
                with torch.no_grad():
                    evaluation_table_quantile = self.evaluation_table.quantile(q=args.quantile_q, dim=0)
                    self.LC_Qs.append(evaluation_table_quantile.mean().cpu().item())
                    self.LC_Qs_difference.append((self.evaluation_table - evaluation_table_quantile).abs().mean().cpu().item())

            else:

                self.LC_scores.append(self.LC_scores[-1])
                self.LC_Qs.append(0)
                self.LC_Qs_difference.append(0)


    def test_performance(self, actor: Actor):

        env = gym.make(args.env_name)
        env.reset(seed=args.seed + 555)

        avg_score = 0.

        for t in range(args.test_n):

            state , _ = env.reset()
            done = False
            reach_step_limit = False

            while (not done) and (not reach_step_limit):

                with torch.no_grad():
                    state = torch.tensor(state, dtype=torch.float32, device=args.device)
                    action = actor(state)
                    action = action.cpu().numpy()

                state , reward , done , reach_step_limit , _ = env.step(action)

                avg_score += reward

        avg_score = avg_score / args.test_n

        return avg_score


    def save(self):

        if (not os.path.exists(args.output_path)):
            os.makedirs(args.output_path)

        file_name = f"[{args.algorithm}][{args.env_name}][{args.seed}][{datetime.date.today()}][Learning Curve][{''.join(random.choices(string.ascii_uppercase, k=6))}].json"
        path = os.path.join(args.output_path, file_name)

        result = {
            "Config": vars(args),
            "Learning Curve": {
                "Steps": self.LC_steps,
                "Score": self.LC_scores,
                "Q value": self.LC_Qs,
                "Q value difference": self.LC_Qs_difference
            }
        }

        with open(path, mode="w") as file:

            json.dump(result, file)