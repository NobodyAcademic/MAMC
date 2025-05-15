import torch

from config import args
from model import Actor


def select_survey_corps(actors: list[Actor], survey_corps_size: int) -> list[Actor]:

    with torch.no_grad():

        min_skill = min([actor.skill for actor in actors])
        max_skill = max([actor.skill for actor in actors])

        min_creativity = min([actor.creativity for actor in actors])
        max_creativity = max([actor.creativity for actor in actors])

        fitnesses = torch.zeros(size=(args.actor_size,), dtype=torch.float32, device=args.device)

        if max_skill - min_skill > 1e-18:
            for i in range(args.actor_size):
                fitnesses[i] += (actors[i].skill - min_skill) / (max_skill - min_skill)

        if max_creativity - min_creativity > 1e-18:
            for i in range(args.actor_size):
                fitnesses[i] += (actors[i].creativity - min_creativity) / (max_creativity - min_creativity)

        survey_corps_indices = fitnesses.argsort(descending=True)
        survey_corps_indices = survey_corps_indices[ : survey_corps_size]

    survey_corps = []
    for index in survey_corps_indices:
        survey_corps.append(actors[index])

    return survey_corps




