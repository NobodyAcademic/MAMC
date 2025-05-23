import torch
import torch.nn.functional as F
import numpy as np

from config import args
from model import Actor, Critic
from replay_buffer import ReplayBuffer


gamma = torch.tensor(args.gamma, dtype=torch.float32, device=args.device)
tau = torch.tensor(args.tau, dtype=torch.float32, device=args.device)

critic_size_tensor = torch.tensor(args.critic_size, dtype=torch.float32, device=args.device)
batch_size_tensor = torch.tensor(args.batch_size, dtype=torch.float32, device=args.device)

order = 0


def train_actors(actors: list[Actor], critics: list[Critic], actor_optimizers: list[torch.optim.Adam], replay_buffer: ReplayBuffer):

    global order

    indices = np.random.randint(replay_buffer.size, size=(args.actor_size, args.batch_size))

    for i in range(args.smr_ratio):

        critic = critics[order]

        multi_states = []
        multi_actions = []
        for j in range(args.actor_size):

            replays = replay_buffer.sample(indices=indices[j])
            states = torch.stack([replay.state for replay in replays])  # shape = (batch_size , state_dim)

            actions = actors[j](states)  # shape = (batch_size , action_dim)

            multi_states.append(states)
            multi_actions.append(actions)

        multi_states = torch.cat(multi_states, dim=0)    # shape = (actor_size * batch_size , state_dim)
        multi_actions = torch.cat(multi_actions, dim=0)  # shape = (actor_size * batch_size , action_dim)

        multi_Qs = critic(multi_states , multi_actions)  # shape = (actor_size * batch_size , 1)

        loss = -multi_Qs.sum() / critic_size_tensor / batch_size_tensor

        for j in range(args.actor_size):
            actor_optimizers[j].zero_grad()

        loss.backward()

        for j in range(args.actor_size):
            actor_optimizers[j].step()


        order = (order + 1) % args.critic_size



def train_critics(critics: list[Critic], critic_targets: list[Critic], actors: list[Actor], critic_optimizers: list[torch.optim.Adam], replay_buffer: ReplayBuffer, exploration_noise: torch.Tensor, max_action: torch.Tensor):

    indices = np.random.randint(replay_buffer.size, size=(args.critic_size, args.batch_size))

    for i in range(args.smr_ratio):

        calculate_target_Q(critic_targets, actors, replay_buffer, indices, exploration_noise, max_action)

        for j in range(args.critic_size):

            replays = replay_buffer.sample(indices=indices[j])

            states = torch.stack([replay.state for replay in replays])
            actions = torch.stack([replay.action for replay in replays])
            target_Qs = torch.stack([replay.target_Q for replay in replays]).unsqueeze_(dim=1)

            Qs = critics[j](states , actions)

            loss = F.mse_loss(Qs , target_Qs)

            critic_optimizers[j].zero_grad()
            loss.backward()
            critic_optimizers[j].step()

            with torch.no_grad():
                for param, target_param in zip(critics[j].parameters(), critic_targets[j].parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)



def calculate_target_Q(critic_targets: list[Critic], actors: list[Actor], replay_buffer: ReplayBuffer, indices: np.ndarray, exploration_noise: torch.Tensor, max_action: torch.Tensor):

    unique_indices = np.unique(indices)
    replays = replay_buffer.sample(indices=unique_indices)

    with torch.no_grad():

        next_states = torch.stack([replay.next_state for replay in replays])  # shape = (replay_size , state_dim)

        multi_next_actions = []
        for actor in actors:

            next_actions = actor(next_states)  # shape = (replay_size , action_dim)

            multi_next_actions.append(next_actions)

        multi_next_states = torch.cat([next_states] * args.actor_size, dim=0)  # shape = (actor_size * replay_size , state_dim)
        multi_next_actions = torch.cat(multi_next_actions, dim=0)              # shape = (actor_size * replay_size , action_dim)

        noise = torch.randn_like(multi_next_actions) * exploration_noise
        multi_next_actions = (multi_next_actions + noise).clamp(-max_action , max_action)

        all_next_Qs = []
        for critic_target in critic_targets:

            multi_next_Qs = critic_target(multi_next_states, multi_next_actions)  # shape = (actor_size * replay_size , 1)
            multi_next_Qs = multi_next_Qs.view(args.actor_size, len(replays))     # shape = (actor_size , replay_size)

            all_next_Qs.append(multi_next_Qs)

        all_next_Qs = torch.stack(all_next_Qs)                            # shape = (critic_target_size , actor_size , replay_size)

        next_Qs = all_next_Qs.quantile(q=args.quantile_q, dim=0)          # shape = (actor_size , replay_size)
        next_Qs = next_Qs.quantile(q=0.5, dim=0)                          # shape = (replay_size)

        rewards = torch.stack([replay.reward for replay in replays])      # shape = (replay_size)
        not_dones = torch.stack([replay.not_done for replay in replays])  # shape = (replay_size)

        target_Qs = rewards + not_dones * gamma * next_Qs                 # shape = (replay_size)

        for i in range(len(replays)):
            replays[i].target_Q = target_Qs[i]





