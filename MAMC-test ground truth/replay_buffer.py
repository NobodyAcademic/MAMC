import torch
import numpy as np

from config import args


class Replay:

    def __init__(self, state, action, reward, next_state, not_done, state_info):

        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.not_done = not_done

        self.target_Q = None

        self.state_info = state_info


class ReplayBuffer:

    def __init__(self):

        self.buffer: list[Replay] = []
        self.size: int = 0


    def push(self, state: np.ndarray, action: np.ndarray, next_state: np.ndarray, reward: np.float64, not_done: bool, state_info: tuple):

        state = torch.tensor(state, dtype=torch.float32, device=args.device)
        action = torch.tensor(action, dtype=torch.float32, device=args.device)
        next_state = torch.tensor(next_state, dtype=torch.float32, device=args.device)
        reward = torch.tensor(reward, dtype=torch.float32, device=args.device)
        not_done = torch.tensor(not_done, dtype=torch.float32, device=args.device)

        replay = Replay(state, action, reward, next_state, not_done, state_info)

        self.buffer.append(replay)

        self.check_size()


    def sample(self, batch_size=args.batch_size, indices=None) -> list[Replay]:

        if indices is None:
            indices = np.random.randint(self.size, size=batch_size)

        return [self.buffer[i] for i in indices]


    def check_size(self):   # Maintain the size of the replay buffer

        if len(self.buffer) > args.replay_buffer_size:
            self.buffer = self.buffer[-args.replay_buffer_size : ]

        self.size = len(self.buffer)





