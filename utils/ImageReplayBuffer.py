import numpy as np
import torch

class ImageReplayBuffer:
    def __init__(self, capacity, image_shape, action_dim):
        self.capacity = capacity
        self.ptr = 0
        self.size = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.state = np.zeros((capacity, *image_shape), dtype=np.uint8)
        self.action = np.zeros((capacity, action_dim), dtype=np.float32)
        self.reward = np.zeros((capacity, 1), dtype=np.float32)
        self.next_state = np.zeros((capacity, *image_shape), dtype=np.uint8)
        self.done = np.zeros((capacity, 1), dtype=np.float32)

    def add(self, state, action, reward, next_state, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.reward[self.ptr] = reward
        self.next_state[self.ptr] = next_state
        self.done[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)
        state = torch.FloatTensor(self.state[ind]).permute(0, 3, 1, 2).to(self.device) / 255.0
        next_state = torch.FloatTensor(self.next_state[ind]).permute(0, 3, 1, 2).to(self.device) / 255.0
        return (
            state,
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            next_state,
            torch.FloatTensor(self.done[ind]).to(self.device)
        )