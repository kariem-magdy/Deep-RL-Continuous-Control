import torch
import torch.nn.functional as F
import torch.optim as optim
from utils.NNArch import DeterministicPolicy, DoubleQNetwork
from utils.ReplayBuffer import ReplayBuffer
from utils.ImageReplayBuffer import ImageReplayBuffer

class TD3Agent:
    def __init__(self, state_dim, action_dim, max_action, config, use_cnn=False):
        self.gamma = config.get('gamma', 0.99)
        self.tau = config.get('tau', 0.005)
        self.policy_noise = 0.2
        self.noise_clip = 0.5
        self.policy_freq = 2
        self.max_action = max_action
        self.total_it = 0
        self.use_cnn = use_cnn
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.actor = DeterministicPolicy(state_dim, action_dim, max_action, use_cnn=use_cnn).to(self.device)
        self.actor_target = DeterministicPolicy(state_dim, action_dim, max_action, use_cnn=use_cnn).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=3e-4)

        self.critic = DoubleQNetwork(state_dim, action_dim, use_cnn=use_cnn).to(self.device)
        self.critic_target = DoubleQNetwork(state_dim, action_dim, use_cnn=use_cnn).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=3e-4)

        if use_cnn:
            self.memory = ImageReplayBuffer(config.get('buffer_size', 100000), (96,96,3), action_dim)
        else:
            self.memory = ReplayBuffer(config.get('buffer_size', 100000), state_dim, action_dim)

    def select_action(self, state, evaluate=False):
        if self.use_cnn:
            state = torch.FloatTensor(state).permute(2,0,1).unsqueeze(0).to(self.device) / 255.0
        else:
            state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        return self.actor(state).cpu().data.numpy().flatten()

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)

    def update(self, batch_size):
        self.total_it += 1
        if self.memory.size < batch_size: return 0

        state, action, reward, next_state, not_done = self.memory.sample(batch_size)
        not_done = 1 - not_done

        with torch.no_grad():
            noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)

            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + not_done * self.gamma * target_Q

        current_Q1, current_Q2 = self.critic(state, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        if self.total_it % self.policy_freq == 0:
            actor_loss = -self.critic(state, self.actor(state))[0].mean()
            self.actor_optim.zero_grad()
            actor_loss.backward()
            self.actor_optim.step()

            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        return critic_loss.item()
    
    def save(self, path):
        torch.save(self.actor.state_dict(), path)