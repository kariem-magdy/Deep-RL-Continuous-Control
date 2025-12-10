import torch
import torch.nn.functional as F
import torch.optim as optim
from utils.NNArch import GaussianPolicy, DoubleQNetwork
from utils.ReplayBuffer import ReplayBuffer
from utils.ImageReplayBuffer import ImageReplayBuffer

class SACAgent:
    def __init__(self, state_dim, action_dim, action_space, config, use_cnn=False):
        self.gamma = config.get('gamma', 0.99)
        self.tau = config.get('tau', 0.005)
        self.alpha = config.get('alpha', 0.2)
        self.lr = config.get('learning_rate', 3e-4)
        self.use_cnn = use_cnn
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy = GaussianPolicy(state_dim, action_dim, action_space=action_space, use_cnn=use_cnn).to(self.device)
        self.critic = DoubleQNetwork(state_dim, action_dim, use_cnn=use_cnn).to(self.device)
        self.critic_target = DoubleQNetwork(state_dim, action_dim, use_cnn=use_cnn).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.policy_optim = optim.Adam(self.policy.parameters(), lr=self.lr)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=self.lr)

        self.target_entropy = -torch.prod(torch.Tensor(action_space.shape).to(self.device)).item()
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optim = optim.Adam([self.log_alpha], lr=self.lr)

        if use_cnn:
            self.memory = ImageReplayBuffer(config.get('buffer_size', 100000), (96,96,3), action_dim)
        else:
            self.memory = ReplayBuffer(config.get('buffer_size', 100000), state_dim, action_dim)

    def select_action(self, state, evaluate=False):
        if self.use_cnn:
            state = torch.FloatTensor(state).permute(2, 0, 1).unsqueeze(0).to(self.device) / 255.0
        else:
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            
        if evaluate:
            _, _, action = self.policy.sample(state)
        else:
            action, _, _ = self.policy.sample(state)
        return action.detach().cpu().numpy()[0]

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)

    def update(self, batch_size):
        if self.memory.size < batch_size: return 0
        state, action, reward, next_state, mask = self.memory.sample(batch_size)

        with torch.no_grad():
            next_action, next_log_prob, _ = self.policy.sample(next_state)
            q1_next, q2_next = self.critic_target(next_state, next_action)
            min_q_next = torch.min(q1_next, q2_next) - self.alpha * next_log_prob
            next_q_value = reward + (1 - mask) * self.gamma * min_q_next

        q1, q2 = self.critic(state, action)
        q_loss = F.mse_loss(q1, next_q_value) + F.mse_loss(q2, next_q_value)

        self.critic_optim.zero_grad()
        q_loss.backward()
        self.critic_optim.step()

        pi, log_pi, _ = self.policy.sample(state)
        q1_pi, q2_pi = self.critic(state, pi)
        min_q_pi = torch.min(q1_pi, q2_pi)
        policy_loss = ((self.alpha * log_pi) - min_q_pi).mean()

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()
        self.alpha = self.log_alpha.exp()

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

        return q_loss.item()
        
    def save(self, path):
        torch.save(self.policy.state_dict(), path)