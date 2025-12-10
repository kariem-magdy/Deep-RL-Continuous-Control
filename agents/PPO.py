import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
from utils.NNArch import GaussianPolicy, ValueNetwork

class PPOAgent:
    def __init__(self, state_dim, action_dim, config, use_cnn=False, action_space=None):
        self.gamma = config.get('gamma', 0.99)
        self.lr = config.get('learning_rate', 3e-4)
        self.eps_clip = 0.2
        self.K_epochs = 10
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_cnn = use_cnn
        self.action_space = action_space
        
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.dones = []

        # Pass action_space to ensure correct scaling
        self.policy = GaussianPolicy(state_dim, action_dim, action_space=action_space, use_cnn=use_cnn).to(self.device)
        self.policy_old = GaussianPolicy(state_dim, action_dim, action_space=action_space, use_cnn=use_cnn).to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.critic = ValueNetwork(state_dim, use_cnn=use_cnn).to(self.device)
        
        self.optimizer = optim.Adam([
            {'params': self.policy.parameters(), 'lr': self.lr},
            {'params': self.critic.parameters(), 'lr': self.lr}
        ])
        self.mse_loss = nn.MSELoss()

    def select_action(self, state, evaluate=False):
        with torch.no_grad():
            if self.use_cnn:
                state_t = torch.FloatTensor(state).permute(2,0,1).unsqueeze(0).to(self.device) / 255.0
            else:
                state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            
            # For evaluation, use deterministic mean
            if evaluate:
                _, _, action = self.policy_old.sample(state_t)
                return action.cpu().numpy().flatten()
            else:
                action, log_prob, _ = self.policy_old.sample(state_t)
            
        self.states.append(state)
        self.actions.append(action.cpu().numpy().flatten())
        self.logprobs.append(log_prob.item())
        
        return action.cpu().numpy().flatten()

    def store_transition(self, state, action, reward, done):
        self.rewards.append(reward)
        self.dones.append(done)

    def evaluate_actions(self, states, actions):
        """
        Evaluate log_probs of OLD actions under NEW policy
        Reverses the Tanh transformation to get the pre-tanh distribution
        """
        mean, log_std = self.policy(states)
        std = log_std.exp()
        dist = Normal(mean, std)
        
        # Unscale and Un-tanh the action
        scale = self.policy.scale
        bias = self.policy.bias
        
        action_u = (actions - bias) / scale
        # Clamp to avoid NaN in atanh
        action_u = torch.clamp(action_u, -0.999999, 0.999999)
        x_t = torch.atanh(action_u)
        
        log_probs = dist.log_prob(x_t)
        # Jacobian correction for Tanh
        log_probs -= torch.log(scale * (1 - action_u.pow(2)) + 1e-6)
        
        return log_probs.sum(1, keepdim=True)

    def update(self):
        rewards = []
        discounted_reward = 0
        for reward, done in zip(reversed(self.rewards), reversed(self.dones)):
            if done: discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
            
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        if self.use_cnn:
            old_states = torch.tensor(np.array(self.states), dtype=torch.float32).permute(0,3,1,2).to(self.device) / 255.0
        else:
            old_states = torch.tensor(np.array(self.states), dtype=torch.float32).to(self.device)
            
        old_actions = torch.tensor(np.array(self.actions), dtype=torch.float32).to(self.device)
        old_logprobs = torch.tensor(np.array(self.logprobs), dtype=torch.float32).to(self.device).unsqueeze(1)
        
        for _ in range(self.K_epochs):
            # Evaluate old actions under new policy
            new_logprobs = self.evaluate_actions(old_states, old_actions)
            
            state_values = self.critic(old_states)
            ratios = torch.exp(new_logprobs - old_logprobs)
            
            advantages = rewards.unsqueeze(1) - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            
            loss = -torch.min(surr1, surr2) + 0.5 * self.mse_loss(state_values, rewards.unsqueeze(1))
            
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.states, self.actions, self.logprobs, self.rewards, self.dones = [], [], [], [], []
        return loss.mean().item()

    def save(self, path):
        torch.save(self.policy.state_dict(), path)