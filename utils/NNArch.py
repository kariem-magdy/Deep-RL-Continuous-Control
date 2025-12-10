import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

class CNNFeatureExtractor(nn.Module):
    """
    Extracts features from 96x96x3 images (CarRacing). 
    Output: 256-dim feature vector.
    """
    def __init__(self, input_channels=3):
        super(CNNFeatureExtractor, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc = nn.Linear(64 * 8 * 8, 256)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        return F.relu(self.fc(x))

class GaussianPolicy(nn.Module):
    """For SAC & PPO (Stochastic Policy)"""
    def __init__(self, input_dim, action_dim, hidden_dim=256, action_space=None, use_cnn=False):
        super(GaussianPolicy, self).__init__()
        self.use_cnn = use_cnn
        
        if use_cnn:
            self.feature_extractor = CNNFeatureExtractor(input_dim)
            input_dim = 256

        self.l1 = nn.Linear(input_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)

        if action_space is None:
            self.scale = torch.tensor(1.0)
            self.bias = torch.tensor(0.0)
        else:
            self.register_buffer('scale', torch.FloatTensor((action_space.high - action_space.low) / 2.))
            self.register_buffer('bias', torch.FloatTensor((action_space.high + action_space.low) / 2.))

    def forward(self, x):
        if self.use_cnn:
            x = self.feature_extractor(x)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        return self.mean(x), torch.clamp(self.log_std(x), -20, 2)

    def sample(self, x):
        mean, log_std = self.forward(x)
        std = log_std.exp()
        dist = Normal(mean, std)
        x_t = dist.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.scale + self.bias
        
        log_prob = dist.log_prob(x_t)
        log_prob -= torch.log(self.scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        
        mean_action = torch.tanh(mean) * self.scale + self.bias
        return action, log_prob, mean_action

class DeterministicPolicy(nn.Module):
    """For TD3 (Deterministic Policy)"""
    def __init__(self, input_dim, action_dim, max_action, hidden_dim=256, use_cnn=False):
        super(DeterministicPolicy, self).__init__()
        self.use_cnn = use_cnn
        self.max_action = max_action

        if use_cnn:
            self.feature_extractor = CNNFeatureExtractor(input_dim)
            input_dim = 256

        self.l1 = nn.Linear(input_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        if self.use_cnn:
            x = self.feature_extractor(x)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        return self.max_action * torch.tanh(self.l3(x))

class DoubleQNetwork(nn.Module):
    """For SAC & TD3 (Critic)"""
    def __init__(self, state_dim, action_dim, hidden_dim=256, use_cnn=False):
        super(DoubleQNetwork, self).__init__()
        self.use_cnn = use_cnn

        if use_cnn:
            self.feature_extractor = CNNFeatureExtractor(state_dim)
            state_dim = 256

        self.l1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, 1)
        
        self.l4 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.l5 = nn.Linear(hidden_dim, hidden_dim)
        self.l6 = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        if self.use_cnn:
            state = self.feature_extractor(state)
            
        xu = torch.cat([state, action], 1)
        
        x1 = F.relu(self.l1(xu))
        x1 = F.relu(self.l2(x1))
        x1 = self.l3(x1)

        x2 = F.relu(self.l4(xu))
        x2 = F.relu(self.l5(x2))
        x2 = self.l6(x2)
        return x1, x2

class ValueNetwork(nn.Module):
    """For PPO (Value Function)"""
    def __init__(self, state_dim, hidden_dim=256, use_cnn=False):
        super(ValueNetwork, self).__init__()
        self.use_cnn = use_cnn
        
        if use_cnn:
            self.feature_extractor = CNNFeatureExtractor(state_dim)
            state_dim = 256

        self.l1 = nn.Linear(state_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        if self.use_cnn:
            x = self.feature_extractor(x)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        return self.v(x)