import pickle
import pandas as pd
import torch
import time
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical, Normal
from tqdm import tqdm
import numpy as np

class PolicyNetwork(nn.Module):
    def __init__(self, observation_space, action_space):
        super(PolicyNetwork, self).__init__()
        self.input_layer = nn.Linear(observation_space, 32)
        self.mu = nn.Linear(32, action_space)
        self.log_std = nn.Linear(32, action_space)
    
    #forward pass
    def forward(self, x):
        #input states
        x = self.input_layer(x)
        #relu activation
        x = F.relu(x)
        #actions
        mu = self.mu(x)
        log_std = self.log_std(x)
        std = log_std.exp()
        
        return mu, std
    

class PolicyResNet(nn.Module):
    
    #Takes in observations and outputs actions
    def __init__(self, observation_space, action_space):
        super(PolicyNetwork, self).__init__()
        self.input_layer1 = nn.Linear(observation_space, 32)
        self.input_layer2 = nn.Linear(32, 32)
        self.input_layer3 = nn.Linear(32, 32)
        self.input_layer4 = nn.Linear(32, 32)
        self.mu = nn.Linear(32, action_space)
        self.log_std = nn.Linear(32, action_space)
    
    #forward pass
    def forward(self, x):
        #input states
        y = self.input_layer1(x)
        y = F.relu(y)
        
        z = self.input_layer2(y)
        z = z + y
        z = F.relu(z)
        
        k = self.input_layer3(z)
        k = k + z
        k = F.relu(k)
        
        j = self.input_layer4(k)
        j = j + k
        j = F.relu(j)

        #actions
        mu = self.mu(j)
        log_std = self.log_std(j)
        std = log_std.exp()
        
        return mu, std


class StateValueResNet(nn.Module):
    
    #Takes in state
    def __init__(self, observation_space):
        super(StateValueNetwork, self).__init__()
        
        self.input_layer1 = nn.Linear(observation_space, 32)
        self.input_layer2 = nn.Linear(32, 32)
        self.input_layer3 = nn.Linear(32, 32)
        self.input_layer4 = nn.Linear(32, 32)
        self.output_layer = nn.Linear(32, 1)
        
    def forward(self, x):
        #input states
        y = self.input_layer1(x)
        y = F.relu(y)
        
        z = self.input_layer2(y)
        z = z + y
        z = F.relu(z)
        
        k = self.input_layer3(z)
        k = k + z
        k = F.relu(k)
        
        j = self.input_layer4(k)
        j = j + k
        j = F.relu(j)
        
        #get state value
        state_value = self.output_layer(j)
        
        return state_value


class StateValueNetwork(nn.Module):
    
    #Takes in state
    def __init__(self, observation_space):
        super(StateValueNetwork, self).__init__()
        
        self.input_layer = nn.Linear(observation_space, 32)
        self.output_layer = nn.Linear(32, 1)
        
    def forward(self, x):
        #input layer
        x = self.input_layer(x)
        
        #activiation relu
        x = F.relu(x)
        
        #get state value
        state_value = self.output_layer(x)
        
        return state_value
    
def select_action(network, state):
    state = torch.from_numpy(state).float().to(DEVICE)
    
    #use network to predict action probabilities
    mu, std = network(state)
    state = state.detach()
    #sample an action
    m = Normal(mu, std)
    action = m.sample()

    return action.numpy(), m.log_prob(action)
