import pandas as pd
import torch
import time
import pickle
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical, Normal
from tqdm import tqdm
import numpy as np

class PolicyNetwork(nn.Module):
    
    def __init__(self, observation_space, action_space):
        super(PolicyNetwork, self).__init__()
        self.input_layer1 = nn.Linear(observation_space, observation_space)
        self.input_layer2 = nn.Linear(observation_space, observation_space)
        self.input_layer3 = nn.Linear(observation_space, observation_space)
        self.input_layer4 = nn.Linear(observation_space, observation_space)
        self.output_layer = nn.Linear(observation_space, action_space)
        self.do = nn.Dropout(0.25)
    
    #forward pass
    def forward(self, x):
        #input states
        y1 = self.input_layer1(x)
        y1 = F.relu(y1 + x)
        y1 = self.do(y1)
        y2 = self.input_layer2(y1)
        y2 = F.relu(y2 + y1)
        y2 = self.do(y2)
        y3 = self.input_layer3(y2)
        y3 = F.relu(y3 + y2)
        y3 = self.do(y3)
        y4 = self.input_layer4(y3)
        y4 = F.relu(y4 + y3)
        #actions
        actions = self.output_layer(y4)
        action_probs = F.softmax(actions, dim=1)
        return action_probs
    
class StateValueNetwork(nn.Module):
    
    #Takes in state
    def __init__(self, observation_space):
        super(StateValueNetwork, self).__init__()
        
        self.input_layer1 = nn.Linear(observation_space, observation_space)
        self.input_layer2 = nn.Linear(observation_space, observation_space)
        self.input_layer3 = nn.Linear(observation_space, observation_space)
        self.input_layer4 = nn.Linear(observation_space, observation_space)
        self.output_layer = nn.Linear(observation_space, 1)
        self.do = nn.Dropout(0.25)
        
    def forward(self, x):
        #input states
        y1 = self.input_layer1(x)
        y1 = F.relu(y1 + x)
        y1 = self.do(y1)
        y2 = self.input_layer2(y1)
        y2 = F.relu(y2 + y1)
        y2 = self.do(y2)
        y3 = self.input_layer3(y2)
        y3 = F.relu(y3 + y2)
        y3 = self.do(y3)
        y4 = self.input_layer4(y3)
        y4 = F.relu(y4 + y3)
        
        #get state value
        state_value = self.output_layer(y4)
        
        return state_value
    
def select_action(network, state):
    state = torch.from_numpy(state).float().to(DEVICE)
    action_probs = network(state)
    state = state.detach()
    
    #sample an action 
    m = Categorical(action_probs)
    action = m.sample()
    
    #return action
    return action.item(), m.log_prob(action)
