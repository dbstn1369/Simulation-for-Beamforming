import torch
import torch.nn as nn
import random

class Actor(nn.Module):
    def __init__(self, num_states, num_actions):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(num_states, 256)  # 뉴런 수 증가
        self.fc2 = nn.Linear(256, 512)  # 뉴런 수 증가
        self.fc3 = nn.Linear(512, 512)  # 새로운 은닉층 추가
        self.fc4 = nn.Linear(512, num_actions)  # 새로운 은닉층 추가
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))  # 새로운 은닉층 추가
        x = self.softmax(self.fc4(x))  # 새로운 은닉층 추가
        return x
    

class Critic(nn.Module):
    def __init__(self, num_states):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(num_states, 256)  # 뉴런 수 증가
        self.fc2 = nn.Linear(256, 512)  # 뉴런 수 증가
        self.fc3 = nn.Linear(512, 512)  # 새로운 은닉층 추가
        self.fc4 = nn.Linear(512, 1)  # 새로운 은닉층 추가

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))  # 새로운 은닉층 추가
        x = self.fc4(x)  # 새로운 은닉층 추가
        return x
    

class MemoryBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, state, action, reward, next_state):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = (state, action, reward, next_state)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)   