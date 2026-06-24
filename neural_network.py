import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import json
import os

from tree_module import MCTS
from connect_4 import mask

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

class policy(nn.Module):
    def __init__(self, name, trunk_structure, policy_structure, value_structure):
        super().__init__()

        self.name = name

        # Conv. Layers ---+

        # Upscale
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=42, kernel_size=3, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(42)

        # ResBlock 1
        self.conv2 = nn.Conv2d(in_channels=42, out_channels=42, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(42)
        self.conv3 = nn.Conv2d(in_channels=42, out_channels=42, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(42)

        # ResBlock 2
        self.conv4 = nn.Conv2d(in_channels=42, out_channels=42, kernel_size=3, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(42)
        self.conv5 = nn.Conv2d(in_channels=42, out_channels=42, kernel_size=3, padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(42)

        # Downscale
        self.conv6 = nn.Conv2d(in_channels=42, out_channels=6, kernel_size=3, padding=1, bias=True)
        self.bn6 = nn.BatchNorm2d(6)

        trunk_layers = []
        policy_layers = []
        value_layers = []

        # Main Trunk
        for i in range(len(trunk_structure) - 1):
            trunk_layers.append(nn.Linear(trunk_structure[i], trunk_structure[i + 1]))

        self.trunk_structure = trunk_structure
        self.trunk_layers = nn.ModuleList(trunk_layers)

        # Policy Arm
        for i in range(len(policy_structure) - 1):
            policy_layers.append(nn.Linear(policy_structure[i], policy_structure[i + 1]))

        self.policy_structure = policy_structure
        self.policy_layers = nn.ModuleList(policy_layers)

        # Value Arm
        for i in range(len(value_structure) - 1):
            value_layers.append(nn.Linear(value_structure[i], value_structure[i + 1]))

        self.value_structure = value_structure
        self.value_layers = nn.ModuleList(value_layers)


    def forward(self, x):
        # Save for Masking
        state_me = x[:42].reshape(6,7) 
        state_them = x[84:].reshape(6,7)

        # Forward Conv.
        x = x.reshape(1,3,6,7)
        x = F.relu(self.bn1(self.conv1(x)))

        # ResBlock 1
        res = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = x + res
        x = F.relu(x)

        # ResBlock 2
        res = x
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.bn5(self.conv5(x))
        x = x + res
        x = F.relu(x)

        # Downscale
        x = F.relu(self.bn6(self.conv6(x)))
        x = x.reshape(252)

        CNN_embedding = x

        # Forward Main
        for i in self.trunk_layers:
            x = F.relu(i(x)) 

        x_policy = x.clone()
        x_value = x.clone()

        # Forward Policy
        for i in self.policy_layers[:-1]:
            x_policy = F.relu(i(x_policy)) 

        x_policy = self.policy_layers[-1](x_policy)

        # Forward Value
        for i in self.value_layers[:-1]:
            x_value = F.relu(i(x_value))

        x_value = self.value_layers[-1](x_value)
        x_value = 2 * torch.sigmoid(x_value) - 1 # Force Value in [-1, 1]

        # Mask Full Columns
        filter_me = mask(state_me)
        filter_them = mask(state_them)
        filter = (filter_me & filter_them)

        to_softmax = x_policy[filter]
        reduced_distribution = F.softmax(to_softmax, dim=0)
        distribution = torch.zeros(7)
        distribution[filter] = reduced_distribution
        output = torch.cat([distribution, x_value.unsqueeze(0)[0]], dim=0)
        output = output.reshape(8)

        return CNN_embedding, output
    
    def fit(self, epochs, nabla, batch_size, generation):
    
        load_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "AlphaZeroDatasets", f"dataset_{generation}.json")
        with open(load_path, "r") as f:
            dataset = json.load(f)
        
        training_inputs = torch.tensor(dataset["States"]).float()
        training_outputs = torch.tensor(dataset["MCTS Visits and Z Scores"]).float()
        
        training_size = training_inputs.shape[0]
        mse = nn.MSELoss()
        optimiser = optim.Adam(self.parameters(), lr=nabla)
        
        for epoch in range(epochs):
            permutation = torch.randperm(training_size)
            training_inputs = training_inputs[permutation]
            training_outputs = training_outputs[permutation]
            
            for i in range(0, training_size, batch_size):
                batch_inputs = training_inputs[i: i + batch_size]
                batch_outputs = training_outputs[i: i + batch_size]
                
                optimiser.zero_grad()
                loss_scalar = 0
                
                # L2 Regularisation
                L2 = 0
                for param in self.parameters():
                    L2 += torch.sum(param ** 2)
                
                # Compute Loss
                for x, y in zip(batch_inputs, batch_outputs):
                    print(x)
                    print(y)
                    _, nn_output = self.forward(x)
                    print(nn_output)
                    print("")
                    value_loss = mse(nn_output[7], y[7])
                    policy_loss = -torch.dot(y[0:7], torch.log(nn_output[0:7] + 1e-8))

                    loss_scalar += value_loss + policy_loss + 1e-4 * L2
                
                # Gradient Descent Step
                loss_scalar.backward()
                optimiser.step()
        
        return "Successfully Trained."