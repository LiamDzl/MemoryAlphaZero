import torch
from neural_network import policy

parameters = policy(trunk_structure=[252],
                    policy_structure=[252, 210, 168, 126, 7],
                    value_structure=[252, 210, 168, 126, 1],
                    name="generation_0")

torch.save(parameters, f"AlphaZeroGenerations/generation_0.pt")