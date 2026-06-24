import torch
from neural_network import policy
from selfplay_functions import alphazero_selfplay

torch.serialization.add_safe_globals([policy])

# ----+

if __name__ == '__main__':
    for generation in range(0, 100):

        policy_network = torch.load(f"AlphaZeroGenerations/generation_{generation}.pt", weights_only=False)
        policy_name = "alphazero.pt"

        alphazero_selfplay(model_path=f"AlphaZeroGenerations/generation_{generation}.pt", generation=generation, cpu_cores=10, game_sets=100)
        policy_network.fit(epochs=10, nabla=0.00002, batch_size=250, generation=generation)

        torch.save(policy_network, f"AlphaZeroGenerations/generation_{generation+1}.pt")
