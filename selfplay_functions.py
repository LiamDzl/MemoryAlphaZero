import torch
import random
import os
import json

from neural_network import policy
from connect_4 import mask, winner, Grid, compute_player
from tree_module import MCTS
from functions import colour, generate_400, softmax_temp, expand_to_126, graphic
from multiprocessing import Pool

def selfplay(policy_network, initial_state, mcts_depth, exploration_constant, epsilon, gamma, game_set, game_sets, root_policy):

    X = [] # Recorded States (Inputs)
    Y = [] # Recorded Distribution + Z Scores

    U = [] # MCTS Root Values
    V = [] # Root CNN Embeddings
    W = [] # Expected Child CNN Embeddings

    Y_Hat = [] # NN Priors

    game_state = initial_state
    initial_player = compute_player(initial_state)
    player = compute_player(game_state)
    end_filter = torch.tensor([False, False, False, False, False, False, False])
    filter = torch.tensor([True, True, True, True, True, True, True])
    game_length = 0

    # Recorded Game Data ------+

    recorded_states = []
    recorded_mcts_dists = []

    while winner(game_state) == 0 and not torch.equal(filter, end_filter):
    
        print(f"\n# {colour(player)}'s Move\n")
        graphic(game_state)
        print("\n")

        tree = MCTS(model=policy_network, iterations=mcts_depth, root_policy=root_policy)
        MCTS_value, CNN_embedding, expected_child_CNN_embedding, mcts_distribution = tree.run(state=game_state,
                                                                                              exploration_constant=exploration_constant,
                                                                                              epsilon=epsilon,
                                                                                              display=False)
        
        U.append(MCTS_value)
        V.append(CNN_embedding)
        W.append(expected_child_CNN_embedding)
   
        # Data collection
        x = game_state.reshape(42)
        x = expand_to_126(x)
        recorded_states.append(x)
        y = mcts_distribution.reshape(7)
        recorded_mcts_dists.append(y)

        # Choose Move
        chosen_move = torch.argmax(mcts_distribution)
        grid = Grid(state=game_state)
        grid.action(chosen_move)

        print(f"# Decision:")
        print(mcts_distribution)
        print("")
        print(f"# 🕹️  Set: {game_set} / {game_sets}")

        # New state
        game_state = grid.state
        player *= -1
        game_length += 1

        # As to end when board full
        filter = mask(game_state)

    if winner(game_state) == 0:
        z = 0

    if winner(game_state) == 1:
        z = compute_player(game_state) * -1 # True Winner, +1 Red, -1 Yellow

    # Concatenate z values to distribution vectors - as to make vectors of length 8 from 7
    dist_value_outputs = []
    z = z * initial_player # if starting player and result match, then reward the initial state... (+1)
    for index, y in enumerate(recorded_mcts_dists):
        discounted = z * (gamma ** (game_length - index - 1)) # Discount Reward
        dist_value_outputs.append(torch.cat((y, torch.tensor([discounted])), dim=0))
        z *= -1

    graphic(game_state)
    print("\n")

    # Append game results into global dataset, X and Y
    for recorded_state in recorded_states:
        X.append(recorded_state)
        _, y_hat = policy_network.forward(recorded_state)
        y_hat = y_hat.detach()
        Y_Hat.append(y_hat)

    for dist_value_output in dist_value_outputs:
        Y.append(dist_value_output)

    return X, Y, U, V, W, Y_Hat  # Statistics

# CPU Parallelisation

def worker(args):
    policy_network, initial_state, mcts_depth, exploration_constant, epsilon, gamma, game_set, game_sets, root_policy = args
    X, Y, U, V, W, Y_Hat = selfplay(policy_network=policy_network,
                        initial_state=initial_state,
                        mcts_depth=mcts_depth,
                        exploration_constant=exploration_constant,
                        epsilon=epsilon,
                        gamma=gamma,
                        game_set=game_set,
                        game_sets=game_sets,
                        root_policy=root_policy)

    return X, Y, U, V, W, Y_Hat

# Generate First 400 States in Connect 4
initial_states = generate_400()
    
def parallel_selfplay(game_set, game_sets, policy_network, mcts_depth, exploration_constant, epsilon, gamma, cpu_cores, root_policy=None):
    random.shuffle(initial_states)
    args_list = [(policy_network, state, mcts_depth, exploration_constant, epsilon, gamma, game_set, game_sets, root_policy) for state in initial_states[:cpu_cores]]

    with Pool(cpu_cores) as p:
        dataset = p.map(worker, args_list)

    # Compute Dataset Size:
    dataset_size = 0
    for X, Y, U, V, W, Y_Hat in dataset:
        dataset_size += len(X)

    # Compute Tensors:
    X_total = torch.zeros(dataset_size, 126)
    Y_total = torch.zeros(dataset_size, 8)
    U_total = []
    V_total = []
    W_total = []
    Y_Hat_total = []

    index = 0
    for X, Y, U, V, W, Y_Hat in dataset:
        for input, output in zip(X, Y):
            X_total[index] = input
            Y_total[index] = output
            index += 1
        U_total.extend(U)
        V_total.extend(V)
        W_total.extend(W)
        Y_Hat_total.extend(Y_Hat)

    print(f"Set Complete. New Data Points: {dataset_size}")
    return X_total, Y_total, U_total, V_total, W_total, Y_Hat_total

def alphazero_selfplay(model_path, generation, cpu_cores, game_sets):

    torch.serialization.add_safe_globals([policy])
    policy_network = torch.load(model_path, weights_only=False)
    policy_network.name = "model"

    # Hyperparameters
    mcts_depth = 250
    exploration_constant = 3.5
    epsilon = 0.25
    gamma = 1

    X_all = []
    Y_all = []
    U_all = []
    V_all = []
    W_all = []
    Y_Hat_all = []

    for game_set in range(game_sets):
        X, Y, U, V, W, Y_Hat = parallel_selfplay(game_set=game_set,
                                game_sets=game_sets,
                                policy_network=policy_network,
                                mcts_depth=mcts_depth,
                                exploration_constant=exploration_constant,
                                epsilon=epsilon,
                                gamma=gamma,
                                cpu_cores=cpu_cores,
                                root_policy=None)
        
        U = -torch.tensor(U) # Negative Since Perspective Change
        V = torch.stack(V)
        W = torch.stack(W)
        Y_Hat = torch.stack(Y_Hat)

        X_all.append(X)
        Y_all.append(Y)
        U_all.append(U)
        V_all.append(V)
        W_all.append(W)
        Y_Hat_all.append(Y_Hat)
    
    X_final = torch.cat(X_all, dim=0)
    Y_final = torch.cat(Y_all, dim=0)
    U_final = torch.cat(U_all, dim=0)
    V_final = torch.cat(V_all, dim=0)
    W_final = torch.cat(W_all, dim=0)
    Y_Hat_final = torch.cat(Y_Hat_all, dim=0)

    # Save to JSON
    dataset = {
        "Generation": generation,
        "States": X_final.tolist(),
        "MCTS Visits and Z Scores": Y_final.tolist(),
        "MCTS Values": U_final.tolist(),
        "CNN Embeddings": V_final.tolist(),
        "Expected CNN Child Embeddings": W_final.tolist(),
        "NN Priors": Y_Hat_final.tolist()
    }

    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "AlphaZeroDatasets")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"dataset_{generation}.json")

    with open(save_path, "w") as f:
        json.dump(dataset, f)
