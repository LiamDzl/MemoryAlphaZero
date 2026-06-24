import torch
import numpy as np
import os
import random
import json
import copy
from connect_4 import Grid, compute_player, visual
from rich import print as richprint
from rich.panel import Panel
from rich.columns import Columns
from rich.padding import Padding

def softmax_temp(dist, temp):
    if temp == 0:
        index = torch.argmax(dist, dim=1)
        output = torch.zeros(1, 7)
        output[0, index] = 1
        return output

    else:
        dist = dist.float()
        mask = (dist == 0)
        logits = torch.log(dist + 1e-12) / temp
        logits = logits - torch.max(logits)
        output = torch.softmax(logits, dim=1)
        output[mask] = 0
        return output

def colour(x):
    if x == 1:
        return "Red"
    else:
        return "Yellow"
    
def generate_400():
    output = []

    empty = torch.zeros((6, 7), dtype=torch.int)
    output.append(empty)

    # depth 1 (7 states)
    depth1 = []
    for i in range(7):
        grid = Grid(state=copy.deepcopy(empty))
        grid.action(column=i)
        depth1.append(grid.state)
        output.append(grid.state)

    # depth 2 (49 states)
    depth2 = []
    for state in depth1:
        for j in range(7):
            grid = Grid(state=copy.deepcopy(state))
            grid.action(column=j)
            depth2.append(grid.state)
            output.append(grid.state)

    # depth 3 (343 states)
    for state in depth2:
        for k in range(7):
            grid = Grid(state=copy.deepcopy(state))
            grid.action(column=k)
            output.append(grid.state)

    return output

def generate_random(lambda_value, games):
    output = []

    for _ in range(games):
        state = torch.zeros((6, 7), dtype=torch.int)
        grid = Grid(state=state.clone())

        # sample depth from Poisson
        depth = np.random.poisson(lambda_value)

        # cap at max possible moves (42 for 6x7 board)
        depth = min(depth, 42)

        for _ in range(depth):
            # find valid columns (top cell empty)
            valid_columns = [c for c in range(7) if grid.state[0, c] == 0]
            if not valid_columns:
                break  # board full

            col = random.choice(valid_columns)
            grid.action(column=col)

        output.append(grid.state.clone())

    return output


def bar_matrix_emoji(dist, height, colour):
    space = " "
    # Normalise
    max_val = dist.max().item()
    if max_val == 0:
        heights = [0] * len(dist)
    else:
        heights = [int((v / max_val) * height) for v in dist]

    # Build matrix top-down
    rows = []
    for row in range(height, 0, -1):
        line = ""
        for index, h in enumerate(heights):
            if h >= row:
                line += f"{colour} "
            else:
                line += "⬛ "
        rows.append(line.rstrip())

    return "\n".join(rows)

def alphazero_display(policy_name, state, tree_dist, tree_value, neural_dist, neural_value, agent_move, noise):
    print("")
    print("+ ------------ AlphaZero UI ------------- +")
    print("")
    print(f"<state>:")
    print("")
    graphic(state)
    print("")
    print(f"<model>: {policy_name}")
    print("")
    print(f"<\dist\\tree>: {tree_dist}")
    print(f"<\dist\\neural>: {neural_dist}")
    print("")

    if -tree_value >= 0:
        richprint(f"<\\value\\tree>: [bold green]{-tree_value}")
    else:
        richprint(f"<\\value\\tree>: [bold red]{-tree_value}")

    if neural_value.item() >= 0:
        richprint(f"<\\value\\neural>: [bold green]{neural_value}")
    else:
        richprint(f"<\\value\\neural>: [bold red]{neural_value}")
    print("")
    print(f"<\decision\\tree>:      <\decision\\neural>:")
    print("")
    g1 = bar_matrix_emoji(tree_dist[0], height=6, colour="⬜")
    g2 = bar_matrix_emoji(neural_dist, height=6, colour="⬜")
    output = Columns([g1, Padding(g2, (0, 0, 0, 2))])
    richprint(output)
    print("")
    print(f"# 🌀 Agent's Choice: {agent_move+1}")
    print("")

def expand_to_126(x42):
    board = x42.reshape(6, 7)

    current = (board == 1).float()
    empty = (board == 0).float()
    opponent = (board == -1).float()

    x = torch.stack([current, empty, opponent], dim=0)

    return x.reshape(126)

def contract_to_42(x126):
    current = x126[:42]
    opponent = x126[84:]

    board = current - opponent
    return board

def graphic(state):
    if state.size()[0] == 126:
        visual(contract_to_42(state).reshape(6,7))
    else:
        visual(state)

def diverge_distribution(beta, child_dist, parent_dist):
    modulated = child_dist * torch.exp(beta * (child_dist - parent_dist))
    scale = torch.sum(modulated)
    return modulated / scale

def jsd(p, q, eps=1e-12):
    p = p.clamp(min=eps)
    q = q.clamp(min=eps)

    p = p / p.sum(dim=-1, keepdim=True)
    q = q / q.sum(dim=-1, keepdim=True)

    m = 0.5 * (p + q)

    js = 0.5 * (p * (p.log() - m.log())).sum(dim=-1) + \
         0.5 * (q * (q.log() - m.log())).sum(dim=-1)

    js = torch.clamp(js, min=0.0)
    return torch.sqrt(js)

def cosine_similarity(x, y):
    return torch.dot(x, y) / (torch.norm(x) * torch.norm(y))

def load_dataset():
    load_path = os.path.join("AlphaZeroDatasets", f"dataset.json")
    with open(load_path, "r") as f:
        dataset = json.load(f)
    
    X = torch.tensor(dataset["States"]).float() 
    Y = torch.tensor(dataset["MCTS Visits and Z Scores"]).float()
    U = torch.tensor(dataset["MCTS Values"]).float() 
    V = torch.tensor(dataset["CNN Embeddings"]).float() 
    W = torch.tensor(dataset["Expected CNN Child Embeddings"]).float()
    
    return X, Y, U, V, W

def dirichlet_smooth(p, alpha=1e-6):
    p = p + alpha
    return p / p.sum(dim=-1, keepdim=True)