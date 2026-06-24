import numpy as np
import torch
import copy


def compute_player(state):
    if torch.sum(state) == 0:
        return 1
    else:
        return -1

def mask(state):
    board = state.reshape(6, 7)
    valid_moves = (board[0] == 0)
    full_mask = valid_moves.reshape(7)

    return full_mask


class Grid():
    def __init__(self, state): # State a 6 x 7 tensor, encodes player
        self.state = copy.deepcopy(state)
        self.player = compute_player(self.state)

    def action(self, column):
        mutable_state = self.state

        if mutable_state[0, column] != 0:
            return "# Illegal Action"
        
        else:
            for i in range(6):
                check = mutable_state[5-i, column]
                if int(check.item()) == 0:
                    mutable_state[5-i, column] = 1 # Always +1 Perspective
                    break

            # Finally, Flip
            mutable_state *= -1
            self.player = compute_player(mutable_state)
            
            return "# Legal Action"
        

def winner(state):
    # Check rows   
    for i in range(6):
        row = state[i]
        for j in range(row.size(0) - 3):
            if row[j] == row[j+1] and row[j+1] == row[j+2] and row[j+2] == row[j+3] and int(row[j].item()) != 0:
                return 1

    # Check columns   
    for i in range(7):
        column = state[:, i]
        for j in range(column.size(0) - 3):
            if column[j] == column[j+1] and column[j+1] == column[j+2] and column[j+2] == column[j+3] and int(column[j].item()) != 0:
                return 1
         
    # Check " \ " diagonals

    for i in range(6):
        diagonal = torch.diagonal(state, offset=i-2, dim1=0, dim2=1)
        for j in range(diagonal.size(0) - 3):
            if diagonal[j] == diagonal[j+1] and diagonal[j+1] == diagonal[j+2] and diagonal[j+2] == diagonal[j+3] and int(diagonal[j].item()) != 0:
                return 1
    
    # Check " / " diagonals

    mirror_state = torch.flip(state, dims=[1]) # Flip matrix to check " / " diagonals

    for i in range(6):
        diagonal = torch.diagonal(mirror_state, offset=i-2, dim1=0, dim2=1)
        for j in range(diagonal.size(0) - 3):
            if diagonal[j] == diagonal[j+1] and diagonal[j+1] == diagonal[j+2] and diagonal[j+2] == diagonal[j+3] and int(diagonal[j].item()) != 0:
                return 1
            
    return 0

def visual(state):
    # Red to Move
    if torch.sum(state) == 0:
        for row in state:
            for j in row:
                if int(j.item()) == 1:
                    print("🔴", end=" ")
                elif int(j.item()) == -1:
                    print("🟡", end=" ")
                elif int(j.item()) == 0:
                    print("⚫", end=" ")
            print("")

    # Yellow to Move
    else:
        for row in state:
            for j in row:
                if int(j.item()) == 1:
                    print("🟡", end=" ")
                elif int(j.item()) == -1:
                    print("🔴", end=" ")
                elif int(j.item()) == 0:
                    print("⚫", end=" ")
            print("")