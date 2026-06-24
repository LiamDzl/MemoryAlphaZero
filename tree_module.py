import torch
import torch.nn.functional as F
from connect_4 import Grid, winner, compute_player
import math
import copy
from functions import expand_to_126, cosine_similarity

# Specified in AlphaZero Paper
dirichlet = torch.distributions.Dirichlet(torch.full((7,), 1.))


def PUCT(parent, child, exploration_constant):
    # Compute Prior (Right of Equation)
    prior_score = child.prior * math.sqrt(parent.visit_count + 1) / (child.visit_count + 1)
    prior_score *= exploration_constant

    # Compute Value (Left of Equation)
    if child.visit_count > 0:
        value_score = child.value()
    else:
        value_score = 0

    return value_score + prior_score

class Node: # Initialise on Discovery
    # Player Computed from Parent State
    def __init__(self, prior, player, model):
        self.prior = prior # Parent's probability viewpoint of choosing child (scalar)
        self.player = player
        self.model = model

        self.state = None # (6 x 7)
        self.nn_dist = None # (1 x 7)
        self.value_sum = 0
        self.visit_count = 0
        self.parent = None # Stores single parent node
        self.parent_action = None
        self.children = [None, None, None, None, None, None, None] # Stores children node objects

        self.is_terminal = False
        self.is_root = False

        self.CNN_embedding = None # Useful for Memory Algorithm ...

    def expand(self, state):
         # Once node is created, all we have is scalar prior and whose turn it is...
         # expanding only necessary if we actually decide to explore this action
         self.state = state
         x = state.reshape(42).float()
         x = expand_to_126(x)

         # Forward pass through NN
         CNN_embedding, output_vector = self.model.forward(x)
         output_vector.detach()

         # Grab relevant info
         nn_dist = output_vector[0:7]
         nn_value = output_vector[7] 

         # Now create set of "ghost nodes" for all non-zero probabilities -- 
         # these represent actions only, we've not computed any of these nodes out properly

         for index, probability in enumerate(nn_dist):
            if probability != 0:
                self.children[index] = Node(prior=probability, player=self.player * -1, model=self.model)
                self.children[index].parent = self
                self.children[index].parent_action = index

                # Note child's index in children list represents which action was taken to get from parent to child

         # Save CNN Embedding for Root and Immediate Children   
         if self.is_root == True or self.parent.is_root == True:
             self.CNN_embedding = CNN_embedding.detach()

         return nn_value, nn_dist

    def reset_root(self, state):
         self.state = state.detach().clone()

    def expanded(self):
            if self.children == [None, None, None, None, None, None, None]:
                return False
            else:
                return True

    def value(self):
        if self.visit_count == 0:
             return 0
        return self.value_sum / (self.visit_count)
    
    def best_child(self, exploration_constant):
        current_selection = None
        for child in self.children:
              if child is not None:
                current_selection = child
                break
        
        for child in self.children: # Compare Childrens' PUCT Scores
            if child == None:
                pass

            # Maximise PUCT
            elif PUCT(self, child, exploration_constant=exploration_constant) > PUCT(self, current_selection, exploration_constant=exploration_constant):
                current_selection = child

        return current_selection

class MCTS:
    def __init__(self, model, iterations, root_policy=None):
        self.model = model
        self.iterations = iterations
        self.root_policy = root_policy

        self.explored_nodes = [] # Maintain Node List
        
    def run(self, state, exploration_constant, epsilon, display):
        depth_values = torch.zeros(1, 100) # Stores Depths
        total_wins = 0
        red_wins = 0
        yellow_wins = 0
        player = compute_player(state) # State Encodes Info

        # Create Root
        root = Node(prior=None, player=player, model=self.model)
        root.is_root = True
        root.expand(state=state)

        self.explored_nodes.append(root)

        # Add Dirichlet Noise to Root
        nablas = dirichlet.sample()
        for index, child in enumerate(root.children):
            try:
                child.prior = (1 - epsilon) * child.prior + epsilon * nablas[index].item()
            except:
                pass
        
        # Main Search Loop
        for iteration in range(self.iterations):
            proceed = True
            depth = 0
            root.reset_root(state)
            current_node = root
            path = [current_node]
            
            # Continuously Check Current Node, Traverse Branch
            while current_node.expanded() == True:
                    
                snapshot = current_node.state
                snap_player = current_node.player
                current_node = current_node.best_child(exploration_constant=exploration_constant)
                path.append(current_node)
                depth += 1

                if current_node.is_terminal == True: # Don't Expand Terminal States
                    proceed = False
                    break

                player = player * -1

            # Above Breaks if "current_node" (un)Expanded

            # Select New Node ("current_node") and Expand
            if proceed == True:
                parent_node = current_node.parent
                grid = Grid(state=parent_node.state)
                
                state_save = copy.deepcopy(parent_node.state) # Debugging Error
                grid.action(column=current_node.parent_action) # New State for New Node
                parent_node.state = state_save

                nn_value, nn_dist = current_node.expand(state=grid.state) # Expand Node, Store Value to Backprop. up the Branch
                self.explored_nodes.append(current_node)

                nn_value = nn_value.item()
                nn_value *= -1 # Correct Perspectives

            # Check if Terminal  
            if winner(current_node.state) != 0:
                #graphic(current_node.state)
                #print("")
                current_node.is_terminal = True
                nn_value = 1 # Overwrite Signal. Fix to +1, Backprop will handle +/- 1
    
                colour = -compute_player(current_node.state)
                if colour == 1: # i.e. a win,
                    red_wins += 1

                else:
                    yellow_wins += 1

                total_wins += 1

            # Backpropagate up the Branch
            for history_node in path:
                history_node.visit_count += 1
                history_node.value_sum += (history_node.player * current_node.player) * nn_value

        ### Stats for Testing --------------------------------------------------------------------+
            
            depth_values[0][depth] += 1 # Store Depth

        if display == True:
            print(f"\n🌿 Tree Search Statistics")
            print("")
            print("+------------------------------------------------------------------------------------------+")
            for index, child in enumerate(root.children):
                if child is not None:
                    print(f"| Column {index+1:<2} | "
                    f"Visits: {child.visit_count:>4} | "
                    f"Q Value: {child.value():>8.5f} | "
                    f"U Boost: {exploration_constant * child.prior * math.sqrt(root.visit_count + 1) / (child.visit_count + 1):>8.5f} | "
                    f"Final PUCT: {PUCT(root, child, exploration_constant=exploration_constant):>9.6f} |")
                    print("+------------------------------------------------------------------------------------------+")
                else:
                    print(f"|   Full!   | "
                    f"Visits:    {0:<1} | "
                    f"Q Value: {0:>8.5f} | "
                    f"U Boost: {0:>8.5f} | "
                    f"Final PUCT: {0:>9.6f} |")
                    print("+------------------------------------------------------------------------------------------+")
                    
                    
        # +---------------------------------------------------------------------------------------+
        
        # Return final counts

        mcts_distribution = torch.zeros(1, 7)
        for child_index, child in enumerate(root.children):
            if child != None:
                mcts_distribution[0, child_index] = child.visit_count / self.iterations
                # Normalise visit counts to yield new distribution
    
        if display == True:
            print(f"\n# Final Distribution\n")
            print(mcts_distribution)
            print(f"\n# Total Wins: {total_wins} / {self.iterations}\n")
            print(f"... With 🔴 Winning {red_wins}, vs 🟡 Winning {yellow_wins}\n")

        root_CNN_embedding = root.CNN_embedding
        expected_child_CNN_embedding = torch.zeros(252)

        for child_node in root.children:
            if child_node is not None:
                if child_node.visit_count != 0:
                    expected_child_CNN_embedding += (child_node.visit_count / self.iterations) * child_node.CNN_embedding
        

        return root.value(), root_CNN_embedding, expected_child_CNN_embedding, mcts_distribution
    
