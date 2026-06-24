import torch
from neural_network import policy
from connect_4 import Grid, winner, compute_player
from tree_module import MCTS
from functions import alphazero_display, expand_to_126, graphic

torch.serialization.add_safe_globals([policy])
policy_network = torch.load("AlphaZeroGenerations/generation_100.pt", weights_only=False)
policy_name = "alphazero.pt"

initial = torch.zeros(6,7)

environment = Grid(state=initial)

column = ""
move = 0
exploration_constant = 3.5
epsilon = 0
search_depth = 250
noise = 0

text = r"""    _    _       _           _____              
   / \  | |_ __ | |__   __ _|__  /___ _ __ ___  
  / _ \ | | '_ \| '_ \ / _` | / // _ \ '__/ _ \ 
 / ___ \| | |_) | | | | (_| |/ /|  __/ | | (_) |
/_/___\_\_| .__/|_| |_|\__,_/____\___|_|_ \___/ 
 / ___|___|_| __  _ __   ___  ___| |_  | || |   
| |   / _ \| '_ \| '_ \ / _ \/ __| __| | || |_  
| |__| (_) | | | | | | |  __/ (__| |_  |__   _| 
 \____\___/|_| |_|_| |_|\___|\___|\__|    |_|   """

print(text)
print("\n")


proceed = False
while not proceed:
    player_colour = input("🔴 / 🟡 ? : ")
    print("")

    if player_colour == "end":
            column = "end"
            break
    
    if player_colour in {"red", "Red", "1", "r"}:
        emoji = "🔴"
        player_colour = 1
        agent_colour = -1
        graphic(environment.state)
        print("")
        proceed = True

    elif player_colour in {"yellow", "Yellow", "2", "y"}:
        emoji = "🟡"
        player_colour = -1
        agent_colour = 1
   
        tree_search = MCTS(model=policy_network, iterations=search_depth, root_policy=None)
        _, _, _, distribution = tree_search.run(state=environment.state,
                                       exploration_constant=exploration_constant,
                                       epsilon=epsilon,
                                       display=True)

        state = environment.state.reshape(42)
        _, output = policy_network.forward(expand_to_126(state))

        neural_distribution = output[:7]
        value = output[7]
        
        root_node = tree_search.explored_nodes[0]
        print(root_node.state)
        root_value = root_node.value_sum / root_node.visit_count

        agent_move = torch.argmax(distribution)

        # Display Network Heads
        alphazero_display(policy_name=policy_name,
                          state=environment.state,
                          tree_dist=distribution,
                          tree_value=root_value,
                          neural_dist=neural_distribution,
                          neural_value=value,
                          agent_move=agent_move,
                          noise=noise)

        environment.action(column=int(agent_move.item()))
        graphic(environment.state)
        print("")
        proceed = True

    else:
        print("❌ Error: Invalid Colour\n")


while (column != "end"):

    proceed = False
    inrange = False
    
    while not proceed:
        column = input(f"{emoji} Select Column (1-7) : ")
        print(f"\n+ --------------- ( Move {move} )")
        colNum = -1
        if column == "end":
            break
        try:
            colNum = int(column)
        except:
            print("\n❌ Error: Not in Range\n")

        if colNum >= 0:
            if colNum >= 1 and colNum <= 7:
                proceed = True
                inrange = True

            if inrange == True:
                if environment.state[0, int(column) - 1] != 0:
                    print("\n❌ Error: Column Full!\n")
                    proceed = False
            
            else:
                print("\n❌ Error: Not in Range\n") 
            

    if column == "end":
        break

    else:
        environment.action(column=int(column)-1)
        environment.player = agent_colour
        move += 1

        if winner(environment.state) == 1:
            player = compute_player(environment.state)
            if player == 1:
                print("\n🟡🟡🟡🟡 Yellow Wins! 🟡🟡🟡🟡\n")
                graphic(environment.state)
                print("\n")
                print("")
                column = "end"
            else:
                print("\n🔴🔴🔴🔴 Red Wins! 🔴🔴🔴🔴\n")
                graphic(environment.state)
                print("")
                column = "end"

        print("")

        if column == "end":
            break

        else:
            graphic(environment.state)
        
        tree_search = MCTS(model=policy_network, iterations=search_depth, root_policy=None)
        _, _, _, distribution = tree_search.run(state=environment.state,
                                       exploration_constant=exploration_constant,
                                       epsilon=epsilon,
                                       display=True)
        
        state = environment.state.reshape(42)
        _, output = policy_network.forward(expand_to_126(state))

        neural_distribution = output[:7]
        value = output[7]

        root_node = tree_search.explored_nodes[0]
        print(root_node.state)
        root_value = root_node.value_sum / root_node.visit_count

        agent_move = torch.argmax(distribution)

        # Display Network Heads
        alphazero_display(policy_name=policy_name,
                          state=environment.state,
                          tree_dist=distribution,
                          tree_value=root_value,
                          neural_dist=neural_distribution,
                          neural_value=value,
                          agent_move=agent_move,
                          noise=noise)

        environment.action(column=int(agent_move.item()))
        graphic(environment.state)
        print("\n")

        if winner(environment.state) == 1:
            player = compute_player(environment.state)
            if player == 1:
                print("🟡🟡🟡🟡 Yellow Wins! 🟡🟡🟡🟡\n")
                graphic(environment.state)
                print("")
                column = "end"
            else:
                print("🔴🔴🔴🔴 Red Wins! 🔴🔴🔴🔴\n")
                graphic(environment.state)
                print("")
                column = "end"


