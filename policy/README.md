# Policy Module
## Functional description: Decision making(sub-goal selection) based on topo-map
* state: current topo-map
* action: selected frontier node or intention node based on topo-map and policy
* next_state: Topo-map after reaching the selected sub-goal
* reward: Reach rewards and step penalties
* action_space: frontier nodes & intention nodes

## Modules: Some basic neural network sub-blocks
* cnn.py
* deepsets.py
* fc.py
* gnn.py
* graph_pointer.py
* lstm.py
* set_transformer.py

## rl_algorithms
* rl_graph.py
    * class RL_Graph(object):
        * self.data: Dict, include arrive_flag, state, reward
        * self.now_node_index: The index of the current node in rl_graph
        * self.all_nodes: all nodes in rl_graph
        * self.all_action_nodes: frontier nodes and intention nodes in rl_graph
        * def reset(self):
            * Reset the rl_graph when update the rl_graph
        * def is_object_see(self, temp_intention_node):
            * Determine whether the intention node is visible when forming rl_graph
        * def select_intention(self, topo_graph):
            * Select the visible and scored intention nodes within the top-k range from the candidate intension nodes
        * def update(self, topo_graph):
            * Build rl_graph for RL decision-making based on topo-graph, including select visible and top-k intention nodes, prevent the number of action nodes from exceeding args.graph_num_action_padding, action_index and action_mask establish, edges establish.

* sac_graph.py
    * class SAC(RL_Policy):
        * self.actor: Input the state and predict the probability distribution of the action space
        * def select_action(self, state, if_train=False):
            * Select action from all frontier nodes and intention nodes
        * def train(self, writer, train_index, batch_size=16):
            * Train the RL policy.
