import numpy as np
from HaifaEnv import HaifaEnv
from typing import List, Tuple
import heapdict
import collections

class Node:
    def __init__(self, state: int, parent: 'Node' = None, action: int = None, path_cost: float = 0.0):
        self.state = state
        self.parent = parent
        self.action = action
        self.path_cost = path_cost
    def __hash__(self):
        return hash(self.state)  

    def __eq__(self, other):
        if isinstance(other, Node):
            return self.state == other.state
        return self.state == other
    
    def __repr__(self):
        return f"Node(state={self.state}, cost={self.path_cost})"

class BFSGAgent:
    def __init__(self) -> None:
        pass

    def _reconstruct_path(self, node: Node) -> List[int]:
        actions = []
        current = node
        while current.parent is not None:
            actions.append(current.action)
            current = current.parent
        return actions[::-1] 

    def search(self, env: HaifaEnv) -> Tuple[List[int], float, int]:
        initial_state = env.get_initial_state()
        start_node = Node(state=initial_state, parent=None, action=None, path_cost=0.0)
        open_queue = collections.deque([start_node])
        open_states = {start_node.state}
        closed_set = set()
        expanded_nodes_count = 1

        while open_queue:
            current_node = open_queue.popleft()
            current_state = current_node.state
            closed_set.add(current_state)
            expanded_nodes_count += 1
            if env.is_final_state(current_state):
                return self._reconstruct_path(current_node), current_node.path_cost, expanded_nodes_count

            for action_taken, (next_state, step_cost, terminated) in env.succ(current_state).items():
                if next_state is None or step_cost == float('inf'):
                    continue

                child_path_cost = current_node.path_cost + step_cost
                child_node = Node(
                    state=next_state,
                    parent=current_node,
                    action=action_taken,
                    path_cost=child_path_cost
                )

                if child_node.state not in closed_set and child_node.state not in open_states:
                    open_queue.append(child_node)
                    open_states.add(child_node.state)

        return [], 0.0, 0



class GreedyAgent():
    def __init__(self) -> None:
        pass
    
    def _reconstruct_path(self, node: Node) -> List[int]:
        actions = []
        current = node
        while current.parent is not None:
            actions.append(current.action)
            current = current.parent
        return actions[::-1] 
    @staticmethod
    def make_min_manhattan(targets):
        targets_np = np.array(targets)
        def min_manhattan(a):
            return int(np.min(np.sum(np.abs(targets_np - np.array(a)), axis=1)))
        return min_manhattan
    
    def init_h(self, env: HaifaEnv):
        goal_row_cols = [env.to_row_col(s) for s in env.get_goal_states()]
        self.f = GreedyAgent.make_min_manhattan(goal_row_cols)
        self.env = env
    
    def haifa_h(self, state: int) -> float:
        row_col = self.env.to_row_col(state)
        return min(self.f(row_col), 100)
    
    def search(self, env: HaifaEnv) -> Tuple[List[int], float, int]:
        initial_state = env.get_initial_state()
        self.init_h(env)
        start_node = Node(state=initial_state, parent=None, action=None, path_cost=0.0 )
        open_queue = heapdict.heapdict()
        open_queue[start_node] = (self.haifa_h(initial_state), initial_state)
        closed_set = set()
        expanded_nodes_count = 1
        while open_queue:
            current_node, (h_value, _)= open_queue.popitem()
            if current_node in closed_set:
                continue
            closed_set.add(current_node)
            expanded_nodes_count += 1
            if env.is_final_state(current_node):
                return self._reconstruct_path(current_node), current_node.path_cost, expanded_nodes_count
            for action_taken, (next_state, step_cost, terminated) in env.succ(current_node.state).items():
                if next_state is None or step_cost == float("inf"):
                    continue
                
                if next_state in closed_set:
                    continue
                
                child_path_cost = current_node.path_cost + step_cost
                child_node = Node(
                    state=next_state,
                    parent=current_node,
                    action=action_taken,
                    path_cost=child_path_cost,
                )
                
                if child_node.state not in closed_set and child_node not in open_queue:
                    open_queue[child_node] = (self.haifa_h(next_state), next_state)
        return [],0,0
                


class AStarEpsilonAgent():
    def __init__(self):
        pass
    
    def _reconstruct_path(self, node: Node) -> List[int]:
        actions = []
        current = node
        while current.parent is not None:
            actions.append(current.action)
            current = current.parent
        return actions[::-1] 
    
    @staticmethod
    def make_min_manhattan(targets):
        targets_np = np.array(targets)
        def min_manhattan(a):
            return int(np.min(np.sum(np.abs(targets_np - np.array(a)), axis=1)))
        return min_manhattan
    
    def init_h(self, env: HaifaEnv):
        goal_row_cols = [env.to_row_col(s) for s in env.get_goal_states()]
        self.f = GreedyAgent.make_min_manhattan(goal_row_cols)
        self.env = env
    
    def haifa_h(self, state: int) -> float:
        row_col = self.env.to_row_col(state)
        return min(self.f(row_col), 100)
    
    def h_focal(self, focal: list) -> int:
        return min(focal, key=lambda s: (self.haifa_h(s), s))

    def search(self, env: HaifaEnv, epsilon: float = None): 
        initial_state = env.get_initial_state()
        self.init_h(env)
        g = {initial_state: 0.0}
        start_node = Node(state=initial_state, parent=None, action=None, path_cost=0.0 )
        open_queue = heapdict.heapdict()
        open_queue[initial_state] = g[initial_state] + self.haifa_h(initial_state)
        open_nodes = {initial_state: start_node}
        closed_set = set()
        expanded_nodes_count = 1
        while open_queue:
            _, min_f = open_queue.peekitem()
            threshold = (1 + epsilon) * min_f
            focal = [s for s, f_val in open_queue.items() if f_val <= threshold]         
            best_state = self.h_focal(focal)
            current_node = open_nodes[best_state]
            del open_queue[best_state]
            del open_nodes[best_state]
            closed_set.add(best_state)
            expanded_nodes_count += 1
            if env.is_final_state(best_state):
                return (self._reconstruct_path(current_node),current_node.path_cost,expanded_nodes_count)
            for action_taken, (next_state, step_cost, terminated) in env.succ(current_node.state).items():
                if next_state is None or step_cost == float("inf"):
                    continue
                
                if next_state in closed_set:
                    continue
                
                new_g = g[best_state] + step_cost

                if next_state not in g or new_g < g[next_state]:
                    g[next_state] = new_g
                    child_node= Node(
                        state=next_state,
                        parent=current_node,
                        action=action_taken,
                        path_cost=new_g,
                    )
                    open_nodes[next_state] = child_node
                    open_queue[next_state] = new_g + self.haifa_h(next_state)
        return [],0,0



class AStarAgent():
    
    def __init__(self):
        pass

    def search(self, env: HaifaEnv) -> Tuple[List[int], float, int]:
        return AStarEpsilonAgent().search(env, epsilon=0.0) 

