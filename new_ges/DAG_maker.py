from dataclasses import dataclass
from collections import defaultdict
import numpy as np
from utils import is_dag

@dataclass
class ActionEdge:
    i: int
    j: int
    action: int 
    score: int
    used: bool = False

    def get_hash(self):
        return (self.i, self.j, self.action)
    
    def get_coord(self):
        return (self.i, self.j)

    def get(self):
        return (self.i, self.j, self.action, self.score)

class combine_CPDAG:

    def __init__(self, size):
        self.size = size
        self.actionEdges = {}

    def add_CPDAG(self, CPDAG, action_matrix, score, include_undirected = True):
        for i in range(self.size):
            for j in range(self.size):
                if CPDAG[i, j] and (include_undirected or not CPDAG[j, i]):
                    edge = ActionEdge(i, j, action_matrix[i, j], score)
                    if edge.get_hash() not in self.actionEdges:
                        self.actionEdges[edge.get_hash()] = edge
                    else:
                        self.actionEdges[edge.get_hash()].score += edge.score

    def combine(self):
        mult_graph = np.zeros((self.size, self.size))
        mult_action = np.zeros_like(mult_graph)
        mult_score = np.zeros_like(mult_graph)

        for edge in self.actionEdges.values():
            i, j, action, score = edge.get()

            if score > mult_score[i, j]:
                mult_graph[i, j] = 1
                mult_action[i, j] = action
                mult_score[i, j] = score

            elif score == mult_score[i, j]:
                print(f"error: claire did something wrong, action {action} has equal score to {mult_action[i, j]}")
        
        print("----------------- initial combined CPDAG -------------------")
        print(mult_graph)
        print(mult_action)
        print(mult_score)
        print(f"is dag : {is_dag(mult_graph)}")

        print('----------------- now thresholding       -------------------')
        m = np.max(mult_score)
        threshold = 0.01
        mult_score /= m
        mult_score[mult_score < threshold] = 0
        mult_graph[mult_score < threshold] = 0
        print(mult_graph)
        print(mult_score)
        print(f"is dag : {is_dag(mult_graph)}")
        print(" ---------------- now picking better edges ------------------")
        for i in range(self.size):
            for j in range(self.size):
                if mult_score[i, j] > mult_score[j, i]:
                    mult_graph[j, i] = 0
                else:
                    mult_graph[i, j] = 0
        
        mult_score = mult_graph * mult_score
        mult_action = mult_graph * mult_action
        print(mult_graph)
        print(mult_score)
        print(f"is dag : {is_dag(mult_graph)}")
        return mult_graph, mult_action, mult_score
            

