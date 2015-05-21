import random as rnd
import numpy as np

class SA():
    def __init__(self, board):
        d = []
        for i in range(0,board[0]+1):
            for j in range(0,board[1]+1):
                for k in range(0,board[2]+1):
                    for l in range(0, board[3] + 1):
                        d.append([i,j,k,l])

        states = {i:d[i] for i in range(len(d))}
        stringstates = [''.join(str(states[j])) for j in states]

        stateindex = {stringstates[i]:i for i in range(len(stringstates))}

        self.states = states
        self.stateindex = stateindex

        actions = []
        for i in range(board[0]):
            actions.append([0,i+1])
        for i in range(board[1]):
            actions.append([1,i+1])
        for i in range(board[2]):
            actions.append([2,i+1])
        for i in range(board[3]):
            actions.append([3,i+1])

        self.actions = actions
