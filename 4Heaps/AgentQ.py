import random as rnd
import numpy as np

class Agent():
    def __init__(self, SA, alpha, gamma, epsilon):
        self.ngames = 0
        self.won = 0

        self.state = 0
        self.action = 0
        
        self.actions = SA.actions
        self.states = SA.states
        self.stateindex = SA.stateindex

        self.Q = np.zeros([len(SA.states),len(SA.actions)])
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

        self.t = 0 # number of optimal moves made
        self.f = 0
        self.moves = 0 # optimal move was possible

    def readBoard(self, board):
        """ read board and return the state s"""
        b = ''.join(str(board))
        s = self.stateindex[b]
        return s

    def chooseAction(self, s):
        """ chooses action according to action policy """
        r = rnd.random()

        if r > self.epsilon: # for epsilon-greedy policy
            a = rnd.randrange(len(self.actions))
        else:       # choose best possible action in this state
            q = list(self.Q[s,:])
            m = max(q)
            if q.count(m) > 1: # if more than 1 action w/ max value
                bestAction = []
                for i in range(len(q)):
                    if q[i] == m:
                        bestAction.append(i)
                a = rnd.choice(bestAction)

            else:
                a = np.argmax(self.Q[s,:])
        if self.isValid(s, a) == False:
            return self.chooseAction(s)
        else:
            return a

    def changeBoard(self, board, a):
        action = self.actions[a]
        heap = action[0]
        amount = action[1]
        board[heap] -= amount
        return board

    def move(self, board):
        
        s = self.readBoard(board)
        a = self.chooseAction(s)

        self.state = s
        self.action = a

        board = self.changeBoard(board, a) 
        sp = self.readBoard(board)  # get s' (new state)
        
        return board

    def optimalAction(self, sp):
        q = list(self.Q[sp,:])
        m = max(q)
        if q.count(m) > 1: # if more than 1 action w/ max value
            bestAction = []
            for i in range(len(q)):
                if q[i] == m:
                    bestAction.append(i)

            ap = rnd.choice(bestAction)
        else:
            ap = np.argmax(self.Q[sp,:])

        if self.isValid(sp, ap) == False:
            return self.optimalAction(sp)
        else:
            return ap
        

    def winUpdate(self, s, a, R):
        self.won += 1
        self.Q[s][a] += self.alpha*(R - self.Q[s][a])

    def update(self, s, a, sp, R):
        ap = self.optimalAction(sp)
        self.Q[s][a] += self.alpha*(R + self.gamma*self.Q[sp][ap] - self.Q[s][a])

    def loseUpdate(self, R):
        s = self.state
        a = self.action
        self.Q[s][a] += self.alpha*(R - self.Q[s][a])
    
    def isValid(self, s, a):
        action = self.actions[a]
        heap = action[0]
        amount = action[1]
        if (self.states[s][heap] - amount) < 0:
            self.Q[s][a] = -10
            return False
        else:
            return True

############### Optimal Policy  ##############

    def policyMove(self, board):
                
        s = self.readBoard(board)
        a = self.policyAction(s)
        board = self.changeBoard(board, a) 
        return board
        



    def policyAction(self, s):
        # choose best possible action in this state
        q = list(self.Q[s,:])
        m = max(q)
        if q.count(m) > 1: # if more than 1 action w/ max value
            bestAction = []
            for i in range(len(q)):
                if q[i] == m:
                    bestAction.append(i)
            a = rnd.choice(bestAction)

        else:
            a = np.argmax(self.Q[s,:])
        if self.isValid(s, a) == False:
            return self.policyAction(s)
        else:
            return a
        return

    
        
        


        
