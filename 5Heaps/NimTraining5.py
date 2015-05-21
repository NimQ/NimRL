import numpy as np
import random as rnd

from SA5 import SA
from AgentQ import Agent

from AgentSarsa import AgentSARSA
import matplotlib.pyplot as plt

import pickle



###### Training Agent vs Computer #######

def play1(board,end,Agent):
    """Agent vs Smart"""
    Agent.move(board)
    if board == end:
##        print(s,a)
        Agent.winUpdate(Agent.state,Agent.action,1)
        return False
    
    smartMove(board)
    if board == end:
        Agent.loseUpdate(-1)
        return False
    s = Agent.state
    a = Agent.action
    sp = Agent.readBoard(board)
    Agent.update(s,a,sp,0)

def play2(board, end , Agent,c):
    """ Smart vs Agent
Computer first """
    
    smartMove(board)
    if board == end:
        Agent.loseUpdate(-1)
        return False
    if c != 0:
        s = Agent.state
        a = Agent.action
        sp = Agent.readBoard(board)
        Agent.update(s,a,sp,0)
        
    Agent.move(board)
    if board == end:
        Agent.winUpdate(Agent.state,Agent.action,1)
        return False



######### Policy #############

def policyPlay1(board, end, Agent):
    """ Agent vs Smart"""
    before  = board[0]^board[1]^board[2]^board[3]^board[4]
    if before != 0:
        
        Agent.moves += 1
        
    Agent.policyMove(board)
   
    after = board[0]^board[1]^board[2]^board[3]^board[4]
  
    if after == 0:
        Agent.t += 1
        
    if board == end:
        Agent.won += 1
        return False
    
    smartMove(board)
    if board == end:
        return False

    
def policyPlay2(board, end, Agent):
    """ Smart vs Agent """

    smartMove(board)
    if board == end:
        return False
    
    before  = board[0]^board[1]^board[2]^board[3]^board[4]
    if before != 0:
        
        Agent.moves += 1
    Agent.policyMove(board)
   
    after = board[0]^board[1]^board[2]^board[3]^board[4]

    if after == 0:
        Agent.t += 1

    if board == end:
        Agent.won += 1
        return False


###### Computer Opponents ##########

def randomMove(board):
    r1 = rnd.randint(0,len(board)-1) # get heap
    
    if board[r1] == 0:
        return randomMove(board)
    elif board[r1] == 1:
        board[r1] -= 1
        return board
    else:
        r2 = rnd.randint(1,board[r1]) # get amount
        board[r1] -= r2
        return board
    
def smartMove(board):
    tryHeap = 0
    bestMove = False
    b = list(board)

    while tryHeap < len(b) and bestMove == False:
        tryValue = 1
        while tryValue <= b[tryHeap] and bestMove == False:
            b[tryHeap] -= tryValue
            
            if b[0]^b[1]^b[2]^b[3]^b[4] == 0:
                bestMove = True
            else:
                b[tryHeap] += tryValue

            tryValue += 1
        tryHeap += 1
    if bestMove == True:
        board[tryHeap-1] -= tryValue -1
        return board
    else:
        return randomMove(board)

##### Agents ##########


board = [1,3,5,7,9]
end = [0,0,0,0,0]

stac = SA(board)    # initialise states and actions



a = [0.1,0.5,0.99] # learning rate parameter
eps = [0.2,0.8]    # epsilon
gam = [0.1,0.5,1]  # discount factor
for y in range(len(a)):
    alpha = a[y]
    for z in range(len(eps)):
        epsilon = eps[z]
        for w in range(len(gam)):
            gamma = gam[w]

            
            a1 = Agent(stac, alpha, gamma, epsilon) # initialise agent
            n = 20000
            rnd.seed(0)
            ########## Train A1 ########
            episode = []
            wins = []
            optmoves = []
            """Against smart """
            for j in range(0,n):
                interval = 25
                
                if j % interval  == 0: # Increase Epsilon over time
                    epslimit = 10000
                    a1.epsilon += interval*(1-epsilon)/epslimit
                    
                    x = 250 # Performance : play 100 games each 1000 episodes
                    a1.ngames = 0
                    a1.won = 0
                    a1.moves = 0
                    a1.t = 0
                    started = 0
                    for i in range(0,x):
                        r = rnd.randrange(2)
                        if r == 0:
                            started += 1
                            while True: # Agent first
                                if policyPlay1(board, end, a1) == False:
                                    break
                            board = [1,3,5,7,9]
                        if r == 1:
                            while True: # Computer first
                                if policyPlay2(board, end, a1) == False:
                                    break
                            board = [1,3,5,7,9]
                    
                    episode.append(j)
                    wins.append(a1.won/(started))
                    optmoves.append(a1.t/a1.moves)


                r = rnd.randrange(2)
                if r == 0: 
                    while True: # a1 goes first
                        if play1(board,end,a1) == False:
                            break
                    board = [1,3,5,7,9]
                    
                if r == 1:
                    c = 0
                    while True: # comp goes first
                        if play2(board,end,a1,c) == False:
                            break
                        c += 1
                    board = [1,3,5,7,9]
                    
            with open('AgentvsSmartEpisodeAlpha_'+str(a1.alpha)+' Gamma_'+str(a1.gamma)+' EpsInc_'+str(epsilon)+'.txt', 'wb') as f:
                pickle.dump(episode, f)

            with open('AgentvsSmartMovesAlpha_'+str(a1.alpha)+' Gamma_'+str(a1.gamma)+' EpsInc_'+str(epsilon)+'.txt', 'wb') as f:
                pickle.dump(optmoves, f)

            with open('AgentvsSmartWinsAlpha_'+str(a1.alpha)+' Gamma_'+str(a1.gamma)+' EpsInc_'+str(epsilon)+'.txt', 'wb') as f:
                pickle.dump(wins, f)
            




        

