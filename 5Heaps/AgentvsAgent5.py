import numpy as np
import random as rnd

from SA5 import SA
import pickle
from AgentQ import Agent
from AgentSarsa import AgentSARSA
import matplotlib.pyplot as plt

###### Agent vs Agent Training ######
""" Both Agents have no prior knowledge"""
def twoPlayer1(board, end, Agent1,  Agent2, c):
    Agent1.move(board)
    if board == end:
        Agent2.loseUpdate(-1)
        Agent1.winUpdate(Agent1.state,Agent1.action,1)
        return False
    
    if c != 0: # Update Agent2
        s = Agent2.state
        a = Agent2.action
        sp = Agent2.readBoard(board)
        Agent2.update(s,a,sp,0)
    
    Agent2.move(board)
    if board == end:
        Agent1.loseUpdate(-1)
        Agent2.winUpdate(Agent2.state,Agent2.action,1)
        return False
    
    s = Agent1.state # Update Agent1
    a = Agent1.action
    sp = Agent1.readBoard(board)
    Agent1.update(s,a,sp,0)
    
def twoPlayer2(board, end, Agent1,  Agent2, c):
    Agent2.move(board)
    if board == end:
        Agent1.loseUpdate(-1)
        Agent2.winUpdate(Agent2.state,Agent2.action,1)
        return False
    
    if c != 0: # Update Agent1
        s = Agent1.state
        a = Agent1.action
        sp = Agent1.readBoard(board)
        Agent1.update(s,a,sp,0)
    
    Agent1.move(board)
    if board == end:
        Agent2.loseUpdate(-1)
        Agent1.winUpdate(Agent1.state,Agent1.action,1)
        return False
    
    s = Agent2.state # Update Agent2
    a = Agent2.action
    sp = Agent2.readBoard(board)
    Agent2.update(s,a,sp,0)


########## agent vs agent

def policyTwoPlayer1(board, end, Agent1,  Agent2):
    """ Agent 1 vs Agent 2"""
    before  = board[0]^board[1]^board[2]^board[3]^board[4]
    if before != 0:
        Agent1.moves += 1
    Agent1.policyMove(board)
    after = board[0]^board[1]^board[2]^board[3]^board[4]
    if after == 0:
        Agent1.t += 1
    if board == end:
        Agent1.won += 1
        return False

    before  = board[0]^board[1]^board[2]^board[3]^board[4]
    if before != 0:
        Agent2.moves += 1
    Agent2.policyMove(board)
    after = board[0]^board[1]^board[2]^board[3]^board[4]
    if after == 0:
        Agent2.t += 1
    if board == end:
        Agent2.won += 1
        return False

def policyTwoPlayer2(board, end, Agent1,  Agent2):
    """ Agent 2 vs Agent 1"""
    
    before  = board[0]^board[1]^board[2]^board[3]^board[4]
    if before != 0:
        Agent2.moves += 1
    Agent2.policyMove(board)
    after = board[0]^board[1]^board[2]^board[3]^board[4]
    if after == 0:
        Agent2.t += 1
    if board == end:
        Agent2.won += 1
        return False
    
    before  = board[0]^board[1]^board[2]^board[3]^board[4]
    if before != 0:
        Agent1.moves += 1
    Agent1.policyMove(board)
    after = board[0]^board[1]^board[2]^board[3]^board[4]
    if after == 0:
        Agent1.t += 1
    if board == end:
        Agent1.won += 1
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
    if after  == 0:
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

stac1 = SA(board)
alpha1 = 0.1
gamma1 = 1
epsilon1 = 0.8
a1 = AgentSARSA(stac1, alpha1, gamma1, epsilon1)

stac2 = SA(board)
alpha2 = 0.1
gamma2 = 1
epsilon2 = 0.8
a2 = Agent(stac2, alpha2, gamma2, epsilon2)


n = 25000 # number of training games

episode= []

wins1 = []
wins2 = []
optmoves1 = []
optmoves2 = []

rnd.seed(0)
for j in range(n):
    interval = 25
    if j % interval == 0:
        ###### Play SARSA against SMART ######

        epslimit = 10000  # Increase Epsilon over time
        a1.epsilon += interval*(1-epsilon1)/epslimit
        
        x = 250 #

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
                while True:
                    if policyPlay2(board, end, a1) == False:
                        break
                board = [1,3,5,7,9]
        episode.append(j)
        wins1.append(a1.won/(started))
        optmoves1.append(a1.t/a1.moves)

        
                ###### Play Q against SMART ######                

        epslimit = 10000  # increase epsilon for agent 2
        a2.epsilon += interval*(1-epsilon2)/epslimit
        x = 250 #

        a2.ngames = 0
        a2.won = 0
        a2.moves = 0
        a2.t = 0
        started = 0
        for i in range(0,x):
            r = rnd.randrange(2)
            if r == 0:
                started += 1
                while True: # Agent first
                    if policyPlay1(board, end, a2) == False:
                        break
                board = [1,3,5,7,9]
            if r == 1:
                while True:
                    if policyPlay2(board, end, a2) == False:
                        break
                board = [1,3,5,7,9]

        wins2.append(a2.won/(started))
        optmoves2.append(a2.t/a2.moves)

####### Q vs SARSA TRINING ######

    r = rnd.randrange(2)
    if r == 0:
        c = 0
        while True: # a1 goes first
            if twoPlayer1(board, end, a1,  a2, c) == False:
                break
            c+=1
        board = [1,3,5,7,9]
        
    if r == 1:
        c = 0
        while True: # comp goes first
            if twoPlayer2(board, end, a1,  a2, c) == False:
                break
            c += 1
        board = [1,3,5,7,9]
        

plt.plot(episode,optmoves1,'r', label = 'S: ' +r'$\alpha=$'+str(alpha1)+r', $\gamma=$'+str(gamma1)+r', $\epsilon_i=$'+str(epsilon1))
plt.plot(episode,optmoves2,'b', label = 'Q: ' +r'$\alpha=$'+str(alpha2)+r', $\gamma=$'+str(gamma2)+r', $\epsilon_i=$'+str(epsilon2))  
plt.ylabel("Ratio of winning moves", fontsize = 20)
plt.xlabel('Number of training games',fontsize=20)
plt.xticks(fontsize = 18)
plt.yticks(fontsize = 18)
plt.ylim(0,1.1)
plt.xlim(0,20000)
plt.legend(loc = 4, fontsize = 18)
plt.tight_layout()
plt.axhline(1,color = 'grey',linestyle = '--')
plt.ylim(0,1.1)
plt.show()

