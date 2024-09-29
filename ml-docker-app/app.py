from __future__ import absolute_import
#docker references:
# https://dev.to/pavanbelagatti/a-step-by-step-guide-to-containerizing-and-deploying-machine-learning-models-with-docker-21al
# https://www.docker.com/blog/docker-best-practices-understanding-the-differences-between-add-and-copy-instructions-in-dockerfiles/

import random
import math
import numpy as np
from random import randrange
import copy
import gymnasium as gym
import time
import argparse
import os
import json
import logging

handle="app.py"
logger = logging.getLogger(handle)
logging.basicConfig(level=logging.INFO)

#import matplotlib.pyplot as plt
#from scipy.integrate import solve_ivp

def load_json_object(json_file_path):
    with open(json_file_path) as json_file:
        return json.load(json_file)

def sign(num):
    return -1 if num < 0 else 1
    
def grow(i, seed, tree):
    random.seed
#all grow will apply any function or leaf
#iteration variable i tracks depth of tree to ensure minimum or maximum depths
#seed is the current tree to be grown
#tree is an object with hyperparmeters min_depth, max_depth, number of sensors, constant range, sensor values array
   
#choose a random number to how tree will grow, operator or leaf (sensor or constant)
#if below min depth cannot be leaf, if at max depth must be leaf
    if i<tree.min_depth:
        key=random.randint(tree.num_sens+1, tree.num_sens+9-1)
    elif i>=tree.max_depth:
        key=random.randint(0, tree.num_sens+1-1)
    else: key=random.randint(0,tree.num_sens+9-1)
    #remove trig functions and log, comment to include
    #while key in range(tree.num_sens+2,tree.num_sens+6):  key=random.randint(0,tree.num_sens+9-1)
    #print(i, key, tree.name)
#if leaf, easy, just return the sensor value or a random constant [-10,10]
    if key in range (0,tree.num_sens):
        i-=1
        growth=tree.sens[key]
        s='s'+str(key)
        tree.name.append(s)
        #tree.name=tree.name+" s"+str(key)
        return growth
    if key==tree.num_sens:
        i-=1
        a=round(random.uniform(-10,10),3)
        tree.name.append(a)
        #tree.name=tree.name+" "+str(a)
        return a
#if function a bit harder, use function then call grow recursively
    if key==tree.num_sens+1:
        i+=1
        #tree.name+=" math.tanh"
        tree.name.append('math.tanh')
        growth=math.tanh(grow(i,seed,tree))
        return growth
    if key==tree.num_sens+2:
        i+=1
        tree.name.append('math.sin')
        #tree.name+=" math.sin"
        growth=math.sin(grow(i,seed,tree))
        return growth
    if key==tree.num_sens+3:
        i+=1
        tree.name.append('math.cos')
        #tree.name+=" math.cos"
        growth=math.cos(grow(i,seed,tree))
        return growth
    if key==tree.num_sens+4:
        i+=1
        #tree.name+=" math.log"
        tree.name.append('math.log')
        b=grow(i,seed,tree)
        #print(b)
        if abs(b)>=10**-3:
            growth=b/abs(b)*math.log(abs(b))
        if abs(b)<10**-3:
            growth=0
        return growth
    if key==tree.num_sens+5:
        i+=1
        tree.name.append('math.exp')
        #tree.name+=" math.exp"
        a=grow(i,seed,tree)
        #prevent float overflow; I seriously doubt e^10 will ever be the right answer
        if a>10:
            a=10
        growth=math.exp(a)
        return growth
#now the hard part, using the +,-,x,/ functions which split branch
#need to track total tree depth on each branch and add new branch
    if key==tree.num_sens+6:
        i+=1
        #tree.name+=" +"
        tree.name.append('+')
        growth=grow(i,seed, tree)+grow(i,seed,tree)
        return growth
    if key==tree.num_sens+7:
        i+=1
        tree.name.append('-')
        #tree.name+=" -"
        a=grow(i,seed,tree)
        b=grow(i,seed,tree)
        #avoid disappearing branches throwing off recursion and junk equation values
        if a==b:  b=b+0.01
        growth=a-b
        #growth=grow(i,seed,tree)-growth(i,seed,tree)
        return growth
    if key==tree.num_sens+8:
        i+=1
       # tree.name+=" *"
        tree.name.append('*')
        a=grow(i,seed, tree)
        b=grow(i,seed, tree)
        growth=a*b
        return growth
    if key==tree.num_sens+9:
        i+=1
        tree.name.append('/')
        a=grow(i,seed, tree)
        #tree.name+=" /"
        check1=len(tree.name)
        b=grow(i,seed, tree)
        check2=len(tree.name)
        #if check2 == check1:  print(tree.name)
        if abs(b)>10**-3:
            growth=a/b
        elif b==0:
            growth=a/10**-3
        elif abs(b)<=10**-3:
            growth=a/(np.sign(b)*10**-3)
        return growth
#additional functions
    if key==tree.num_sens+10:
        i+=1
        tree.name.append('abs')
        growth=abs(grow(i,seed,tree))
        return growth
        
def growTrig(i, seed, tree):
    random.seed
#all grow will apply any function or leaf
#iteration variable i tracks depth of tree to ensure minimum or maximum depths
#seed is the current tree to be grown
#tree is an object with hyperparmeters min_depth, max_depth, number of sensors, constant range, sensor values array
   
#choose a random number to how tree will grow, operator or leaf (sensor or constant)
#if below min depth cannot be leaf, if at max depth must be leaf
    if i<tree.min_depth:
        key=random.randint(tree.num_sens+1, tree.num_sens+6-1)
    elif i>=tree.max_depth:
        key=random.randint(0, tree.num_sens+1-1)
    else: key=random.randint(0,tree.num_sens+6-1)
    #remove trig functions and log, comment to include
    #while key in range(tree.num_sens+2,tree.num_sens+6):  key=random.randint(0,tree.num_sens+9-1)
    #print(i, key, tree.name)
#if leaf, easy, just return the sensor value or a random constant [-10,10]
    if key in range (0,tree.num_sens):
        i-=1
        growth=tree.sens[key]
        s='s'+str(key)
        tree.name.append(s)
        #tree.name=tree.name+" s"+str(key)
        return growth
    if key==tree.num_sens:
        i-=1
        a=round(random.uniform(-10,10),3)
        tree.name.append(a)
        #tree.name=tree.name+" "+str(a)
        return a
#if function a bit harder, use function then call grow recursively
    if key==tree.num_sens+1:
        i+=1
        #tree.name+=" math.tanh"
        tree.name.append('math.sin')
        growth=math.tanh(growTrig(i,seed,tree))
        return growth
    if key==tree.num_sens+2:
        i+=1
        tree.name.append('math.cos')
        #tree.name+=" math.sin"
        growth=math.sin(growTrig(i,seed,tree))
        return growth
#now the hard part, using the +,-,x,/ functions which split branch
#need to track total tree depth on each branch and add new branch
    if key==tree.num_sens+3:
        i+=1
        #tree.name+=" +"
        tree.name.append('+')
        growth=growTrig(i,seed, tree)+growTrig(i,seed,tree)
        return growth
    if key==tree.num_sens+4:
        i+=1
        tree.name.append('-')
        #tree.name+=" -"
        a=growTrig(i,seed,tree)
        b=growTrig(i,seed,tree)
        #avoid disappearing branches throwing off recursion and junk equation values
        if a==b:  b=b+0.01
        growth=a-b
        #growth=grow(i,seed,tree)-growth(i,seed,tree)
        return growth
    if key==tree.num_sens+5:
        i+=1
       # tree.name+=" *"
        tree.name.append('*')
        a=growTrig(i,seed, tree)
        b=growTrig(i,seed, tree)
        growth=a*b
        return growth
    if key==tree.num_sens+6:
        i+=1
        tree.name.append('/')
        a=growTrig(i,seed, tree)
        #tree.name+=" /"
        check1=len(tree.name)
        b=growTrig(i,seed, tree)
        check2=len(tree.name)
        #if check2 == check1:  print(tree.name)
        if abs(b)>10**-3:
            growth=a/b
        elif b==0:
            growth=a/10**-3
        elif abs(b)<=10**-3:
            growth=a/(np.sign(b)*10**-3)
        return growth
        
def evaluate(token ):

    #evaluate takes a set of sensor mesaruments (array of size tree.num_sens).  Sensors values are seperate from tree.sens
#this is basically opposite of tree building
#evaluate must be called from total evaluate to replace the sensor with appropriate values

    token=copy.copy(token) #this took awhile to figure out
    t=token.pop(0)

    if t=='+':
#         print(t, '+')
        a,token=evaluate(token)
        b,token=evaluate(token)
        #print(a,b)
        return a+b,token
    elif t=='-':
#         print(t,'-')
        a,token=evaluate(token)
        b,token=evaluate(token)
        if a==b:  b=b+0.01
        #print(a,b)
        return a-b,token
    elif t=='*':
#         print(t,'*')
        a,token=evaluate(token)
        b,token=evaluate(token)
        #print(a,b)
        return a*b, token
    elif t=='/':
#         print(t,'/')
        a,token=evaluate(token)
        b,token=evaluate(token) #protect against divide by 0
        if b==0:
            return a/10**-3, token
        elif abs(b)<10**-3:
            return a/(np.sign(b)*10**-3),token
        else:
            return a/b,token
    elif (isinstance(t,np.float64) or isinstance(t, float)):
        #print(t, 'float')
        return t, token
    #if it isn't one of the above, then it is a transcendental function and will be func(lower branch)
    else:
#         print(t)
        temp,token=evaluate(token)
#         print(temp,token)
        if t=='math.exp':
            if abs(temp)>50:
                temp=sign(temp)*50
        if t=='math.log':
            if abs(temp)>10**-3:
                temp=abs(temp)
            else:
                temp=10**-3
        a=t+'('+str(temp)+')'
#         print(t,a)
        return eval(a),token
        
def branch(n, tree):
#branch defines a branch from a tree, starting from point n in the tree, where n is an int 
#for an address in the list that defines the tree
#iterate through the list defining the tree starting at n until enough leafs have been encountered to define the branch
#need one leaf plus one extra leaf per operand (+, -, *, /).  Leaf is sensor or constant
#to make easier, change all sensors to a float locally in the function
#function returns index for the end of the branch
    tree=copy.copy(tree)
    op_list =['+', '-', '*', '/']
    c=1 #leaf counter
    while c>0:
        if n>=len(tree.name):  print('branch fail', n, tree.name)
        if type(tree.name[n])==float:
            c-=1
            #print(tree.name[n], c, 'loss')
        elif tree.name[n].startswith('s'):
            c-=1
            #print(tree.name[n], c, 'loss')
        for x in op_list:
            if tree.name[n] == x:
                c+=1
                #print(tree.name[n], c, 'gain')
        n+=1
    return n
    
def cutgrow(x):
    tree=copy.copy(x)
    random.seed
    cut=randrange(len(tree.name))
    end=branch(cut,tree)
    #print(cut, end)
    if cut>0:
        keepLeft=tree.name[0:cut]
    else:  keepLeft=[]
    if end<len(tree.name):
        keepRight=tree.name[end:len(tree.name)]
    else:  keepRight=[]
    dummy=sapling(tree.min_depth, tree.max_depth-1, tree.num_sens) 
    #need to pass a sapling object to grow a branch, so make a dummy one to grow
    #reduce depth by one to make new sub branch a little cleaner
    total=grow(0,2,dummy)
 
    new_name=keepLeft
    new_name+= dummy.name
    new_name+=keepRight
    #print(new_name)
    return new_name
    
#shrink replaces tree with randomly chosen leaf.  For now, make it a new random leaf (sensor or constant)
def shrink(x):
    tree=copy.copy(x)
    cut=randrange(len(tree.name))
    end=branch(cut, tree)
    #print(cut, end)
    if cut>0:
        keepLeft=tree.name[0:cut]
    else:  keepLeft=[]
    if end<len(tree.name):
        keepRight=tree.name[end:len(tree.name)]
    else:  keepRight=[]
    key=random.randint(0,tree.num_sens)
    new_name=keepLeft
    if key in range (0,tree.num_sens):
        s='s'+str(key)
        new_name.append(s)
    if key==tree.num_sens:
        a=round(random.uniform(-10,10),3)
        new_name.append(a)   
    new_name+=keepRight
    #print(new_name)
    return new_name

#hoist replaces tree by random subtree
def hoist(x):
    tree=copy.copy(x)
    cut=randrange(len(tree.name))
    end=branch(cut, tree)
    new_name=tree.name[cut:end]
    #print(new_name, cut, end)
    return new_name

#reparmaterize replaces each constant with random different constant
#this is the only one that directly changes tree
def reparam(x):
    tree=copy.copy(x)
    for i in range(len(tree.name)):
        if type(tree.name[i])==float:
            tree.name[i]=round(random.uniform(-10,10),3)
    return tree.name

#crossover swaps subtrees between trees
def cross(x1, x2):
    tree1=copy.copy(x1)
    if len(tree1.name)== 0:  print('cross tree1 empty')
    tree2=copy.copy(x2) 
    cut1=randrange(len(tree1.name))
    if len(tree1.name)<=cut1:  print('cross 1 short', len(tree1.name), cut1)
    end1=branch(cut1, tree1)
    if cut1>0:
        keepLeft1=tree1.name[0:cut1]
    else:  keepLeft1=[]
    if end1<len(tree1.name):
        keepRight1=tree1.name[end1:len(tree1.name)]
    else:  keepRight1=[]
    swap1=tree1.name[cut1:end1]

    cut2=randrange(len(tree2.name))
    if len(tree2.name)== 0:  print('cross tree2 empty')
    if len(tree2.name)<=cut2:  print('cross 2 short', len(tree2.name), cut)
    end2=branch(cut2, tree2)
    if cut2>0:
        keepLeft2=tree2.name[0:cut2]
    else:  keepLeft2=[]
    if end2<len(tree2.name):
        keepRight2=tree2.name[end2:len(tree2.name)]
    else:  keepRight2=[]
    swap2=tree2.name[cut2:end2]

    new_name1=keepLeft1
    new_name1+=swap2
    new_name1+=keepRight1
    
    new_name2=keepLeft2
    new_name2+=swap1
    new_name2+=keepRight2
    
    return new_name1, new_name2    
    
def total_evaluate(b,s,tree):
#s is a 2d array of sensor values across all sensors and all time/space
#b is 1d list of target values across time
#pass all sensors one time step at a time to evaluate tree value at each time step
#each sensor is a row, each timestep a column
    J=0.
    tree=copy.copy(tree)
    for i in range(len(b)):
        dummy=copy.copy(tree)
        if tree.num_sens>1:
            for j in range(tree.num_sens):
                sensor="s"+str(j)
                dummy.name=[s[j,i] if t==sensor else t for t in dummy.name]
                if len(dummy.name)==0: print("total nsens empty")
        else:
            dummy.name=[s[i] if t=='s0' else t for t in tree.name]
            if len(dummy.name)==0: print("total 1sens empty")
        step,dummy=evaluate(dummy.name)

        J+=1/len(b)*(b[i]-step)**2
        #print(J, b[i], s[i], step)
    return J

#define the cost function
def cost(scale, target, current):
#scale is the scaling factor for averaging cost at each step, target is desired value, current is function value
    return (target-current)**2/scale
    
def cost_extract(tree):
    return tree.total

def selection(Ni, Np, forest):
    new_forest=[] #next generation
    random.seed
    while len(new_forest)<Ni:
        choices=random.sample(range(0,Ni), Np)
        pool=[] #holds local pool
        pool_value=[] #holds cost score for each tree in pool
        for c in choices:
            pool.append(forest[c])
            pool_value.append(forest[c].total)
        #only take 1 minimum value from each
        winner=np.where(pool_value==np.min(pool_value))[0][0]
        new_forest.append(pool[winner])
    return new_forest
    
def evolution(Ne, Pc, Pm, new_forest):

    new_forest.sort(key=cost_extract) #rank by ascending cost to identify elites
#     for t in new_forest:
#         print('sort ',t.name, t.total)
    copy = new_forest[Ne:]
    random.shuffle(copy)
    new_forest[Ne:]= copy #shuffle rest of list
    breeders=[]
    for t in range(Ne,len(new_forest)):
        if len(new_forest[t].name)==0:  print("evo empty", t, new_forest[t].name, new_forest[t-1].name, new_forest[t+1].name, last_roll)
        random.seed
        roll=random.uniform(0,1)
        #print('roll ',roll)
        if roll<=Pc:
            breeders.append(t)
            if len(breeders)>1: #cross breed each pair as they are selected
                #print(new_forest[breeders[0]].name,new_forest[breeders[1]].name)
                if len(new_forest[breeders[0]].name)==0:  print('breeder 0 empty', breeders[0])
                if len(new_forest[breeders[1]].name)==0:  print('breeder 1 empty', breeders[1])
                new_forest[breeders[0]].name,new_forest[breeders[1]].name = \
                cross(new_forest[breeders[0]],new_forest[breeders[1]])
                #print(new_forest[breeders[0]].name,new_forest[breeders[1]].name)
                breeders=[]
        if (roll>Pc and roll <Pc+Pm):
            if roll<0.75:
                #print(new_forest[t].name)
                new_forest[t].name==cutgrow(new_forest[t]) #take out cutgrow for troubleshooting
                #print(new_forest[t].name, 'cutgrow')
            elif roll<0.80:
                #print(new_forest[t].name)
                new_forest[t].name=shrink(new_forest[t]) #take out shirnk for troubleshooting
                #print(new_forest[t].name, 'shrink')
            elif roll<0.85:
                #print(new_forest[t].name)
                new_forest[t].name=hoist(new_forest[t])
                #print(new_forest[t].name, 'hoist')
            else:
                new_forest[t].name=reparam(new_forest[t])
        last_roll=roll
    return new_forest
    
def evolution_reseed(Nn, Ni, Ne, Pc, Pm, forest):

    forest.sort(key=cost_extract) #rank by ascending cost to identify elites
    #shennanigans to include elites in mutation popoulation and get around Python's way of handling memory and lists
    elites=[]
    for i in range(0,Ne):
        elites.append(sapling(forest[0].min_depth, forest[0].max_depth, forest[0].num_sens))
        elites[i].name=forest[i].name.copy()
    #shuffle after elites to avoid cross breeding similar trees    
    copy = list(forest[Ne:int((1-Nn)*Ni)])
    random.shuffle(copy)
    
    new_forest=list(forest)
    new_forest[Ne:int((1-Nn)*Ni)]= copy.copy() #shuffle top half of list
    new_forest[Ne:Ne+Ne]=list(elites) #put elites into breeding pool

    breeders=[]
    checkname=new_forest[0].name
    for t in range(Ne,int((1-Nn)*Ni)): #
        random.seed
        roll=random.uniform(0,1)
        #print('roll ',roll)
        if roll<=Pc:
            breeders.append(t)
            if len(breeders)>1: #cross breed each pair as they are selected
                #print(new_forest[breeders[0]].name,new_forest[breeders[1]].name)
                new_forest[breeders[0]].name,new_forest[breeders[1]].name = \
                cross(new_forest[breeders[0]],new_forest[breeders[1]])
                #print(new_forest[breeders[0]].name,new_forest[breeders[1]].name)
                breeders=[]
        if (roll>Pc and roll <Pc+Pm):
            if roll<0.68:
                #print(new_forest[t].name)
                new_forest[t].name==cutgrow(new_forest[t]) #take out cutgrow for troubleshooting
                #print(new_forest[t].name, 'cutgrow')
            elif roll<0.75:
                #print(new_forest[t].name)
                new_forest[t].name=shrink(new_forest[t]) #take out shirnk for troubleshooting
                #print(new_forest[t].name, 'shrink')
            elif roll<0.83:
                #print(new_forest[t].name)
                new_forest[t].name=hoist(new_forest[t])
                #print(new_forest[t].name, 'hoist')
            else:
                new_forest[t].name=reparam(new_forest[t])

    for t in range(int((1-Nn)*Ni),Ni):
            new_forest[t].name=[]
            dummy=grow(0,2,new_forest[t])

    return new_forest
    
class sapling:
    def __init__(self,mind,maxd,num):
        self.min_depth=mind
        self.max_depth=maxd
        self.num_sens=num
        self.total=[]
        self.sens=[] #for now this is dummy numbers to make tree builder work properly
        self.name=[]
        for x in range(num):
            self.sens.append(0.5)
  
def train(forest, maxi, Nn, Ni, Ne,Pr, Pc, Pm, time0):
    maxJ=-50000 #set initial "best" cost
    k=0 #iteration counter

    while (maxJ<-3 and k<maxi):
        #make random starting point each forest generation, but same for each tree per generation, different for each batch
        randSeeds=random.sample(range(5000),n_batches)#random.randint(0,10000)
        for t in forest:
            #reset the environment for each tree using same enivornment for each tree with each time step
            t.total=0
            for n in randSeeds:
                observation, info = env.reset(seed=n)
                #iterate through a list of trees and store best
                #to determine cost, need K and reward for every timestep, env.step provides
                #make sure reward is reset in case tree is reused
                s=observation
                subtotal=0
                
                #print(control_name[0])
                #need to take a pass of '500' time steps through environment each tree, each iteration
                for i in range(env.spec.max_episode_steps):
                    control_name=t.name
                    for j in range(len(s)):
                        obs_name='s'+str(j)
                        control_name=[round(s[j].item(),3) if c==obs_name else c for c in control_name]
                    K,dummy=evaluate(control_name)
                    if K>20:  K=20.0
                    elif K<-20:  K=-20.0
                    K=np.asarray([K])
                    start=time.time()
                    obs, reward, terminated, truncated, info = env.step(K)
                    end=time.time()
                    if end-start>.3: #catch long gym times
                        reward=-10000
                        print("gym trip", control_name)
                        break
                    subtotal+=reward#+len(control_name)/4.
                    #print(subtotal)
                    s=obs
                    if (terminated or truncated):  break
                if randSeeds.index(n)>0:  t.total=t.total*(randSeeds.index(n))
                t.total+=subtotal#/(i+1) #average over all iterations
                t.total=t.total/(1+randSeeds.index(n))
                #print('total:  ',t.total)
                env.close()
                if t.total<-200.: 
                    break #if tree doesn't work well for first of batch, don't try others 
            #if best tree, store before moving to next tree   
            #print(t.total)
            
            if t.total>maxJ: 
                maxJ=t.total
                bestTree=t.name
                print(maxJ, bestTree, n)
    #either "tournament" or reseed random half of the forest
        #uncomment below for reseed random half of forest
        forest=evolution_reseed(Nn, Ni, Ne,Pc, Pm, forest)
        k+=1
        print(k, maxJ, bestTree)
        ProcessTime=time.process_time()-time0
        logger.info("Reward: {:.4f}, ProccesTime:  {:.6f}, Iteration:  {:.0f} \n".format(maxJ, ProcessTime, k))
    return maxJ, bestTree, k
 
        
if __name__ == "__main__":
    env = gym.make("MGM-v0")
    observation, info = env.reset(seed=42)   
    s=observation
    #will need to refactor for "online" programming, rather than having all measurments avilable at once

    #####notes on using gymnasium######
    #need size of observation space for number of states
    #need size of action space for number of MLC equations and limits to cap action
    # use list(env.observation_space.shape)[0] for size of obs, action spaces
    #use env.action_space.low(high).item() for action space min, max

    #use MLC for invertd double pendulum gymnasium per:  https://gymnasium.farama.org/environments/mujoco/inverted_double_pendulum/
    #generalize for any observation space size
    logger.info("Invoking user training script.")
    #try hyperparmeter parser per
    # https://github.com/aws/amazon-sagemaker-examples/blob/0efd885ef2a5c04929d10c5272681f4ca17dac17/advanced_functionality/custom-training-containers/framework-container/notebook/source_dir/train.py
    parser = argparse.ArgumentParser()

    # sagemaker-containers passes hyperparameters as arguments
    parser.add_argument("--Ni", type=int, default= 350)
    parser.add_argument("--Ne", type=int, default=3)
    parser.add_argument("--Nn", type=float, default=0.3)
    parser.add_argument("--maxi", type=int, default=250)
    parser.add_argument("--Pr", type=float, default=0.2)
    parser.add_argument("--Pm", type=float, default=0.5)

    args = parser.parse_args()

    #if os.path.exists(hyperparameters_file_path):
    #    hyp = load_json_object(hyperparameters_file_path)
    
    Ni=args.Ni#int(hyp['Ni'])#30#args.Ni #number of individuals
    Ne=args.Ne#int(hyp['Ne'])#3#args.Ne  #best Ne of each forest advanced unchanged
    Nn=args.Nn#float(hyp['Nn'])#0.3#args.Nn #for reseed, precent reseeded each round
    Ns=list(env.observation_space.shape)[0] #number of sensors
    n_batches=3#args.n_batches  #number of times to run through the environment with different seeds 
                #per tree per iteration to avoid lucky starts
    min_TreeSize=3#args.min_TreeSize
    max_TreeSize=10#args.max_TreeSize

    Pr=args.Pr #probably of replication
    Pm=args.Pm #probably of mutation
    Pc=1.0-Pr-Pm #probably of crossover

    maxi=args.maxi #set max number of iterations
    forest=[]
    #generate forest of Ni trees
    for i in range(0,Ni):
        seed=sapling(min_TreeSize,max_TreeSize,Ns)
        trial=growTrig(0,2,seed)
        forest.append(seed)
    time0=time.process_time()
    maxJ, bestTree, k=train(forest, maxi, Nn, Ni, Ne,Pr, Pc, Pm, time0)

    print("best:  ",maxJ, bestTree, k)
 