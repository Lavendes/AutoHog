from z3 import *
import itertools as it
import math
import numpy as np
from itertools import permutations
from copy import deepcopy
import time


def Tconstruct_solver(truth_table):
    input_num = int(math.log(len(truth_table), 2))
    n = len(truth_table)
    
    X = [[0]*input_num for i in range(n)]
    for i, row in enumerate(it.product([1, 3], repeat=input_num)):
        X[i] = [num - 2 for num in row]

    w = [Int(f'w_{i}') for i in range(input_num)]
    a = [Int(f'a_{i}') for i in range(n)]  
    s = Optimize()
    for j in range(n):
        s.add(a[j] == sum(X[j][i] * w[i] for i in range(input_num)))
    for i in range(n):
        for j in range(i + 1, n):
            s.add(Implies(a[i] == a[j], truth_table[i] == truth_table[j]))
    for i in range(input_num):
        s.add(w[i] >= 0)
    max_a = Real('max_a')
    s.add([max_a >= a_i for a_i in a])
    s.minimize(max_a)
    if s.check() == sat:
        m = s.model()
        w_sol = [m[w_i].as_long() for w_i in w]
        a_sol = [m[a_i].as_long() for a_i in a]
        # num = eval_size(a_sol) 
        min0 = min(a_sol)/2    
        for i in range(n):
            a_sol[i]=int(a_sol[i]/2 - min0)
        temp = [list(t) for t in zip(a_sol,truth_table)]
        array = sorted(list(set([tuple(t) for t in temp])))
        return w_sol, array
    else:
        return None, None

truthtable = [1,1,1,1,1,1,0,1]

print(f'Truthtable: {truthtable}')
start_time = time.time() 
T_optimal, array_optimal = Tconstruct_solver(truthtable)
end_time = time.time() 
print(f'time: {end_time - start_time}s')
print(f'Optimal weight: {T_optimal}')
print(f'Optimal array: {array_optimal}')