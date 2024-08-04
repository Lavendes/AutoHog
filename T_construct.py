from z3 import *
import itertools as it
import math
import numpy as np
from itertools import permutations
from copy import deepcopy
import time
import random


def eval_size(arr):
    N = len(arr)
    for i in range(math.ceil(N/2), N):
        tt = 0
        for j in range(i+1,N):
            if arr[tt][1] == arr[j][1] : break
            tt = tt+1
            if j==N-1 :  return i
    return i


def Tconstruction(truth_table,upper):  
    n = len(truth_table)
    input_num = int(math.log(n, 2))

    # truthtable generation
    X = np.array(list(it.product([1, 3], repeat=input_num))) - 2
    tt = np.column_stack((X, truth_table))

    str1 = np.array([" ".join(str(a) for a in row) for row in tt])
    index0, index = [], []
    count = 0
    flag = np.ones(input_num, dtype=bool)
    for i in range(input_num):
        if not flag[i]: continue
        tt1 = tt.copy()
        tt1[:, i] = -tt1[:, i]
        str2 = np.array([" ".join(str(a) for a in row) for row in tt1])
        if np.array_equal(np.sort(str1), np.sort(str2)):
            index0.append(i)
            flag[i] = 0
        else:
            index.append([i])
            for j in range(i+1, input_num):
                tt1 = tt.copy()
                tt1[:, [i, j]] = tt1[:, [j, i]]
                str2 = np.array([" ".join(str(a) for a in row) for row in tt1])
                if np.array_equal(np.sort(str1), np.sort(str2)):
                    index[count].append(j)
                    flag[j] = 0
            count += 1

    #生成系数        
    if not index:
        return [0 for _ in range(len(index0))], None
    coeffs_values = 2**np.arange(input_num)
    w_values = []
    for perm in permutations(coeffs_values, len(index)):
        w = [0] * (input_num)
        for idx, group_values in enumerate(perm):
            for i in index[idx]:
                w[i] = group_values
        if sum(w) < upper * 2:
            w_values.append(w)
            
    w_opt, array_opt = None, None
    found = False
    for w in w_values:
        if found:
            break  
        c = np.dot(X, w)
        array = list(set(tuple(t) for t in np.column_stack((c, truth_table))))
        flag = 1
        array.sort()
        for j in range(len(array) - 1):
            if array[j][0] == array[j+1][0]:
                flag = 0
                break 
        if flag == 0:
            continue
        else:
            min_index = array[0][0]
            max_index = array[-1][0]
            new_array = [(i, 0) for i in range((max_index - min_index) // 2 + 1)]
            for old_index, value in array:
                new_index = (old_index - min_index) // 2
                new_array[new_index] = (new_index, value)
            size = eval_size(new_array)
            #size = len(array)
            if size <= upper:  
                size_opt = size
                w_opt = w
                array_opt = new_array[:size]
                return w_opt, array_opt
    return None,None

# b= np.ones(2**14 - 1)
truthtable = [1,1,1,1,1,0,1,1,1,1,0,1,1,1,1,1]
print(f'Truthtable: {truthtable}')
start_time = time.time() 
T_optimal, array_optimal = Tconstruction(truthtable,8)
end_time = time.time()
print(f'time: {end_time - start_time}s')
print(f'Optimal weight: {T_optimal}')
print(f'Optimal array: {array_optimal}')