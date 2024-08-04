import copy
import functools
import itertools as it
import json
import math
import time
from collections import deque
from itertools import permutations,groupby
from operator import not_, and_, or_ ,itemgetter
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from z3 import *
from collections import deque
import pickle
import argparse
import random

def json_to_dag(json_file):
    with open(json_file, 'r') as file:
        data = json.load(file)

    G = nx.MultiDiGraph()
    modules = data["modules"]
    module_key = next(iter(modules))  
    module = modules[module_key]  
    cells = module.get('cells', {})
    for cell_def in cells.values():
        if 'connections' in cell_def:
            cell_name = cell_def['connections']['Y'][0]  
            cell_type = cell_def['type'].replace('$_', '').replace('_', '')
            G.add_node(cell_name, type=cell_type)
            if 'A' in cell_def['connections']:
                source_node_a = cell_def['connections']['A'][0]
                G.add_edge(source_node_a, cell_name, sort=0) 
            if 'B' in cell_def['connections']:
                source_node_b = cell_def['connections']['B'][0]
                G.add_edge(source_node_b, cell_name, sort=1) 
    ports = module.get('ports', {})
    for port_name, port_attr in ports.items():
        port_type = port_attr['direction']
        count = 0
        for  bit_node in port_attr['bits']:
            count = count + 1
            if port_type == 'output':
                    new_name = f"{port_name}_{bit_node}"
                    G.add_node(new_name ,type=port_type)  
                    G.add_edge(bit_node, new_name )
            elif port_type == 'input':
                if bit_node in G:
                    G.nodes[bit_node]['type'] = port_type
                else:
                    G.add_node(bit_node, type=port_type)
    return G


def rename_input(json_file, G):
    with open(json_file, 'r') as file:
        data = json.load(file)
    modules = data["modules"]
    module_key = next(iter(modules))  # 获取第一个键
    module = modules[module_key]  # 通过键获取值
    ports = module.get('ports', {})
    for port_name, port_attr in ports.items():
        if port_attr['direction'] == 'input':
            for bit_id in port_attr['bits']:
                if bit_id in G:
                    original_attrs = G.nodes[bit_id]
                    new_id = f"{port_name}_{bit_id}"
                    
                    # 添加新节点并复制属性
                    G.add_node(new_id, **original_attrs)
                    
                    # 重建与原节点连接的边
                    for predecessor, edge_data in G.pred[bit_id].items():
                        for key, attr in edge_data.items():
                            G.add_edge(predecessor, new_id, key=key, **attr)
                    for successor, edge_data in G.succ[bit_id].items():
                        for key, attr in edge_data.items():
                            G.add_edge(new_id, successor, key=key, **attr)
                    
                    # 删除原节点
                    G.remove_node(bit_id)
                    
def define_io(dag, visited, node_id=None):
    subgraph = nx.MultiDiGraph(dag.subgraph(visited))
    output_dict = {}
    input_dict = {}
    output_node_id = 0
    if node_id == None:
        for node in visited:
            for source, target, key, data in dag.out_edges(node, data=True, keys=True):
                new_output_node = f"output_{output_node_id:03}"
                output_node_id += 1
                subgraph.add_node(new_output_node, type='output')
                subgraph.add_edge(node, new_output_node, **data)  
                output_dict[new_output_node] = (source, target, key)                    

    else:
        subgraph.add_node("output_node", type='output')
        subgraph.add_edge(node_id, "output_node")


    edge_list = []
    for target_node in visited:
        for source_node, edges_data in dag.pred[target_node].items():
            if source_node in visited:
                continue
            for edge_key, edge_data in edges_data.items():
                truthtable = edge_data.get('truthtable')
                if truthtable is not None:
                    truthtable = tuple(truthtable)
                edge_list.append((source_node, target_node, edge_key, truthtable))
    edge_list.sort(key=lambda x: (x[0], x[3]))  
    edge_groups = {k: list(v) for k, v in groupby(edge_list, key=lambda x: (x[0],x[3],0))}
    keys_to_divide = [k for k, v in edge_groups.items() if len(set((e[0], e[1]) for e in v)) < len(v)]
    for key in keys_to_divide:
        group1 = []
        group2 = []
        seen_pairs = set()
        for edge in edge_groups[key]:
            pair = (edge[0], edge[1])
            if pair in seen_pairs:
                group2.append(edge)
            else:
                group1.append(edge)
                seen_pairs.add(pair)
        edge_groups[key] = group1
        edge_groups[key[:-1] + (1,)] = group2

    input_dict = {}  
    for i, key in enumerate(edge_groups.keys()):
        input_node = f'input_{i:03}'
        subgraph.add_node(input_node, type='input')
        edges = edge_groups[key]
        input_dict[input_node] = [edges[0][0], edges[0][1], edges[0][2]]

        for edge in edge_groups[key]:
            
            attr_dict = dag.edges[edge[0], edge[1], edge[2]]
            subgraph.add_edge(input_node, edge[1],**attr_dict)
    return subgraph, input_dict, output_dict
    
def count_inputs(subgraph):
    count = 0
    for node, data in subgraph.nodes(data=True):  
        if data['type'] == 'input':  
            count += 1
    return count
 
def define_inputs(dag, node_list):
    edge_list = []
    for target_node in node_list:
        for source_node, edges_data in dag.pred[target_node].items():
            if source_node in node_list:
                continue
            for edge_key, edge_data in edges_data.items():
                truthtable = edge_data.get('truthtable')
                if truthtable is not None:
                    truthtable = tuple(truthtable)
                edge_list.append((source_node, target_node, edge_key, truthtable))
                
    edge_list.sort(key=lambda x: (x[0], x[3]))  
    edge_groups = {k: list(v) for k, v in groupby(edge_list, key=lambda x: (x[0],x[3],0))}
    keys_to_divide = [k for k, v in edge_groups.items() if len(set((e[0], e[1]) for e in v)) < len(v)]
    
    for key in keys_to_divide:
        group1 = []
        group2 = []
        seen_pairs = set()
        for edge in edge_groups[key]:
            pair = (edge[0], edge[1])
            if pair in seen_pairs:
                group2.append(edge)
            else:
                group1.append(edge)
                seen_pairs.add(pair)
        edge_groups[key] = group1
        edge_groups[key[:-1] + (1,)] = group2
    
    input_list = []
    for key, group in edge_groups.items():
        input_list.append(group[0][0])

    return input_list

def count_gate(dag):
    excluded_gates = {'input', 'output', 'NOT','BUFF'}
    gate_count = 0
    for node_id in dag.nodes():  
        node_data = dag.nodes[node_id]  
        if node_data.get('type') not in excluded_gates:  
            gate_count += 1
    return gate_count

def truthtable_cal(sorted_input_values, truthtable):
    input_str = "".join(str(value) for value in sorted_input_values)
    index = int(input_str, 2)
    output = int(truthtable[index])
    return output

def compute_node(graph, node, assignments, truthtable=None):
    node_type = graph.nodes[node]['type']
    if node_type == 'input':
        return assignments[node]
    preds = list(graph.predecessors(node))
      
    if node_type == 'output':
        assert len(preds) == 1
        pred_node = preds[0]
        pred_node_type = graph.nodes[pred_node]['type']
        if 'Hom' in pred_node_type:
            edge_data = graph[pred_node][node][0]
            truthtable_temp = edge_data.get('truthtable') if 'truthtable' in edge_data else None
            return compute_node(graph, pred_node, assignments, truthtable=truthtable_temp)
        return compute_node(graph, pred_node, assignments)

    sorted_input_values = []
    edge_vals_with_sort = []  
    for pred_node in preds:
        edges = graph.get_edge_data(pred_node, node, default={})
        for edge_key, edge_data in edges.items():
            sort = edge_data.get('sort', 0)
            pred_node_type = graph.nodes[pred_node]['type']
            if 'Hom' in pred_node_type:
                val = compute_node(graph, pred_node, assignments, truthtable=edge_data.get('truthtable', None))
            else:
                val = compute_node(graph, pred_node, assignments)
            edge_vals_with_sort.append((sort, val))

    edge_vals_with_sort.sort(key=lambda x: x[0])  
    sorted_input_values.extend([val for _, val in edge_vals_with_sort])
    
    
    if truthtable is None:
        op_type = graph.nodes[node]['type']
        result = logic_gate(op_type, *sorted_input_values)
    else:
        result = truthtable_cal(sorted_input_values, truthtable)
    return result

def logic_gate(op_type, *args):
    if op_type == 'AND':
        return int(all(args))
    elif op_type == 'ANDNOT':
        return int(args[0] and not args[1])
    elif op_type == 'NAND':
        return int(not all(args))
    elif op_type == 'NOR':
        return int(not any(args))
    elif op_type == 'NOT':
        return int(not args[0])
    elif op_type == 'OR':
        return int(any(args))
    elif op_type == 'ORNOT':
        return int(args[0] or not args[1])
    elif op_type == 'XNOR':
        return int(not bool(sum(args) % 2))
    elif op_type == 'XOR':
        return int(bool(sum(args) % 2))
    elif op_type == 'BUFF':
        return int(args[0])
    else:
        raise ValueError("Unsupported operation type")
 
def get_truth_table(graph):
    inputs = sorted([node for node, attr in graph.nodes(data=True) if attr['type'] == 'input'])
    outputs = [node for node, attr in graph.nodes(data=True) if attr['type'] == 'output']
    truth_tables = {}
    for output in outputs:
        truth_table_result = []
        for values in it.product([0, 1], repeat=len(inputs)):
            assignments = dict(zip(inputs, values))  
            output_value = compute_node(graph, output, assignments)
            truth_table_result.append(output_value)
        truth_tables[output] = truth_table_result
    return truth_tables

def Tconstruct(truth_table,upper):  
    n = len(truth_table)
    input_num = int(math.log(n, 2))

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
            size = len(new_array)
            if size <= upper:  
                size_opt = size
                w_opt = w
                array_opt = new_array
                return w_opt, array_opt
    return None,None

def gate_replace(dag, node_id):
    visited = visited_dict[node_id]
    for id in visited:
        if id not in dag or dag.nodes[id]['type'] == 'HomGateS'  :
            return
    dag.nodes[node_id]['type'] = 'HomGateS'   
    dag.nodes[node_id]['weights'] = weight_dict[node_id]
    for successor in dag.successors(node_id):
        dag.edges[node_id,successor,0]['tableT'] = tableT_dict[node_id]
        dag.edges[node_id,successor,0]['truthtable'] = truthtable_dict[node_id]
    edge_details = []
    for i, values in enumerate(input_dict[node_id].values()):
        source, target, key = values
        edge_data = dag[source][target][key]
        tableT = edge_data.get('tableT', None)
        truthtable = edge_data.get('truthtable', None)
        edge_details.append((source, node_id, i, tableT, truthtable))

    connected_nodes = [pred for pred in dag.predecessors(node_id) if pred in visited]
    for u, v in list(dag.in_edges(node_id)):
        dag.remove_edge(u, v)
    check_nodes = connected_nodes.copy()
    while check_nodes:
        current_node = check_nodes.pop(0)
        if dag.has_node(current_node): 
            check_nodes.extend([pred for pred in dag.predecessors(current_node) if pred in visited])
            
            if dag.out_degree(current_node) == 0:
                dag.remove_node(current_node)
       
    for detail in edge_details:
        source, node_id, sort_i, tableT, truthtable = detail
        dag.add_edge(source, node_id, sort=sort_i, tableT=tableT, truthtable=truthtable)

def sort_dict_by_length(temp_dict):
    sorted_items = sorted(temp_dict.items(), key=lambda item: len(item[1]), reverse=True)
    sorted_dict = {k: v for k, v in sorted_items}
    return sorted_dict
           
def find_sameinput(dag):
    same_input_nodes = {}
    for node in dag.nodes:
        node_type = dag.nodes[node].get('type')
        node_in_degree = dag.in_degree(node) 
        if node_type != 'HomGateS' or node_in_degree <= 5 :
            continue
        predecessors = list(dag.predecessors(node))
        input_weights = []  
        for pred in sorted(predecessors):
            weight_info = dag.nodes[node].get('weights', [])
            edge_sort = dag.edges[pred, node, 0].get('sort', None)
            weight = weight_info[edge_sort]
            input_weights.append((pred, weight))
        input_config = tuple(sorted(input_weights))  
        input_config_key = frozenset(input_config)
        if input_config_key in same_input_nodes:
            same_input_nodes[input_config_key].append(node)
        else:
            same_input_nodes[input_config_key] = [node]
    return same_input_nodes
      
def dfs(graph, start, end):
    stack = [(start, [start])]
    while stack:
        (vertex, path) = stack.pop()
        for next_node in set(graph.neighbors(vertex)) - set(path):
            if next_node == end:
                return True
            else:
                stack.append((next_node, path + [next_node]))
    return False

def gate_combine_1(dag, node_dic):
    primary_node = node_dic[0]  
    dag.nodes[primary_node]['type'] = 'HomGateM'   
    secondary_nodes = node_dic[1:]
    for node in secondary_nodes:
        for successor in dag.successors(node):
            for key in dag[node][successor]:
                edge_data = dag[node][successor][key]
                dag.add_edge(primary_node, successor, **edge_data)           
        dag.remove_node(node)
        
def combine_candidates(dag, target_node):   
    candidate_nodes = deque(node for node in dag.nodes if len(list(dag.predecessors(node))) <= 5 and (dag.nodes[node]['type'] not in ['input', 'output', 'HomGateM', 'NOT', 'BUFF']))
    # candidate_nodes = sorted(candidate_nodes, key=lambda x: abs(x - target_node))
    # random.shuffle(candidate_nodes)
    merge_candidates = [target_node]  
    inputs = []
    for node in candidate_nodes:
        all_targets = set(merge_candidates)
        all_targets.add(node)
        inputs_temp = define_inputs(dag, all_targets)
        if len(inputs_temp) > 5:
            continue
        if any(dfs(dag, node, m) or dfs(dag, m, node) for m in merge_candidates):
            continue
        merge_candidates.append(node)
        inputs = inputs_temp
    merge_candidates = list(set(merge_candidates))
    return merge_candidates,inputs

def gate_combine_2(dag, node_list, input_nodes):
    primary_node = node_list[0]
    subgraph, inputs, outputs = define_io(dag, node_list)
    truthtable = get_truth_table(subgraph)
    weight = [2 ** i for i in range(len(inputs)-1, -1, -1)]
    
    dag.nodes[primary_node]['type'] = 'HomGateM'
    dag.nodes[primary_node]['weights'] = weight
    edge_details = []

    for i, values in enumerate(inputs.values()):
        source, target, key = values
        edge_data = dag[source][target][key]
        tableT = edge_data.get('tableT', None)
        tt = edge_data.get('truthtable', None)
        edge_details.append((source, primary_node, i, tableT, tt))
    for u, v in list(dag.in_edges(primary_node)):
        dag.remove_edge(u, v)

    for key, values in outputs.items():
        source, target, key_ = values
        edge_data = dag[source][target][key_]
        sort = edge_data.get('sort', None)
        tt = truthtable[key]
        edge_details.append((primary_node, target, sort, tt, tt))
    for u, v in list(dag.out_edges(primary_node)):
        dag.remove_edge(u, v)
    for detail in edge_details:
        source, node_id, sort_i, tableT, truthtable = detail
        dag.add_edge(source, node_id, sort=sort_i, tableT=tableT, truthtable=truthtable)
        
    for node in node_list[1:]:
        dag.remove_node(node)

def update_dag(dag):
    truth_tables = {
        'AND': [0, 0, 0, 1],
        'ANDNOT': [0, 0, 1, 0], 
        'NAND': [1, 1, 1, 0],
        'NOR': [1, 0, 0, 0],
        'NOT': [1, 0], 
        'OR': [0, 1, 1, 1],
        'ORNOT': [0, 1, 0, 1], 
        'XNOR': [1, 0, 0, 1],
        'XOR': [0, 1, 1, 0],
        'BUFF':[0,1]
    }
    
    for node in list(dag.nodes):
        node_data = dag.nodes[node]
        if node_data['type'] not in ['input', 'output', 'HomGateS']:
            hom_type = 'Hom' + node_data['type']
            dag.nodes[node]['type'] = hom_type
            dag.nodes[node]['weights'] = [1, 2]
            
            preds = list(dag.predecessors(node))
            for i, pred in enumerate(preds):
                dag.edges[pred, node]['sort'] = i
                
            sucs = list(dag.successors(node))
            for suc in sucs:
                original_type = node_data['type'][3:]  
                if original_type in truth_tables:
                    dag.edges[node, suc]['tableT'] = truth_tables[original_type]
                    dag.edges[node, suc]['truthtable'] = truth_tables[original_type]

def save_graph(graph, path):
       with open(path, 'wb') as f:
           pickle.dump(graph, f)

#input
dag = nx.MultiDiGraph()

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('filename', type=str, help='File name to process')
parser.add_argument('inputnum_up', type=int, help='The upper limit of input number')
parser.add_argument('inputnum_low', type=int, help='The lower limit of input number')
parser.add_argument('replace_num', type=int, help='The number to replace')
args = parser.parse_args()
filename = args.filename
dag = json_to_dag('Verilog_file/' + filename + '.json')
inputnum_low = args.inputnum_low
inputnum_up =  args.inputnum_up
replace_num = args.replace_num
# print(f"Filename: {args.filename}, Inputnum_up: {args.inputnum_up}, Inputnum_low: {args.inputnum_low}, Replace_num: {args.replace_num}")

# filename = ''
# dag = json_to_dag('build/circuit/c2670-1.json')
# inputnum_low = 4
# inputnum_up =  6
# #gate size to be replaced
# replace_num = 3
dag_copy = dag.copy()
rename_input('Verilog_file/' + filename + '.json',dag_copy)
save_graph(dag_copy, 'Test_Circuit/Dag/'+ filename + '.pkl')

T1 = time.perf_counter()
weight_dict = {}
tableT_dict = {}
visited_dict = {} 
truthtable_dict = {} 
input_dict = {}

for node_id in dag.nodes:  
    queue = deque([node_id])
    visited = set()
    tableT = set()
    while queue:
        node = queue.popleft()
        node_type = dag.nodes[node]['type']  
        if node_type == 'input' or  node_type == 'output':
            continue
        visited.add(node)
        subgraph,input_temp,output = define_io(dag,visited,node_id)
        if  2 <= count_inputs(subgraph) <= inputnum_low - 1:     
            for predecessor in dag.predecessors(node):
                if dag.nodes[predecessor]['type'] == 'input':
                        continue
                if predecessor not in visited and predecessor not in queue:
                        queue.append(predecessor)
        if  inputnum_low <= count_inputs(subgraph) <= inputnum_up:  
            truth_table_temp = next(iter(get_truth_table(subgraph).values()))
            weight_temp, tableT_solution = Tconstruct(truth_table_temp,32)
            
            if weight_temp is not None:
                if tableT_solution is None:
                    tableT = [0, 0]
                else:
                    tableT = [item[1] for item in tableT_solution]
                weight = weight_temp
                truthtable = truth_table_temp   
                inputs = input_temp
                visited_record = list(visited) 
                for predecessor in dag.predecessors(node):
                    if dag.nodes[predecessor]['type'] == 'input':
                        continue
                    if predecessor not in visited and predecessor not in queue:
                        queue.append(predecessor)
            else:  
                visited.remove(node)
                  
        if tableT:
            visited_dict[node_id] = visited_record 
            weight_dict[node_id] = weight
            tableT_dict[node_id] = tableT
            truthtable_dict[node_id] = truthtable
            input_dict[node_id] = inputs

print('Original gate num: ',count_gate(dag))
sorted_node_ids = sort_dict_by_length(visited_dict)
for node_id in sorted_node_ids.keys():
    if len(visited_dict[node_id]) >= replace_num and dag.has_node(node_id):
        gate_replace(dag,node_id)
print('...')
weight_dict.clear()
tableT_dict.clear()
visited_dict.clear()
truthtable_dict.clear()
input_dict.clear()
sorted_node_ids.clear()

#multi-input Homgate(>5 input)
same_input_nodes = find_sameinput(dag)
sorted_same_input = sort_dict_by_length({key: value for key, value in same_input_nodes.items() if len(value)>1})
same_input_nodes_filt = []
for node_list in sorted_same_input.values():
    while node_list:
        primary_node = node_list.pop(0)
        new_list = [primary_node]
        for m in node_list[:]:
            if not (dfs(dag, primary_node, m) or dfs(dag, m, primary_node)):
                new_list.append(m)
                node_list.remove(m)
        if len(new_list) >= 2:
            same_input_nodes_filt.append(new_list)

for value in same_input_nodes_filt :
    gate_combine_1(dag, value)
same_input_nodes.clear()
sorted_same_input.clear()


#combine gate(<5 input)
nodes_copy = list(dag.nodes)
for node in nodes_copy:
    if node in dag.nodes:
        if  len(list(dag.predecessors(node))) <= 5 and (dag.nodes[node]['type'] not in ['input', 'output', 'HomGateM', 'NOT' ,'BUFF']):
            node_list, input_list = combine_candidates(dag, node)
            if len(node_list) > 1:
                gate_combine_2(dag, node_list, input_list)
            
print('Optimized gate num: ',count_gate(dag))

T2 = time.perf_counter()
# print("time: {} s".format(T2-T1))     

dag_copy = dag.copy()
rename_input('Verilog_file/' + filename + '.json',dag_copy)
save_graph(dag_copy, 'Test_Circuit/Dag/'+ filename + '_opt.pkl')

