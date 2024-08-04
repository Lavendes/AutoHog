import networkx as nx
import json
import pickle
import argparse
import random


def check_graph(graph):
    for node, node_data in graph.nodes(data=True):
        if node_data.get('type') in ['HomGateS', 'HomGateM']:
            weights = node_data.get('weights')
            if weights is None:
                return f'Node {node} with {node_data.get("type")} lacks weight attribute'

            in_edges_count = graph.in_degree(node)
            if in_edges_count != len(weights):
                return f'Node {node} has {in_edges_count} in-edges, but {len(weights)} weights.'

            plain = [random.randint(0, 1) for _ in range(len(weights))]

            result1 = int(''.join(map(str, plain)), 2)

            encoded_plain = [-0.5 if p == 0 else 0.5 for p in plain]
            result2 = sum(w * e for w, e in zip(weights, encoded_plain))
            weight_sum = sum(weights)
            result2 += weight_sum / 2

            for _, target, edge_data in graph.out_edges(node, data=True):
                tableT = edge_data.get('tableT')
                truthtable = edge_data.get('truthtable')
                if tableT is None or truthtable is None:
                    return f'Edge from {node} to {target} lacks tableT or truthtable attribute'

                truthtable_result = truthtable[result1]

                if result2 < len(tableT):
                    tableT_result = tableT[int(result2)]
                else:
                    tableT_result = tableT[int(result2) - len(tableT)] ^ 1

                if truthtable_result != tableT_result:
                    return {
                        'node': node,
                        'message': f'Node {node} has mismatched results: truthtable[{result1}] = {truthtable_result}, tableT[{int(result2)}] = {tableT_result}',
                        'tableT': tableT,
                        'truthtable': truthtable
                    }

    return 'Dag check passed ...'


def convert_to_custom_format(num):
    if num <= 26:
        return f"{chr(ord('a') + num)}"
    
def load_graph(path):
    with open(path, 'rb') as f:
        return pickle.load(f)
    
def add_sort_out_attributes(dag):
    for node in dag.nodes:
        in_degree = dag.in_degree(node)
        edge_attrs = {}
        for i, (src, dest, key) in enumerate(dag.out_edges(node, keys=True)):
            edge_attrs[(src, dest, key)] = {'sort_out': in_degree + i }
        nx.set_edge_attributes(dag, edge_attrs)


def modify_input_nodes_ids(dag):
    mapping = {}
    for node_id, node_attrs in dag.nodes(data=True):
        if node_attrs.get("type") == "input":
            new_node_id = f"{node_id}"
            mapping[node_id] = new_node_id
    nx.relabel_nodes(dag, mapping, copy=False)


def dag_to_json(dag):
    circuit_dict = {
        "circuit_name": "Example Circuit",  
        "ports": {},
        "cells": {},
    }

    count1 = 0
    for node_id in nx.topological_sort(dag):
        node_attrs = dag.nodes[node_id]
        if node_attrs['type'] in ['input', 'output']:
            connected_nodes = (dag.successors(node_id) if node_attrs['type'] == 'input' 
                               else dag.predecessors(node_id))
            circuit_dict["ports"][node_id] = {
                "direction": node_attrs['type'],
                "bits": list( str(node_id) for node_id in list(connected_nodes)) # 转换为列表，因为返回的是迭代器
            }
        else:
            # 获取入边和出边
            count1 = count1 + 1
            in_edges = list(dag.in_edges(node_id, keys=True))
            in_edges = sorted(in_edges, key=lambda edge: dag.get_edge_data(edge[0], edge[1], edge[2])['sort'])
            out_edges = list(dag.out_edges(node_id, keys=True))
            out_edges = sorted(out_edges, key=lambda edge: dag.get_edge_data(edge[0], edge[1], edge[2])['sort_out'])
            port_directions = {}
            connections = {} 
            weight_dict = {}
            for i, (src, dest, key) in enumerate(in_edges):
                if dag.has_edge(src, dest, key):
                    sort_value = dag[src][dest][key].get('sort')
                port_name = "${}$".format(convert_to_custom_format(sort_value))
                port_directions[port_name] = "input"
                if dag.nodes[src]['type'] in ['input', 'output']:
                    connections[port_name] = {"port": str(src)}
                else:
                    sort_out = dag.get_edge_data(src, node_id, key)['sort_out']
                    connections[port_name] = {
                        "cell": str(src),
                        "port": f"${convert_to_custom_format(sort_out)}$"
                    }
                if 'weights' in node_attrs:
                    weight_dict[port_name] = node_attrs['weights'][i]
            num_inputs = len(in_edges)  
            tableT_dict = {} 
            for i, (src, dest, key) in enumerate(out_edges):
                if dag.has_edge(src, dest, key):
                    sort_value = dag[src][dest][key].get('sort_out')
                port_name = "${}$".format(convert_to_custom_format(sort_value))
                port_directions[port_name] = "output"
 
                if dag.nodes[dest]['type'] in ['input', 'output']:
                    connections[port_name] = {"port": str(dest)}
                else:
                    sort_out = dag.get_edge_data(node_id, dest, key)['sort']
                    connections[port_name] = {
                        "cell": str(dest),
                        "port": f"${convert_to_custom_format(sort_out)}$"
                    }
               
                edge_data = dag.get_edge_data(src, dest, key)
                if 'tableT' in edge_data and edge_data['tableT'] is not None:
   
                    tableT_dict[port_name] = edge_data['tableT']

            formatted_cells_value = f'G{count1:04}'
            circuit_dict["cells"][formatted_cells_value ] = {
                "cell_name":str(node_id),
                "hide_name": 1,
                "type": node_attrs['type'],
                "parameters": {},
                "attributes": {},
                "port_directions": port_directions,
                "connections": connections,
                "weights": weight_dict,
                "tableT": tableT_dict
        }

    return circuit_dict

def format_dict(d, indent=0):
    spaces = '  ' * indent
    if isinstance(d, dict):
        if not d:
            formatted_str = '{'
        else :         
            formatted_str = '{\n' 
        items = list(d.items())
        for i, (key, value) in enumerate(items):
            formatted_key_value = f'{spaces}  "{key}": '
            if isinstance(value, dict):
                formatted_value = format_dict(value, indent + 1)
                formatted_str += (f'{formatted_key_value}{formatted_value}'
                                  if indent > 0 else f'{formatted_key_value}{formatted_value}')
            elif isinstance(value, list):
                formatted_list = '[' + ', '.join([format_dict(v, indent + 1) for v in value]) + ']'
                formatted_str += f'{formatted_key_value}{formatted_list}'
            else:
                formatted_str += (f'{formatted_key_value}"{value}"'
                                  if isinstance(value, str) else f'{formatted_key_value}{value}')
            if i < len(items) - 1:  
                formatted_str += ',\n'
        formatted_str += '\n' + spaces + '}' 
        return formatted_str
    elif isinstance(d, list):
        return '[' + ', '.join([format_dict(v, indent) for v in d]) + ']'
    else:
        return f'"{d}"' if isinstance(d, str) else str(d)
    

def save_formatted_dict_to_file(dct, file_path):
    formatted_dict = format_dict(dct)
    with open(file_path, 'w') as f:
        f.write(formatted_dict)



parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('filename', type=str, help='File name')
args = parser.parse_args()
filename = args.filename
dag = load_graph('Test_Circuit/Dag/'+ filename + '_opt.pkl')
print(check_graph(dag))
add_sort_out_attributes(dag)
modify_input_nodes_ids(dag)
save_formatted_dict_to_file(dag_to_json(dag), 'Test_Circuit/Json/'+ filename + '.json')
