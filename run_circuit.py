import sys
import subprocess
import random
import pickle
import json


def load_graph(path):
    with open(path, 'rb') as f:
        return pickle.load(f)
    
circuit_name = sys.argv[1]
optional_param = sys.argv[2] if len(sys.argv) > 2 else None

with open('config.json', 'r') as config_file:
    config = json.load(config_file)
    
searchnum_up = config.get('searchnum_up', 5)
searchnum_low = config.get('searchnum_low', 5)
replace_num = config.get('replace_num', 3)

print('Circuit:', circuit_name)

if optional_param != 'cal':
    verilog_file_path = f"Verilog_file/{circuit_name}"
    template_file = 'build_template.ys'
    output_file = 'build.ys'
    
    with open(template_file, 'r') as file:
        template = file.read()

    ys_content = template.replace("{filename}", verilog_file_path)
    with open(output_file, 'w') as file:
        file.write(ys_content)

    print('Generating netlist from Verilog...')
    yosys_result = subprocess.run(['./thirdparties/yosys/yosys', output_file], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    print()

    print('Optimizing netlist...')
    subprocess.run(['python', 'search_replace.py', circuit_name, str(searchnum_up), str(searchnum_low), str(replace_num)])
    print()

G = load_graph('Test_Circuit/Dag/'+ circuit_name + '.pkl')
inputnum = len([node for node, attr in G.nodes(data=True) if attr.get('type') == 'input'])

subprocess.run(['python', 'dag2json.py', circuit_name])
print('Transforming DAG to Json file...')

print()
print('-----Executing Circuit------')
input_value = ''.join(random.choice('01') for _ in range(inputnum))
#print(input_value)
result = subprocess.run(["./build/CircuitEval", circuit_name, input_value], capture_output=True, text=True)
print(result.stdout)
