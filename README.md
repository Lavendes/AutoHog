# AutoHoG: Automating Homomorphic Gate Design for Large-Scale Logic Circuit Evaluation

This code is the implementation of the paper "AutoHoG: Automating Homomorphic Gate Design for Large-Scale Logic Circuit Evaluation".

## Requirements

To set up the environment for this project, you will need the following:
```
git 
clang
cmake >= 3.16
python >= 3.10
```

### AutoHoG Dependencies

Install [Yosys Open Synthesis Suite](https://github.com/YosysHQ/yosys/tree/main). This can be done with the following commands:
```
git submodule update --init --recursive
sudo apt-get install build-essential clang lld bison flex \
	libreadline-dev gawk tcl-dev libffi-dev git \
	graphviz xdot pkg-config python3 libboost-system-dev \
	libboost-python-dev libboost-filesystem-dev zlib1g-dev
cd ./thirdparties/yosys/ && make && cd -
```

Install the required packages for python：
```
pip install -r requirements.txt
```

## Building AutohoG

You can build the AutoHoG for your machine by executing the following commands:
```
mkdir build
cd build
cmake -DCMAKE_CXX_COMPILER=clang++ ..
make && cd -
```

## Examples Test

To generate an optimized homomorphic circuit netlist (.json) in `./Test_Circuit/Json`  from a Verilog file (.v)  in `./Verilog_file`, execute the following command:
```
python run_circuit.py <circuit_name> 
```

For example, to optimize and execute `c432` circuit, use the following command:
```
python run_circuit.py c432
```

If the homomorphic circuit netlist has already been generated, use the following command to execute it:
```
python run_circuit.py <circuit_name> cal
```
You can modify the optimization parameters in `config.json` file to achieve different optimization results.

## Citation

To cite AutoHoG, please use the following BibTeX entries.
```
@article{guan2024autohog,
  title={AutoHoG: Automating Homomorphic Gate Design for Large-Scale Logic Circuit Evaluation},
  author={Guan, Zhenyu and Mao, Ran and Zhang, Qianyun and Zhang, Zhou and Zhao, Zian and Bian, Song},
  journal={IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems},
  year={2024},
  publisher={IEEE}
}
```
