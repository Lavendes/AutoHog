#ifdef USE_PERFforlvl
#include <gperftools/profiler.h>
#endif

#include <map>
#include <cassert>
#include <chrono>
#include <iostream>
#include <fstream>
#include <random>
#include <string>
#include "json.hpp"
#include "HomGate.h"
#include "timer.h"
using namespace std;
using namespace TFHEpp;
using json = nlohmann::json;

int main(int argc, char* argv[]) 
{   
    if (argc < 3) {
        cerr << "Usage: " << argv[0] << " <filename> <binary_sequence>" << endl;
        return 1;
    }

    string filename = argv[1];
    string sequence = argv[2]; // 读取传递的序列
    string filepath = "Test_Circuit/Json/" + filename + ".json";
    ifstream file(filepath);
    if (!file.is_open()) {
        cerr << "Failed to open file: " << filepath << endl;
        return 1;
    }
    nlohmann::json circuit;
    try {
        file >> circuit;
    } catch (const std::exception& e) {
        cerr << "Failed to parse JSON: " << e.what() << endl;
        return 1;
    }
    file.close();
    
    int input_num = 0;
    int output_num = 0;
    auto ports =  circuit["ports"];
    for (const auto& [key, value] : ports.items()) {
        string direction = value["direction"];
        if (direction == "input") {
            input_num++;
        } else if (direction == "output") {
            output_num++;
        }
    }

    // 确保 sequence 的长度与 input_num 匹配
    if (sequence.length() != input_num) {
        cerr << "Error: Binary sequence length does not match input number" << endl;
        return 1;
    }

    SecretKey* sk = new SecretKey();
    TFHEpp::EvalKey ek;
    ek.emplacebkfft<TFHEpp::lvl02param>(*sk);
    ek.emplaceiksk<TFHEpp::lvl20param>(*sk);

    uint8_t plain[input_num];
    for (int i = 0; i < input_num; i++) {
        plain[i] = (sequence[i] == '1') ? 1 : 0;
    }
    int temp_t = 0;
    map<string, TLWE<TFHEpp::lvl2param>> input_dict;
    cout << "Input values:" << endl ;
    for (const auto& [key, value] : ports.items()) {
        string direction = value["direction"];
        if (direction == "input") {
            input_dict[key] = bootsSymEncrypt<TFHEpp::lvl2param>(plain[temp_t], *sk);
            cout << key << ":" << static_cast<int>(plain[temp_t]) << endl;
            temp_t ++;
        }
    }
    map<string, TLWE<TFHEpp::lvl2param>> output_dict;
    map<string, map<string, TLWE<TFHEpp::lvl2param>>> cell_dict;
    
    double time2{0.};
    {
    AutoTimer timer(&time2);
    for (const auto& [cell_key, cell_value] : circuit["cells"].items()) {
        vector<TLWE<TFHEpp::lvl2param>> cipher_in, cipher_out;
        string cell_type = cell_value["type"];
        int out_num = 0;
        const auto& port_directions = cell_value["port_directions"];
        const auto& connections = cell_value["connections"];

        for (const auto& [port_key, direction] : port_directions.items()) {
            if (direction == "input") {
                string port = connections[port_key]["port"];
                if (connections[port_key].contains("cell")) {
                    cipher_in.push_back(cell_dict[connections[port_key]["cell"]][port]);
                } else {
                    cipher_in.push_back(input_dict[port]);
                }
            }
            if (direction == "output")    out_num ++;
        }
        cipher_out.resize(out_num);

        if (cell_type == "HomGateM") {
            vector<int> weight;
            vector<vector<int>> lut;
            for (const auto& weight_value : cell_value["weights"]) weight.push_back(weight_value);
            for (const auto& lut_entry : cell_value["tableT"].items()) {
                vector<int> lut_row;
                for (const auto& lut_element : lut_entry.value()) {
                    lut_row.push_back(lut_element);
                }
                lut.push_back(lut_row);
            }
            
            TLWE<TFHEpp::lvl2param> linear;
            TFHEpp::Linear<TFHEpp::lvl2param>(linear, cipher_in, weight, size(lut[0]));
            cipher_out.resize(lut.size());
            TFHEpp::PBS_Multi(cipher_out, linear, ek, lut);
        }
        else if(cell_type == "HomGateS")
        {
            vector<int> weight;
            vector<vector<int>> lut;
            for (const auto& weight_value : cell_value["weights"]) weight.push_back(weight_value);
            for (const auto& lut_entry : cell_value["tableT"].items()) {
                vector<int> lut_row;
                for (const auto& lut_element : lut_entry.value()) {
                    lut_row.push_back(lut_element);
                }
                lut.push_back(lut_row);
            }
            TLWE<TFHEpp::lvl2param> linear;
            TFHEpp::Linear<TFHEpp::lvl2param>(linear, cipher_in, weight, size(lut[0]));
            TLWE<TFHEpp::lvl2param> res;
            cipher_out.resize(lut.size());
            TFHEpp::PBS_Single(res, linear, ek, lut);
            fill(cipher_out.begin(), cipher_out.end(), res);
        }
        
        else {
            TLWE<TFHEpp::lvl2param> res;
            if (cell_type == "NOT") TFHEpp::HomNOT<TFHEpp::lvl2param>(res, cipher_in[0]);
            else if (cell_type == "AND") HomGate<TFHEpp::lvl2param, 1, 1, -lvl2param::μ>(res, cipher_in[0], cipher_in[1], ek);
            else if (cell_type == "NAND") HomGate<TFHEpp::lvl2param, -1, -1, lvl2param::μ>(res, cipher_in[0], cipher_in[1], ek);
            else if (cell_type == "NOR") HomGate<TFHEpp::lvl2param, -1, -1, -lvl2param::μ>(res, cipher_in[0], cipher_in[1], ek);
            else if (cell_type == "OR")  HomGate<TFHEpp::lvl2param, 1, 1, lvl2param::μ>(res, cipher_in[0], cipher_in[1], ek);
            else if (cell_type == "XNOR") HomGate<TFHEpp::lvl2param, -2, -2, -2 * lvl2param::μ>(res, cipher_in[0], cipher_in[1], ek);
            else if (cell_type == "XOR")  HomGate<TFHEpp::lvl2param, 2, 2, 2 * lvl2param::μ>(res, cipher_in[0], cipher_in[1], ek);
            else if (cell_type == "ANDNOT")  HomGate<TFHEpp::lvl2param, 1, -1, -lvl2param::μ>(res, cipher_in[0], cipher_in[1], ek);
            else if (cell_type == "ORNOT")  HomGate<TFHEpp::lvl2param, 1, -1, lvl2param::μ>(res, cipher_in[0], cipher_in[1], ek);
            fill(cipher_out.begin(), cipher_out.end(), res);
        }
        
        size_t index = 0;
        for (const auto& [port_key, direction] : port_directions.items()) {
            if (direction == "output") {
                if (connections[port_key].contains("cell")) {
                    
                    cell_dict[cell_value["cell_name"]][port_key] = cipher_out[index];
                    cipher_out.pop_back();
                } else {
                    output_dict[connections[port_key]["port"]] = cipher_out[index];
                    cipher_out.pop_back();
                }
            index ++ ;
            }
        } 
    }
    }
    cout << endl;   
    cout << "Output values:" << endl ;
    for (const auto& [key, value] : output_dict) 
            cout << key << ":" << static_cast<int>(bootsSymDecrypt<lvl2param>(value,*sk)) << endl;
    cout << endl; 
    cout << endl; 
    cout << "Evaluation.time " << fixed << setprecision(3) << time2 << " ms" << endl;
    return 0;
}
      
