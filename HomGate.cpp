//evaluate a compound Homgate


#ifdef USE_PERFforlvl
#include <gperftools/profiler.h>
#endif

#include "HomGate.h"
#include "timer.h"
#include <algorithm> 

using namespace std;
using namespace TFHEpp;


int main()
{
    random_device seed_gen;
    default_random_engine engine(seed_gen());
    uniform_int_distribution<uint32_t> binary(0, 1);
    int input_num = 10;
    int output_num = 4;

    SecretKey* sk = new SecretKey();
    uint8_t plain[input_num],pres[output_num];
    for (int i = 0; i < input_num; i++) plain[i] = binary(engine) > 0;
    vector<vector<int>> lut={
        {0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
    };
    vector<int> weight={1,8,1,8,1,8,1,1,1,1};
    

    TFHEpp::EvalKey ek;
    ek.emplacebkfft<TFHEpp::lvl02param>(*sk);
    ek.emplaceiksk<TFHEpp::lvl20param>(*sk);

    std::vector<TLWE<TFHEpp::lvl2param>> cipher_in(input_num),cipher_out(output_num);
    TLWE<TFHEpp::lvl2param> linear;
    uint64_t μ = 1ULL << ( 64-7 );
    for (int i = 0; i < input_num; i++) cipher_in[i] = bootsSymEncrypt<lvl2param>(plain[i], *sk);
    
    double Bootstrapping_time2{0.};
    {
    AutoTimer timer(&Bootstrapping_time2);
    TFHEpp::Linear<lvl2param>(linear, cipher_in, weight, size(lut[0]));
    TFHEpp::PBS_Multi(cipher_out, linear, ek, lut);
    //HomGate<TFHEpp::lvl2param, -1, -1, lvl2param::μ>(cipher_out[0], cipher_in[0], cipher_in[1], ek);
    
    }   
    cout << endl;
    for (int k = 0; k < output_num; k++)  pres[k] = bootsSymDecrypt<lvl2param>(cipher_out[k],*sk);   
    std::cout << " res = ";
    for (size_t i = 0; i < output_num; ++i) {
        std::cout << static_cast<int>(pres[i]) << " ";
    }

    std::vector<double> encoded_plain(input_num);
    for (int i = 0; i < input_num; ++i) {
        encoded_plain[i] = plain[i] == 0 ? -0.5 : 0.5;
    }
    double result = 0;
    for (int i = 0; i < weight.size(); ++i) {
        result += weight[i] * encoded_plain[i];
    }
    double weight_sum = std::accumulate(weight.begin(), weight.end(), 0);
    result += weight_sum / 2;
    int index = static_cast<int>(result);
    
    std::cout << endl << "pres = ";
    for (const auto& row : lut) {
        if (index < row.size())
            std::cout << row[index] << " ";
        else
            std::cout << 1 - row[index - row.size()] << " ";
    }

    std::cout <<  std::endl ;
    printf("Bootstrapping.time %f ms", Bootstrapping_time2);
    std::cout <<  std::endl ;
    std::cout <<  std::endl ;
    return 0;
}
      
