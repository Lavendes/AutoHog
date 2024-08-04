#include <cereal/archives/portable_binary.hpp>
#include <cereal/types/vector.hpp>
#include <fstream>
#include <memory>
#include <random>
#include <tfhe++.hpp>
#include <vector>

int main()
{
    // generate a random key
    std::unique_ptr<TFHEpp::SecretKey> sk =
        std::make_unique<TFHEpp::SecretKey>();
    std::unique_ptr<TFHEpp::GateKey> gk =
        std::make_unique<TFHEpp::GateKey>(*sk);

    // export the secret key to file for later use
    {
        std::ofstream ofs{"secret.key", std::ios::binary};
        cereal::PortableBinaryOutputArchive ar(ofs);
        sk->serialize(ar);
    };

    // export the cloud key to a file (for the cloud)
    {
        std::ofstream ofs{"cloud.key", std::ios::binary};
        cereal::PortableBinaryOutputArchive ar(ofs);
        gk->serialize(ar);
    };

    // get client input
    uint16_t client_input;
    std::cout << "Type client input (16bit unsigned interger)" << std::endl;
    std::cin >> client_input;

    // encrypt the input
    std::vector<uint8_t> p(16);
    for (int i; i < 16; i++) p[i] = (client_input >> i) & 1;
    std::vector<TFHEpp::TLWE<TFHEpp::lvl0param>> ciphertext =
        TFHEpp::bootsSymEncrypt(p, *sk);

    // export the ciphertexts to a file
    {
        std::ofstream ofs{"cloud.data", std::ios::binary};
        cereal::PortableBinaryOutputArchive ar(ofs);
        ar(ciphertext);
    };
}