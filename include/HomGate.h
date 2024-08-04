
#include <mulfft.hpp>
#include <tfhe++.hpp>


namespace TFHEpp
{   
    template <class P>
    void Linear(TLWE<P> &cres, const std::vector<TLWE<P>> &cipher, const std::vector<int> weight, const int t_size)
    {   
        int input_num = std::size(weight);
        for (int j = 0; j <= P::k * P::n; j++)  cres[j] =  0 ;
        int sum = 2 * (32 - t_size);
        for (int i = 0; i < input_num  ; i++)  sum = sum + weight[i];
        for (int i = 0; i < input_num  ; i++)
            for (int j = 0; j <= P::k * P::n; j++)   cres[j] =  cres[j] + uint64_t(cipher[i][j] * weight[i]);
        cres[P::k * P::n] += ( sum + 1 ) * (P::μ);
    }


    template <typename lvl2param::T μ = lvl2param::μ>
    void PBS_Single(TLWE<lvl2param> &res, const TLWE<lvl2param> &tlwe, const EvalKey &ek, const std::vector<std::vector<int>> &lut)
    {
       int output_num = std::size(lut);
        int t_size = std::size(lut[0]);
        TLWE<lvl0param> tlwelvl0;
        IdentityKeySwitch<lvl20param>(tlwelvl0, tlwe, *ek.iksklvl20);
        Polynomial<lvl2param> poly;
        for (typename lvl2param::T &p : poly) p = 0;
        int padding = 64;
        for (int j = 0; j < padding; j++) poly[j] = lvl2param::μ;
        TRLWE<lvl02param::targetP> acc, cres;
        BlindRotate<lvl02param>(acc, tlwelvl0, *ek.bkfftlvl02, poly);
        std::array<typename lvl2param::T, lvl2param::n> t;
        for (int i = 0; i < lvl2param::n; i++) t[i] = 0;
        for (int i = 0; i < t_size ; i++)  t[(i + 32 - t_size) * 64] = (lut[0][i]) * 2 - 1;
        for (int j = 0; j < lvl2param::k + 1; j++) PolyMul<lvl2param>(cres[j], acc[j], t);
        SampleExtractIndex<lvl02param::targetP>(res, cres, 0);
    }


    template <typename lvl2param::T μ = lvl2param::μ>
    void PBS_Multi(std::vector<TLWE<lvl2param>> &res, const TLWE<lvl2param> &tlwe, const EvalKey &ek, const std::vector<std::vector<int>> &lut)
    {
        int output_num = std::size(lut);
        int t_size = std::size(lut[0]);
        TLWE<lvl0param> tlwelvl0;
        IdentityKeySwitch<lvl20param>(tlwelvl0, tlwe, *ek.iksklvl20);
        Polynomial<lvl2param> poly;
        for (typename lvl2param::T &p : poly) p = 0;
        int padding = 64;
        for (int j = 0; j < padding; j++) poly[j] = lvl2param::μ;
        TRLWE<lvl02param::targetP> acc, cres;
        BlindRotate<lvl02param>(acc, tlwelvl0, *ek.bkfftlvl02, poly);
        for (int k = 0; k < output_num; k++)
        {
            std::array<typename lvl2param::T, lvl2param::n> t;
            for (int i = 0; i < lvl2param::n; i++) t[i] = 0;
            for (int i = 0; i < t_size ; i++)  t[(i + 32 - t_size) * 64] = (lut[k][i]) * 2 - 1;
            for (int j = 0; j < lvl2param::k + 1; j++) PolyMul<lvl2param>(cres[j], acc[j], t);
            SampleExtractIndex<lvl02param::targetP>(res[k], cres, 0);
        }
    }

    
    template <class P>
    void HomNOT(TLWE<P> &res, const TLWE<P> &ca)
    {
        for (int i = 0; i <= P::k * P::n; i++) res[i] = -ca[i];
    }

    
    template <class P, int casign, int cbsign, typename P::T offset>
    inline void HomGate(TLWE<P> &res, const TLWE<P> &ca, const TLWE<P> &cb,
                        const EvalKey &ek)
    {
        for (int i = 0; i <= P::k * P::n; i++)
            res[i] = 16 * (casign * ca[i] + cbsign * cb[i]);
        res[P::k * P::n] += 16 * offset;
        GateBootstrapping(res, res, ek);
    }
    
}

