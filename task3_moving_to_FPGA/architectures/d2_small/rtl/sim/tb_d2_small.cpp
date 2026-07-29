#include <array>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>
#include "Vd2_small_top.h"
#include "verilated.h"

// 中文：读取 Python 生成的整数向量，逐拍写入 CHW 输入缓冲，并核对分类和最大 logit。
// English: Load Python integer vectors, write CHW input cycle by cycle, and check class/max logit.
namespace {
constexpr size_t kInputSize = 3 * 32 * 32;
constexpr uint64_t kTimeout = 100000000ULL;
struct Test { std::array<int8_t,kInputSize> x{}; int label=0, cls=0; int32_t max=0; std::array<int32_t,10> logits{}; };

std::vector<Test> load(const char* input_path,const char* golden_path){
    std::ifstream fi(input_path),fg(golden_path); if(!fi||!fg) throw std::runtime_error("cannot open RTL vectors");
    std::vector<Test> tests;
    while(true){ Test t; int v; if(!(fi>>v)) break; if(v < -128 || v > 127) throw std::runtime_error("input outside INT8");
        t.x[0]=static_cast<int8_t>(v); for(size_t i=1;i<kInputSize;i++){if(!(fi>>v))throw std::runtime_error("truncated input");t.x[i]=static_cast<int8_t>(v);}
        if(!(fg>>t.label>>t.cls>>t.max))throw std::runtime_error("truncated golden header");
        for(auto &logit:t.logits)if(!(fg>>logit))throw std::runtime_error("truncated golden logits"); tests.push_back(t); }
    return tests;
}
void tick(Vd2_small_top& top,uint64_t& cycle){top.clk=0;top.eval();top.clk=1;top.eval();cycle++;}
void reset(Vd2_small_top& top,uint64_t& c){top.start=0;top.input_we=0;top.rst_n=0;for(int i=0;i<5;i++)tick(top,c);top.rst_n=1;tick(top,c);}
}
int main(int argc,char**argv){
    Verilated::commandArgs(argc,argv); auto tests=load("rtl/sim/data/input_int8.txt","rtl/sim/data/golden_logits.txt");
    if(tests.empty())throw std::runtime_error("no test vectors"); Vd2_small_top top; bool pass=true;
    for(size_t n=0;n<tests.size();n++){uint64_t c=0;reset(top,c);
        for(size_t a=0;a<kInputSize;a++){top.input_we=1;top.input_addr=a;top.input_data=static_cast<uint8_t>(tests[n].x[a]);tick(top,c);}
        top.input_we=0;tick(top,c);top.start=1;tick(top,c);top.start=0;const uint64_t begin=c;
        while(!top.done){if(c-begin>kTimeout)throw std::runtime_error("D2 RTL timeout");tick(top,c);}
        const int32_t rtl_max=static_cast<int32_t>(top.max_logit); bool one=top.class_id==tests[n].cls&&rtl_max==tests[n].max;
        std::cout<<"case="<<n<<" label="<<tests[n].label<<" class="<<int(top.class_id)<<" golden="<<tests[n].cls
                 <<" max_logit="<<rtl_max<<" golden_max="<<tests[n].max<<" cycles="<<(c-begin)<<" "<<(one?"PASS":"FAIL")<<"\n";pass&=one;}
    top.final();std::cout<<(pass?"D2 RTL PASS\n":"D2 RTL FAIL\n");return pass?0:1;
}
