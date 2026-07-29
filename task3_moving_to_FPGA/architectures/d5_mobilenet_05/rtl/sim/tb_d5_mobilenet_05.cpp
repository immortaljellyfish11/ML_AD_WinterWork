#include "Vd5_mobilenet_05_top.h"
#include "verilated.h"
#include <iostream>
#include <stdexcept>
namespace { void tick(Vd5_mobilenet_05_top& d){d.clk=0;d.eval();d.clk=1;d.eval();} }
int main(int argc,char**argv){
    Verilated::commandArgs(argc,argv); Vd5_mobilenet_05_top d; d.clk=0;d.rst_n=0;d.start=0;d.layer_done=0;
    for(int i=0;i<3;i++)tick(d);d.rst_n=1;tick(d);d.start=1;tick(d);d.start=0;
    for(int id=0;id<=22;id++){
        if(!d.layer_start||d.layer_id!=id) throw std::runtime_error("descriptor sequence mismatch");
        if(id==13||id==15){if(d.op_kind!=1||d.in_channels!=256||d.out_channels!=256)throw std::runtime_error("256 depthwise descriptor mismatch");}
        if(id==14||id==16){if(d.op_kind!=2||d.in_channels!=256||d.out_channels!=256)throw std::runtime_error("256 pointwise descriptor mismatch");}
        d.layer_done=1;tick(d);d.layer_done=0;
    }
    if(!d.done) throw std::runtime_error("D5 did not finish");
    std::cout<<"D5 external-weight sequencer PASS: stem + 10 blocks + GAP + FC\n"; d.final();return 0;
}
