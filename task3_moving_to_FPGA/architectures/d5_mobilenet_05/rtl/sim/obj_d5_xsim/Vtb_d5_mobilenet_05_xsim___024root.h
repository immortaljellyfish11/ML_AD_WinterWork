// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design internal header
// See Vtb_d5_mobilenet_05_xsim.h for the primary calling header

#ifndef VERILATED_VTB_D5_MOBILENET_05_XSIM___024ROOT_H_
#define VERILATED_VTB_D5_MOBILENET_05_XSIM___024ROOT_H_  // guard

#include "verilated.h"
#include "verilated_timing.h"


class Vtb_d5_mobilenet_05_xsim__Syms;

class alignas(VL_CACHE_LINE_BYTES) Vtb_d5_mobilenet_05_xsim___024root final {
  public:

    // DESIGN SPECIFIC STATE
    CData/*0:0*/ tb_d5_mobilenet_05_xsim__DOT__clk;
    CData/*0:0*/ tb_d5_mobilenet_05_xsim__DOT__rst_n;
    CData/*0:0*/ tb_d5_mobilenet_05_xsim__DOT__start;
    CData/*0:0*/ tb_d5_mobilenet_05_xsim__DOT__layer_done;
    CData/*0:0*/ tb_d5_mobilenet_05_xsim__DOT__layer_start;
    CData/*4:0*/ tb_d5_mobilenet_05_xsim__DOT__layer_id;
    CData/*2:0*/ tb_d5_mobilenet_05_xsim__DOT__op_kind;
    CData/*5:0*/ tb_d5_mobilenet_05_xsim__DOT__height;
    CData/*5:0*/ tb_d5_mobilenet_05_xsim__DOT__width;
    CData/*1:0*/ tb_d5_mobilenet_05_xsim__DOT__stride;
    CData/*0:0*/ tb_d5_mobilenet_05_xsim__DOT__busy;
    CData/*0:0*/ tb_d5_mobilenet_05_xsim__DOT__done;
    CData/*0:0*/ tb_d5_mobilenet_05_xsim__DOT__dut__DOT__active;
    CData/*0:0*/ __Vtrigprevexpr___TOP__tb_d5_mobilenet_05_xsim__DOT__clk__0;
    CData/*0:0*/ __Vtrigprevexpr___TOP__tb_d5_mobilenet_05_xsim__DOT__rst_n__0;
    CData/*0:0*/ __VactPhaseResult;
    CData/*0:0*/ __VinactPhaseResult;
    CData/*0:0*/ __VnbaPhaseResult;
    SData/*9:0*/ tb_d5_mobilenet_05_xsim__DOT__in_channels;
    SData/*9:0*/ tb_d5_mobilenet_05_xsim__DOT__out_channels;
    IData/*31:0*/ tb_d5_mobilenet_05_xsim__DOT__weight_offset;
    IData/*31:0*/ tb_d5_mobilenet_05_xsim__DOT__dut__DOT____VlemCall_0__layer_weight_bytes;
    IData/*31:0*/ __VactIterCount;
    IData/*31:0*/ __VinactIterCount;
    IData/*31:0*/ __Vi;
    VlUnpacked<QData/*63:0*/, 1> __VactTriggered;
    VlUnpacked<QData/*63:0*/, 1> __VactTriggeredAcc;
    VlUnpacked<QData/*63:0*/, 1> __VnbaTriggered;
    VlDelayScheduler __VdlySched;
    VlTriggerScheduler __VtrigSched_hc3011dec__0;
    VlTriggerScheduler __VtrigSched_hc3011a3e__0;

    // INTERNAL VARIABLES
    Vtb_d5_mobilenet_05_xsim__Syms* vlSymsp;
    const char* vlNamep;

    // CONSTRUCTORS
    Vtb_d5_mobilenet_05_xsim___024root(Vtb_d5_mobilenet_05_xsim__Syms* symsp, const char* namep);
    ~Vtb_d5_mobilenet_05_xsim___024root();
    VL_UNCOPYABLE(Vtb_d5_mobilenet_05_xsim___024root);

    // INTERNAL METHODS
    void __Vconfigure(bool first);
};


#endif  // guard
