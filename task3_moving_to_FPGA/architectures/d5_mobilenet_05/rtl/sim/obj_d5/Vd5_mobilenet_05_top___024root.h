// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design internal header
// See Vd5_mobilenet_05_top.h for the primary calling header

#ifndef VERILATED_VD5_MOBILENET_05_TOP___024ROOT_H_
#define VERILATED_VD5_MOBILENET_05_TOP___024ROOT_H_  // guard

#include "verilated.h"


class Vd5_mobilenet_05_top__Syms;

class alignas(VL_CACHE_LINE_BYTES) Vd5_mobilenet_05_top___024root final {
  public:

    // DESIGN SPECIFIC STATE
    VL_IN8(clk,0,0);
    VL_IN8(rst_n,0,0);
    VL_IN8(start,0,0);
    VL_IN8(layer_done,0,0);
    VL_OUT8(layer_start,0,0);
    VL_OUT8(layer_id,4,0);
    VL_OUT8(op_kind,2,0);
    VL_OUT8(height,5,0);
    VL_OUT8(width,5,0);
    VL_OUT8(stride,1,0);
    VL_OUT8(busy,0,0);
    VL_OUT8(done,0,0);
    CData/*0:0*/ d5_mobilenet_05_top__DOT__active;
    CData/*0:0*/ __Vtrigprevexpr___TOP__clk__0;
    CData/*0:0*/ __Vtrigprevexpr___TOP__rst_n__0;
    CData/*0:0*/ __VactPhaseResult;
    CData/*0:0*/ __VnbaPhaseResult;
    VL_OUT16(in_channels,9,0);
    VL_OUT16(out_channels,9,0);
    VL_OUT(weight_offset,31,0);
    IData/*31:0*/ d5_mobilenet_05_top__DOT____VlemCall_0__layer_weight_bytes;
    IData/*31:0*/ __VactIterCount;
    VlUnpacked<QData/*63:0*/, 1> __VactTriggered;
    VlUnpacked<QData/*63:0*/, 1> __VnbaTriggered;

    // INTERNAL VARIABLES
    Vd5_mobilenet_05_top__Syms* vlSymsp;
    const char* vlNamep;

    // CONSTRUCTORS
    Vd5_mobilenet_05_top___024root(Vd5_mobilenet_05_top__Syms* symsp, const char* namep);
    ~Vd5_mobilenet_05_top___024root();
    VL_UNCOPYABLE(Vd5_mobilenet_05_top___024root);

    // INTERNAL METHODS
    void __Vconfigure(bool first);
};


#endif  // guard
