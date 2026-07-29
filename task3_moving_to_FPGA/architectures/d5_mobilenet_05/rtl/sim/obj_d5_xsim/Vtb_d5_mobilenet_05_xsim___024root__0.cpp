// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design implementation internals
// See Vtb_d5_mobilenet_05_xsim.h for the primary calling header

#include "Vtb_d5_mobilenet_05_xsim__pch.h"

VlCoroutine Vtb_d5_mobilenet_05_xsim___024root___eval_initial__TOP__Vtiming__0(Vtb_d5_mobilenet_05_xsim___024root* vlSelf);
VlCoroutine Vtb_d5_mobilenet_05_xsim___024root___eval_initial__TOP__Vtiming__1(Vtb_d5_mobilenet_05_xsim___024root* vlSelf);

void Vtb_d5_mobilenet_05_xsim___024root___eval_initial(Vtb_d5_mobilenet_05_xsim___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtb_d5_mobilenet_05_xsim___024root___eval_initial\n"); );
    Vtb_d5_mobilenet_05_xsim__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    Vtb_d5_mobilenet_05_xsim___024root___eval_initial__TOP__Vtiming__0(vlSelf);
    Vtb_d5_mobilenet_05_xsim___024root___eval_initial__TOP__Vtiming__1(vlSelf);
}

void Vtb_d5_mobilenet_05_xsim___024root____VbeforeTrig_hc3011dec__0(Vtb_d5_mobilenet_05_xsim___024root* vlSelf, const char* __VeventDescription);
void Vtb_d5_mobilenet_05_xsim___024root____VbeforeTrig_hc3011a3e__0(Vtb_d5_mobilenet_05_xsim___024root* vlSelf, const char* __VeventDescription);

VlCoroutine Vtb_d5_mobilenet_05_xsim___024root___eval_initial__TOP__Vtiming__0(Vtb_d5_mobilenet_05_xsim___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtb_d5_mobilenet_05_xsim___024root___eval_initial__TOP__Vtiming__0\n"); );
    Vtb_d5_mobilenet_05_xsim__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Locals
    IData/*31:0*/ tb_d5_mobilenet_05_xsim__DOT__id;
    tb_d5_mobilenet_05_xsim__DOT__id = 0;
    IData/*31:0*/ tb_d5_mobilenet_05_xsim__DOT__unnamedblk1_1__DOT____Vrepeat0;
    tb_d5_mobilenet_05_xsim__DOT__unnamedblk1_1__DOT____Vrepeat0 = 0;
    IData/*31:0*/ __Vtask_tb_d5_mobilenet_05_xsim__DOT__check_descriptor__0__expected_id;
    __Vtask_tb_d5_mobilenet_05_xsim__DOT__check_descriptor__0__expected_id = 0;
    IData/*31:0*/ __Vtask_tb_d5_mobilenet_05_xsim__DOT__check_descriptor__0____VlefCall_1__expected_offset;
    __Vtask_tb_d5_mobilenet_05_xsim__DOT__check_descriptor__0____VlefCall_1__expected_offset = 0;
    IData/*31:0*/ __Vtask_tb_d5_mobilenet_05_xsim__DOT__check_descriptor__0____VlefCall_0__expected_offset;
    __Vtask_tb_d5_mobilenet_05_xsim__DOT__check_descriptor__0____VlefCall_0__expected_offset = 0;
    IData/*31:0*/ __Vfunc_tb_d5_mobilenet_05_xsim__DOT__expected_offset__1__Vfuncout;
    __Vfunc_tb_d5_mobilenet_05_xsim__DOT__expected_offset__1__Vfuncout = 0;
    IData/*31:0*/ __Vfunc_tb_d5_mobilenet_05_xsim__DOT__expected_offset__1__id;
    __Vfunc_tb_d5_mobilenet_05_xsim__DOT__expected_offset__1__id = 0;
    IData/*31:0*/ __Vfunc_tb_d5_mobilenet_05_xsim__DOT__expected_offset__1____VlefCall_0__weight_bytes;
    __Vfunc_tb_d5_mobilenet_05_xsim__DOT__expected_offset__1____VlefCall_0__weight_bytes = 0;
    IData/*31:0*/ __Vfunc_tb_d5_mobilenet_05_xsim__DOT__expected_offset__1__i;
    __Vfunc_tb_d5_mobilenet_05_xsim__DOT__expected_offset__1__i = 0;
    IData/*31:0*/ __Vfunc_tb_d5_mobilenet_05_xsim__DOT__weight_bytes__2__Vfuncout;
    __Vfunc_tb_d5_mobilenet_05_xsim__DOT__weight_bytes__2__Vfuncout = 0;
    IData/*31:0*/ __Vfunc_tb_d5_mobilenet_05_xsim__DOT__weight_bytes__2__id;
    __Vfunc_tb_d5_mobilenet_05_xsim__DOT__weight_bytes__2__id = 0;
    IData/*31:0*/ __Vfunc_tb_d5_mobilenet_05_xsim__DOT__expected_offset__3__Vfuncout;
    __Vfunc_tb_d5_mobilenet_05_xsim__DOT__expected_offset__3__Vfuncout = 0;
    IData/*31:0*/ __Vfunc_tb_d5_mobilenet_05_xsim__DOT__expected_offset__3__id;
    __Vfunc_tb_d5_mobilenet_05_xsim__DOT__expected_offset__3__id = 0;
    IData/*31:0*/ __Vfunc_tb_d5_mobilenet_05_xsim__DOT__expected_offset__3____VlefCall_0__weight_bytes;
    __Vfunc_tb_d5_mobilenet_05_xsim__DOT__expected_offset__3____VlefCall_0__weight_bytes = 0;
    IData/*31:0*/ __Vfunc_tb_d5_mobilenet_05_xsim__DOT__expected_offset__3__i;
    __Vfunc_tb_d5_mobilenet_05_xsim__DOT__expected_offset__3__i = 0;
    IData/*31:0*/ __Vfunc_tb_d5_mobilenet_05_xsim__DOT__weight_bytes__4__Vfuncout;
    __Vfunc_tb_d5_mobilenet_05_xsim__DOT__weight_bytes__4__Vfuncout = 0;
    IData/*31:0*/ __Vfunc_tb_d5_mobilenet_05_xsim__DOT__weight_bytes__4__id;
    __Vfunc_tb_d5_mobilenet_05_xsim__DOT__weight_bytes__4__id = 0;
    IData/*31:0*/ __Vtask_tb_d5_mobilenet_05_xsim__DOT__check_descriptor__5__expected_id;
    __Vtask_tb_d5_mobilenet_05_xsim__DOT__check_descriptor__5__expected_id = 0;
    IData/*31:0*/ __Vtask_tb_d5_mobilenet_05_xsim__DOT__check_descriptor__5____VlefCall_1__expected_offset;
    __Vtask_tb_d5_mobilenet_05_xsim__DOT__check_descriptor__5____VlefCall_1__expected_offset = 0;
    IData/*31:0*/ __Vtask_tb_d5_mobilenet_05_xsim__DOT__check_descriptor__5____VlefCall_0__expected_offset;
    __Vtask_tb_d5_mobilenet_05_xsim__DOT__check_descriptor__5____VlefCall_0__expected_offset = 0;
    IData/*31:0*/ __Vfunc_tb_d5_mobilenet_05_xsim__DOT__expected_offset__6__Vfuncout;
    __Vfunc_tb_d5_mobilenet_05_xsim__DOT__expected_offset__6__Vfuncout = 0;
    IData/*31:0*/ __Vfunc_tb_d5_mobilenet_05_xsim__DOT__expected_offset__6__id;
    __Vfunc_tb_d5_mobilenet_05_xsim__DOT__expected_offset__6__id = 0;
    IData/*31:0*/ __Vfunc_tb_d5_mobilenet_05_xsim__DOT__expected_offset__6____VlefCall_0__weight_bytes;
    __Vfunc_tb_d5_mobilenet_05_xsim__DOT__expected_offset__6____VlefCall_0__weight_bytes = 0;
    IData/*31:0*/ __Vfunc_tb_d5_mobilenet_05_xsim__DOT__expected_offset__6__i;
    __Vfunc_tb_d5_mobilenet_05_xsim__DOT__expected_offset__6__i = 0;
    IData/*31:0*/ __Vfunc_tb_d5_mobilenet_05_xsim__DOT__weight_bytes__7__Vfuncout;
    __Vfunc_tb_d5_mobilenet_05_xsim__DOT__weight_bytes__7__Vfuncout = 0;
    IData/*31:0*/ __Vfunc_tb_d5_mobilenet_05_xsim__DOT__weight_bytes__7__id;
    __Vfunc_tb_d5_mobilenet_05_xsim__DOT__weight_bytes__7__id = 0;
    IData/*31:0*/ __Vfunc_tb_d5_mobilenet_05_xsim__DOT__expected_offset__8__Vfuncout;
    __Vfunc_tb_d5_mobilenet_05_xsim__DOT__expected_offset__8__Vfuncout = 0;
    IData/*31:0*/ __Vfunc_tb_d5_mobilenet_05_xsim__DOT__expected_offset__8__id;
    __Vfunc_tb_d5_mobilenet_05_xsim__DOT__expected_offset__8__id = 0;
    IData/*31:0*/ __Vfunc_tb_d5_mobilenet_05_xsim__DOT__expected_offset__8____VlefCall_0__weight_bytes;
    __Vfunc_tb_d5_mobilenet_05_xsim__DOT__expected_offset__8____VlefCall_0__weight_bytes = 0;
    IData/*31:0*/ __Vfunc_tb_d5_mobilenet_05_xsim__DOT__expected_offset__8__i;
    __Vfunc_tb_d5_mobilenet_05_xsim__DOT__expected_offset__8__i = 0;
    IData/*31:0*/ __Vfunc_tb_d5_mobilenet_05_xsim__DOT__weight_bytes__9__Vfuncout;
    __Vfunc_tb_d5_mobilenet_05_xsim__DOT__weight_bytes__9__Vfuncout = 0;
    IData/*31:0*/ __Vfunc_tb_d5_mobilenet_05_xsim__DOT__weight_bytes__9__id;
    __Vfunc_tb_d5_mobilenet_05_xsim__DOT__weight_bytes__9__id = 0;
    // Body
    tb_d5_mobilenet_05_xsim__DOT__unnamedblk1_1__DOT____Vrepeat0 = 3U;
    while (VL_LTS_III(32, 0U, tb_d5_mobilenet_05_xsim__DOT__unnamedblk1_1__DOT____Vrepeat0)) {
        Vtb_d5_mobilenet_05_xsim___024root____VbeforeTrig_hc3011dec__0(vlSelf, 
                                                                       "@(posedge tb_d5_mobilenet_05_xsim.clk)");
        co_await vlSelfRef.__VtrigSched_hc3011dec__0.trigger(0U, 
                                                             nullptr, 
                                                             "@(posedge tb_d5_mobilenet_05_xsim.clk)", 
                                                             "rtl/sim/tb_d5_mobilenet_05_xsim.sv", 
                                                             104);
        tb_d5_mobilenet_05_xsim__DOT__unnamedblk1_1__DOT____Vrepeat0 
            = (tb_d5_mobilenet_05_xsim__DOT__unnamedblk1_1__DOT____Vrepeat0 
               - (IData)(1U));
    }
    vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__rst_n = 1U;
    Vtb_d5_mobilenet_05_xsim___024root____VbeforeTrig_hc3011a3e__0(vlSelf, 
                                                                   "@(negedge tb_d5_mobilenet_05_xsim.clk)");
    co_await vlSelfRef.__VtrigSched_hc3011a3e__0.trigger(0U, 
                                                         nullptr, 
                                                         "@(negedge tb_d5_mobilenet_05_xsim.clk)", 
                                                         "rtl/sim/tb_d5_mobilenet_05_xsim.sv", 
                                                         108);
    vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__start = 1U;
    Vtb_d5_mobilenet_05_xsim___024root____VbeforeTrig_hc3011dec__0(vlSelf, 
                                                                   "@(posedge tb_d5_mobilenet_05_xsim.clk)");
    co_await vlSelfRef.__VtrigSched_hc3011dec__0.trigger(0U, 
                                                         nullptr, 
                                                         "@(posedge tb_d5_mobilenet_05_xsim.clk)", 
                                                         "rtl/sim/tb_d5_mobilenet_05_xsim.sv", 
                                                         110);
    co_await vlSelfRef.__VdlySched.delay(0x00000000000003e8ULL, 
                                         nullptr, "rtl/sim/tb_d5_mobilenet_05_xsim.sv", 
                                         110);
    vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__start = 0U;
    if (VL_UNLIKELY(((1U & (~ (IData)(vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__busy)))))) {
        VL_WRITEF_NX("[%0t] %%Fatal: tb_d5_mobilenet_05_xsim.sv:112: Assertion failed in %m: sequencer did not enter busy state\n",3, 'M',vlSymsp->name(),"tb_d5_mobilenet_05_xsim", 'T',-9
                     , '#',64,VL_TIME_UNITED_Q(1000));
        VL_STOP_MT("rtl/sim/tb_d5_mobilenet_05_xsim.sv", 112, "", false);
    }
    __Vtask_tb_d5_mobilenet_05_xsim__DOT__check_descriptor__0__expected_id = 0U;
    if (VL_UNLIKELY(((1U & (~ (IData)(vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__layer_start)))))) {
        VL_WRITEF_NX("[%0t] %%Fatal: tb_d5_mobilenet_05_xsim.sv:71: Assertion failed in %m: descriptor %0d has no layer_start pulse\n",4, 'M',vlSymsp->name(),"tb_d5_mobilenet_05_xsim.check_descriptor", 'T',-9
                     , '#',64,VL_TIME_UNITED_Q(1000)
                     , '~',32,__Vtask_tb_d5_mobilenet_05_xsim__DOT__check_descriptor__0__expected_id);
        VL_STOP_MT("rtl/sim/tb_d5_mobilenet_05_xsim.sv", 71, "", false);
    }
    if (VL_UNLIKELY((((IData)(vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__layer_id) 
                      != (0x0000001fU & __Vtask_tb_d5_mobilenet_05_xsim__DOT__check_descriptor__0__expected_id))))) {
        VL_WRITEF_NX("[%0t] %%Fatal: tb_d5_mobilenet_05_xsim.sv:73: Assertion failed in %m: expected descriptor %0d, got %0d\n",5, 'M',vlSymsp->name(),"tb_d5_mobilenet_05_xsim.check_descriptor", 'T',-9
                     , '#',64,VL_TIME_UNITED_Q(1000)
                     , '~',32,__Vtask_tb_d5_mobilenet_05_xsim__DOT__check_descriptor__0__expected_id
                     , '#',5,(IData)(vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__layer_id));
        VL_STOP_MT("rtl/sim/tb_d5_mobilenet_05_xsim.sv", 73, "", false);
    }
    __Vfunc_tb_d5_mobilenet_05_xsim__DOT__expected_offset__1__id 
        = __Vtask_tb_d5_mobilenet_05_xsim__DOT__check_descriptor__0__expected_id;
    __Vfunc_tb_d5_mobilenet_05_xsim__DOT__expected_offset__1__Vfuncout = 0;
    __Vfunc_tb_d5_mobilenet_05_xsim__DOT__expected_offset__1__i = 0;
    __Vfunc_tb_d5_mobilenet_05_xsim__DOT__expected_offset__1__Vfuncout = 0U;
    __Vfunc_tb_d5_mobilenet_05_xsim__DOT__expected_offset__1__i = 0U;
    while (VL_LTS_III(32, __Vfunc_tb_d5_mobilenet_05_xsim__DOT__expected_offset__1__i, __Vfunc_tb_d5_mobilenet_05_xsim__DOT__expected_offset__1__id)) {
        __Vfunc_tb_d5_mobilenet_05_xsim__DOT__weight_bytes__2__id 
            = __Vfunc_tb_d5_mobilenet_05_xsim__DOT__expected_offset__1__i;
        __Vfunc_tb_d5_mobilenet_05_xsim__DOT__weight_bytes__2__Vfuncout = 0;
        __Vfunc_tb_d5_mobilenet_05_xsim__DOT__weight_bytes__2__Vfuncout 
            = (((((((((0U == __Vfunc_tb_d5_mobilenet_05_xsim__DOT__weight_bytes__2__id) 
                      | (1U == __Vfunc_tb_d5_mobilenet_05_xsim__DOT__weight_bytes__2__id)) 
                     | (2U == __Vfunc_tb_d5_mobilenet_05_xsim__DOT__weight_bytes__2__id)) 
                    | (3U == __Vfunc_tb_d5_mobilenet_05_xsim__DOT__weight_bytes__2__id)) 
                   | (4U == __Vfunc_tb_d5_mobilenet_05_xsim__DOT__weight_bytes__2__id)) 
                  | (5U == __Vfunc_tb_d5_mobilenet_05_xsim__DOT__weight_bytes__2__id)) 
                 | (6U == __Vfunc_tb_d5_mobilenet_05_xsim__DOT__weight_bytes__2__id)) 
                | (7U == __Vfunc_tb_d5_mobilenet_05_xsim__DOT__weight_bytes__2__id))
                ? ((0U == __Vfunc_tb_d5_mobilenet_05_xsim__DOT__weight_bytes__2__id)
                    ? 0x000001b0U : ((1U == __Vfunc_tb_d5_mobilenet_05_xsim__DOT__weight_bytes__2__id)
                                      ? 0x00000090U
                                      : ((2U == __Vfunc_tb_d5_mobilenet_05_xsim__DOT__weight_bytes__2__id)
                                          ? 0x00000200U
                                          : ((3U == __Vfunc_tb_d5_mobilenet_05_xsim__DOT__weight_bytes__2__id)
                                              ? 0x00000120U
                                              : ((4U 
                                                  == __Vfunc_tb_d5_mobilenet_05_xsim__DOT__weight_bytes__2__id)
                                                  ? 0x00000800U
                                                  : 
                                                 ((5U 
                                                   == __Vfunc_tb_d5_mobilenet_05_xsim__DOT__weight_bytes__2__id)
                                                   ? 0x00000240U
                                                   : 
                                                  ((6U 
                                                    == __Vfunc_tb_d5_mobilenet_05_xsim__DOT__weight_bytes__2__id)
                                                    ? 0x00001000U
                                                    : 0x00000240U)))))))
                : (((((((((8U == __Vfunc_tb_d5_mobilenet_05_xsim__DOT__weight_bytes__2__id) 
                          | (9U == __Vfunc_tb_d5_mobilenet_05_xsim__DOT__weight_bytes__2__id)) 
                         | (0x0000000aU == __Vfunc_tb_d5_mobilenet_05_xsim__DOT__weight_bytes__2__id)) 
                        | (0x0000000bU == __Vfunc_tb_d5_mobilenet_05_xsim__DOT__weight_bytes__2__id)) 
                       | (0x0000000cU == __Vfunc_tb_d5_mobilenet_05_xsim__DOT__weight_bytes__2__id)) 
                      | (0x0000000dU == __Vfunc_tb_d5_mobilenet_05_xsim__DOT__weight_bytes__2__id)) 
                     | (0x0000000eU == __Vfunc_tb_d5_mobilenet_05_xsim__DOT__weight_bytes__2__id)) 
                    | (0x0000000fU == __Vfunc_tb_d5_mobilenet_05_xsim__DOT__weight_bytes__2__id))
                    ? ((8U == __Vfunc_tb_d5_mobilenet_05_xsim__DOT__weight_bytes__2__id)
                        ? 0x00002000U : ((9U == __Vfunc_tb_d5_mobilenet_05_xsim__DOT__weight_bytes__2__id)
                                          ? 0x00000480U
                                          : ((0x0000000aU 
                                              == __Vfunc_tb_d5_mobilenet_05_xsim__DOT__weight_bytes__2__id)
                                              ? 0x00004000U
                                              : ((0x0000000bU 
                                                  == __Vfunc_tb_d5_mobilenet_05_xsim__DOT__weight_bytes__2__id)
                                                  ? 0x00000480U
                                                  : 
                                                 ((0x0000000cU 
                                                   == __Vfunc_tb_d5_mobilenet_05_xsim__DOT__weight_bytes__2__id)
                                                   ? 0x00004000U
                                                   : 
                                                  ((0x0000000dU 
                                                    == __Vfunc_tb_d5_mobilenet_05_xsim__DOT__weight_bytes__2__id)
                                                    ? 0x00000900U
                                                    : 
                                                   ((0x0000000eU 
                                                     == __Vfunc_tb_d5_mobilenet_05_xsim__DOT__weight_bytes__2__id)
                                                     ? 0x00008000U
                                                     : 0x00000900U)))))))
                    : ((0x00000010U == __Vfunc_tb_d5_mobilenet_05_xsim__DOT__weight_bytes__2__id)
                        ? 0x00010000U : ((0x00000011U 
                                          == __Vfunc_tb_d5_mobilenet_05_xsim__DOT__weight_bytes__2__id)
                                          ? 0x00000900U
                                          : ((0x00000012U 
                                              == __Vfunc_tb_d5_mobilenet_05_xsim__DOT__weight_bytes__2__id)
                                              ? 0x00020000U
                                              : ((0x00000013U 
                                                  == __Vfunc_tb_d5_mobilenet_05_xsim__DOT__weight_bytes__2__id)
                                                  ? 0x00001200U
                                                  : 
                                                 ((0x00000014U 
                                                   == __Vfunc_tb_d5_mobilenet_05_xsim__DOT__weight_bytes__2__id)
                                                   ? 0x00040000U
                                                   : 
                                                  ((0x00000016U 
                                                    == __Vfunc_tb_d5_mobilenet_05_xsim__DOT__weight_bytes__2__id)
                                                    ? 0x00001400U
                                                    : 0U))))))));
        __Vfunc_tb_d5_mobilenet_05_xsim__DOT__expected_offset__1____VlefCall_0__weight_bytes 
            = __Vfunc_tb_d5_mobilenet_05_xsim__DOT__weight_bytes__2__Vfuncout;
        __Vfunc_tb_d5_mobilenet_05_xsim__DOT__expected_offset__1__Vfuncout 
            = (__Vfunc_tb_d5_mobilenet_05_xsim__DOT__expected_offset__1__Vfuncout 
               + __Vfunc_tb_d5_mobilenet_05_xsim__DOT__expected_offset__1____VlefCall_0__weight_bytes);
        __Vfunc_tb_d5_mobilenet_05_xsim__DOT__expected_offset__1__i 
            = ((IData)(1U) + __Vfunc_tb_d5_mobilenet_05_xsim__DOT__expected_offset__1__i);
    }
    __Vtask_tb_d5_mobilenet_05_xsim__DOT__check_descriptor__0____VlefCall_0__expected_offset 
        = __Vfunc_tb_d5_mobilenet_05_xsim__DOT__expected_offset__1__Vfuncout;
    if (VL_UNLIKELY(((vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__weight_offset 
                      != __Vtask_tb_d5_mobilenet_05_xsim__DOT__check_descriptor__0____VlefCall_0__expected_offset)))) {
        __Vfunc_tb_d5_mobilenet_05_xsim__DOT__expected_offset__3__id 
            = __Vtask_tb_d5_mobilenet_05_xsim__DOT__check_descriptor__0__expected_id;
        __Vfunc_tb_d5_mobilenet_05_xsim__DOT__expected_offset__3__Vfuncout = 0;
        __Vfunc_tb_d5_mobilenet_05_xsim__DOT__expected_offset__3__i = 0;
        __Vfunc_tb_d5_mobilenet_05_xsim__DOT__expected_offset__3__Vfuncout = 0U;
        __Vfunc_tb_d5_mobilenet_05_xsim__DOT__expected_offset__3__i = 0U;
        while (VL_LTS_III(32, __Vfunc_tb_d5_mobilenet_05_xsim__DOT__expected_offset__3__i, __Vfunc_tb_d5_mobilenet_05_xsim__DOT__expected_offset__3__id)) {
            __Vfunc_tb_d5_mobilenet_05_xsim__DOT__weight_bytes__4__id 
                = __Vfunc_tb_d5_mobilenet_05_xsim__DOT__expected_offset__3__i;
            __Vfunc_tb_d5_mobilenet_05_xsim__DOT__weight_bytes__4__Vfuncout = 0;
            __Vfunc_tb_d5_mobilenet_05_xsim__DOT__weight_bytes__4__Vfuncout 
                = (((((((((0U == __Vfunc_tb_d5_mobilenet_05_xsim__DOT__weight_bytes__4__id) 
                          | (1U == __Vfunc_tb_d5_mobilenet_05_xsim__DOT__weight_bytes__4__id)) 
                         | (2U == __Vfunc_tb_d5_mobilenet_05_xsim__DOT__weight_bytes__4__id)) 
                        | (3U == __Vfunc_tb_d5_mobilenet_05_xsim__DOT__weight_bytes__4__id)) 
                       | (4U == __Vfunc_tb_d5_mobilenet_05_xsim__DOT__weight_bytes__4__id)) 
                      | (5U == __Vfunc_tb_d5_mobilenet_05_xsim__DOT__weight_bytes__4__id)) 
                     | (6U == __Vfunc_tb_d5_mobilenet_05_xsim__DOT__weight_bytes__4__id)) 
                    | (7U == __Vfunc_tb_d5_mobilenet_05_xsim__DOT__weight_bytes__4__id))
                    ? ((0U == __Vfunc_tb_d5_mobilenet_05_xsim__DOT__weight_bytes__4__id)
                        ? 0x000001b0U : ((1U == __Vfunc_tb_d5_mobilenet_05_xsim__DOT__weight_bytes__4__id)
                                          ? 0x00000090U
                                          : ((2U == __Vfunc_tb_d5_mobilenet_05_xsim__DOT__weight_bytes__4__id)
                                              ? 0x00000200U
                                              : ((3U 
                                                  == __Vfunc_tb_d5_mobilenet_05_xsim__DOT__weight_bytes__4__id)
                                                  ? 0x00000120U
                                                  : 
                                                 ((4U 
                                                   == __Vfunc_tb_d5_mobilenet_05_xsim__DOT__weight_bytes__4__id)
                                                   ? 0x00000800U
                                                   : 
                                                  ((5U 
                                                    == __Vfunc_tb_d5_mobilenet_05_xsim__DOT__weight_bytes__4__id)
                                                    ? 0x00000240U
                                                    : 
                                                   ((6U 
                                                     == __Vfunc_tb_d5_mobilenet_05_xsim__DOT__weight_bytes__4__id)
                                                     ? 0x00001000U
                                                     : 0x00000240U)))))))
                    : (((((((((8U == __Vfunc_tb_d5_mobilenet_05_xsim__DOT__weight_bytes__4__id) 
                              | (9U == __Vfunc_tb_d5_mobilenet_05_xsim__DOT__weight_bytes__4__id)) 
                             | (0x0000000aU == __Vfunc_tb_d5_mobilenet_05_xsim__DOT__weight_bytes__4__id)) 
                            | (0x0000000bU == __Vfunc_tb_d5_mobilenet_05_xsim__DOT__weight_bytes__4__id)) 
                           | (0x0000000cU == __Vfunc_tb_d5_mobilenet_05_xsim__DOT__weight_bytes__4__id)) 
                          | (0x0000000dU == __Vfunc_tb_d5_mobilenet_05_xsim__DOT__weight_bytes__4__id)) 
                         | (0x0000000eU == __Vfunc_tb_d5_mobilenet_05_xsim__DOT__weight_bytes__4__id)) 
                        | (0x0000000fU == __Vfunc_tb_d5_mobilenet_05_xsim__DOT__weight_bytes__4__id))
                        ? ((8U == __Vfunc_tb_d5_mobilenet_05_xsim__DOT__weight_bytes__4__id)
                            ? 0x00002000U : ((9U == __Vfunc_tb_d5_mobilenet_05_xsim__DOT__weight_bytes__4__id)
                                              ? 0x00000480U
                                              : ((0x0000000aU 
                                                  == __Vfunc_tb_d5_mobilenet_05_xsim__DOT__weight_bytes__4__id)
                                                  ? 0x00004000U
                                                  : 
                                                 ((0x0000000bU 
                                                   == __Vfunc_tb_d5_mobilenet_05_xsim__DOT__weight_bytes__4__id)
                                                   ? 0x00000480U
                                                   : 
                                                  ((0x0000000cU 
                                                    == __Vfunc_tb_d5_mobilenet_05_xsim__DOT__weight_bytes__4__id)
                                                    ? 0x00004000U
                                                    : 
                                                   ((0x0000000dU 
                                                     == __Vfunc_tb_d5_mobilenet_05_xsim__DOT__weight_bytes__4__id)
                                                     ? 0x00000900U
                                                     : 
                                                    ((0x0000000eU 
                                                      == __Vfunc_tb_d5_mobilenet_05_xsim__DOT__weight_bytes__4__id)
                                                      ? 0x00008000U
                                                      : 0x00000900U)))))))
                        : ((0x00000010U == __Vfunc_tb_d5_mobilenet_05_xsim__DOT__weight_bytes__4__id)
                            ? 0x00010000U : ((0x00000011U 
                                              == __Vfunc_tb_d5_mobilenet_05_xsim__DOT__weight_bytes__4__id)
                                              ? 0x00000900U
                                              : ((0x00000012U 
                                                  == __Vfunc_tb_d5_mobilenet_05_xsim__DOT__weight_bytes__4__id)
                                                  ? 0x00020000U
                                                  : 
                                                 ((0x00000013U 
                                                   == __Vfunc_tb_d5_mobilenet_05_xsim__DOT__weight_bytes__4__id)
                                                   ? 0x00001200U
                                                   : 
                                                  ((0x00000014U 
                                                    == __Vfunc_tb_d5_mobilenet_05_xsim__DOT__weight_bytes__4__id)
                                                    ? 0x00040000U
                                                    : 
                                                   ((0x00000016U 
                                                     == __Vfunc_tb_d5_mobilenet_05_xsim__DOT__weight_bytes__4__id)
                                                     ? 0x00001400U
                                                     : 0U))))))));
            __Vfunc_tb_d5_mobilenet_05_xsim__DOT__expected_offset__3____VlefCall_0__weight_bytes 
                = __Vfunc_tb_d5_mobilenet_05_xsim__DOT__weight_bytes__4__Vfuncout;
            __Vfunc_tb_d5_mobilenet_05_xsim__DOT__expected_offset__3__Vfuncout 
                = (__Vfunc_tb_d5_mobilenet_05_xsim__DOT__expected_offset__3__Vfuncout 
                   + __Vfunc_tb_d5_mobilenet_05_xsim__DOT__expected_offset__3____VlefCall_0__weight_bytes);
            __Vfunc_tb_d5_mobilenet_05_xsim__DOT__expected_offset__3__i 
                = ((IData)(1U) + __Vfunc_tb_d5_mobilenet_05_xsim__DOT__expected_offset__3__i);
        }
        __Vtask_tb_d5_mobilenet_05_xsim__DOT__check_descriptor__0____VlefCall_1__expected_offset 
            = __Vfunc_tb_d5_mobilenet_05_xsim__DOT__expected_offset__3__Vfuncout;
        VL_WRITEF_NX("[%0t] %%Fatal: tb_d5_mobilenet_05_xsim.sv:75: Assertion failed in %m: descriptor %0d offset %0d, expected %0d\n",6, 'M',vlSymsp->name(),"tb_d5_mobilenet_05_xsim.check_descriptor", 'T',-9
                     , '#',64,VL_TIME_UNITED_Q(1000)
                     , '~',32,__Vtask_tb_d5_mobilenet_05_xsim__DOT__check_descriptor__0__expected_id
                     , '#',32,vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__weight_offset
                     , '#',32,__Vtask_tb_d5_mobilenet_05_xsim__DOT__check_descriptor__0____VlefCall_1__expected_offset);
        VL_STOP_MT("rtl/sim/tb_d5_mobilenet_05_xsim.sv", 75, "", false);
    }
    if ((0U == __Vtask_tb_d5_mobilenet_05_xsim__DOT__check_descriptor__0__expected_id)) {
        if (VL_UNLIKELY((((((((0U != (IData)(vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__op_kind)) 
                              | (3U != (IData)(vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__in_channels))) 
                             | (0x0010U != (IData)(vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__out_channels))) 
                            | (0x20U != (IData)(vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__height))) 
                           | (0x20U != (IData)(vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__width))) 
                          | (1U != (IData)(vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__stride)))))) {
            VL_WRITEF_NX("[%0t] %%Fatal: tb_d5_mobilenet_05_xsim.sv:83: Assertion failed in %m: invalid stem descriptor\n",3, 'M',vlSymsp->name(),"tb_d5_mobilenet_05_xsim.check_descriptor", 'T',-9
                         , '#',64,VL_TIME_UNITED_Q(1000));
            VL_STOP_MT("rtl/sim/tb_d5_mobilenet_05_xsim.sv", 83, "", false);
        }
    } else if ((1U == __Vtask_tb_d5_mobilenet_05_xsim__DOT__check_descriptor__0__expected_id)) {
        if (VL_UNLIKELY(((((1U != (IData)(vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__op_kind)) 
                           | (0x0010U != (IData)(vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__in_channels))) 
                          | (0x0010U != (IData)(vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__out_channels)))))) {
            VL_WRITEF_NX("[%0t] %%Fatal: tb_d5_mobilenet_05_xsim.sv:85: Assertion failed in %m: invalid first depthwise descriptor\n",3, 'M',vlSymsp->name(),"tb_d5_mobilenet_05_xsim.check_descriptor", 'T',-9
                         , '#',64,VL_TIME_UNITED_Q(1000));
            VL_STOP_MT("rtl/sim/tb_d5_mobilenet_05_xsim.sv", 85, "", false);
        }
    } else if ((2U == __Vtask_tb_d5_mobilenet_05_xsim__DOT__check_descriptor__0__expected_id)) {
        if (VL_UNLIKELY(((((2U != (IData)(vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__op_kind)) 
                           | (0x0010U != (IData)(vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__in_channels))) 
                          | (0x0020U != (IData)(vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__out_channels)))))) {
            VL_WRITEF_NX("[%0t] %%Fatal: tb_d5_mobilenet_05_xsim.sv:87: Assertion failed in %m: invalid first pointwise descriptor\n",3, 'M',vlSymsp->name(),"tb_d5_mobilenet_05_xsim.check_descriptor", 'T',-9
                         , '#',64,VL_TIME_UNITED_Q(1000));
            VL_STOP_MT("rtl/sim/tb_d5_mobilenet_05_xsim.sv", 87, "", false);
        }
    } else if (((0x0000000dU == __Vtask_tb_d5_mobilenet_05_xsim__DOT__check_descriptor__0__expected_id) 
                || (0x0000000fU == __Vtask_tb_d5_mobilenet_05_xsim__DOT__check_descriptor__0__expected_id))) {
        if (VL_UNLIKELY(((((1U != (IData)(vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__op_kind)) 
                           | (0x0100U != (IData)(vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__in_channels))) 
                          | (0x0100U != (IData)(vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__out_channels)))))) {
            VL_WRITEF_NX("[%0t] %%Fatal: tb_d5_mobilenet_05_xsim.sv:89: Assertion failed in %m: invalid 256-channel depthwise descriptor\n",3, 'M',vlSymsp->name(),"tb_d5_mobilenet_05_xsim.check_descriptor", 'T',-9
                         , '#',64,VL_TIME_UNITED_Q(1000));
            VL_STOP_MT("rtl/sim/tb_d5_mobilenet_05_xsim.sv", 89, "", false);
        }
    } else if (((0x0000000eU == __Vtask_tb_d5_mobilenet_05_xsim__DOT__check_descriptor__0__expected_id) 
                || (0x00000010U == __Vtask_tb_d5_mobilenet_05_xsim__DOT__check_descriptor__0__expected_id))) {
        if (VL_UNLIKELY(((((2U != (IData)(vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__op_kind)) 
                           | (0x0100U != (IData)(vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__in_channels))) 
                          | (0x0100U != (IData)(vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__out_channels)))))) {
            VL_WRITEF_NX("[%0t] %%Fatal: tb_d5_mobilenet_05_xsim.sv:91: Assertion failed in %m: invalid retained 256-to-256 pointwise descriptor\n",3, 'M',vlSymsp->name(),"tb_d5_mobilenet_05_xsim.check_descriptor", 'T',-9
                         , '#',64,VL_TIME_UNITED_Q(1000));
            VL_STOP_MT("rtl/sim/tb_d5_mobilenet_05_xsim.sv", 91, "", false);
        }
    } else if ((0x00000015U == __Vtask_tb_d5_mobilenet_05_xsim__DOT__check_descriptor__0__expected_id)) {
        if (VL_UNLIKELY(((((((3U != (IData)(vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__op_kind)) 
                             | (0x0200U != (IData)(vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__in_channels))) 
                            | (0x0200U != (IData)(vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__out_channels))) 
                           | (2U != (IData)(vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__height))) 
                          | (2U != (IData)(vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__width)))))) {
            VL_WRITEF_NX("[%0t] %%Fatal: tb_d5_mobilenet_05_xsim.sv:94: Assertion failed in %m: invalid GAP descriptor\n",3, 'M',vlSymsp->name(),"tb_d5_mobilenet_05_xsim.check_descriptor", 'T',-9
                         , '#',64,VL_TIME_UNITED_Q(1000));
            VL_STOP_MT("rtl/sim/tb_d5_mobilenet_05_xsim.sv", 94, "", false);
        }
    } else if ((0x00000016U == __Vtask_tb_d5_mobilenet_05_xsim__DOT__check_descriptor__0__expected_id)) {
        if (VL_UNLIKELY(((((((4U != (IData)(vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__op_kind)) 
                             | (0x0200U != (IData)(vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__in_channels))) 
                            | (0x000aU != (IData)(vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__out_channels))) 
                           | (1U != (IData)(vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__height))) 
                          | (1U != (IData)(vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__width)))))) {
            VL_WRITEF_NX("[%0t] %%Fatal: tb_d5_mobilenet_05_xsim.sv:97: Assertion failed in %m: invalid FC descriptor\n",3, 'M',vlSymsp->name(),"tb_d5_mobilenet_05_xsim.check_descriptor", 'T',-9
                         , '#',64,VL_TIME_UNITED_Q(1000));
            VL_STOP_MT("rtl/sim/tb_d5_mobilenet_05_xsim.sv", 97, "", false);
        }
    }
    tb_d5_mobilenet_05_xsim__DOT__id = 0U;
    while (VL_GTS_III(32, 0x00000016U, tb_d5_mobilenet_05_xsim__DOT__id)) {
        Vtb_d5_mobilenet_05_xsim___024root____VbeforeTrig_hc3011a3e__0(vlSelf, 
                                                                       "@(negedge tb_d5_mobilenet_05_xsim.clk)");
        co_await vlSelfRef.__VtrigSched_hc3011a3e__0.trigger(0U, 
                                                             nullptr, 
                                                             "@(negedge tb_d5_mobilenet_05_xsim.clk)", 
                                                             "rtl/sim/tb_d5_mobilenet_05_xsim.sv", 
                                                             119);
        vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__layer_done = 1U;
        Vtb_d5_mobilenet_05_xsim___024root____VbeforeTrig_hc3011dec__0(vlSelf, 
                                                                       "@(posedge tb_d5_mobilenet_05_xsim.clk)");
        co_await vlSelfRef.__VtrigSched_hc3011dec__0.trigger(0U, 
                                                             nullptr, 
                                                             "@(posedge tb_d5_mobilenet_05_xsim.clk)", 
                                                             "rtl/sim/tb_d5_mobilenet_05_xsim.sv", 
                                                             121);
        co_await vlSelfRef.__VdlySched.delay(0x00000000000003e8ULL, 
                                             nullptr, 
                                             "rtl/sim/tb_d5_mobilenet_05_xsim.sv", 
                                             121);
        vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__layer_done = 0U;
        __Vtask_tb_d5_mobilenet_05_xsim__DOT__check_descriptor__5__expected_id 
            = ((IData)(1U) + tb_d5_mobilenet_05_xsim__DOT__id);
        if (VL_UNLIKELY(((1U & (~ (IData)(vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__layer_start)))))) {
            VL_WRITEF_NX("[%0t] %%Fatal: tb_d5_mobilenet_05_xsim.sv:71: Assertion failed in %m: descriptor %0d has no layer_start pulse\n",4, 'M',vlSymsp->name(),"tb_d5_mobilenet_05_xsim.check_descriptor", 'T',-9
                         , '#',64,VL_TIME_UNITED_Q(1000)
                         , '~',32,__Vtask_tb_d5_mobilenet_05_xsim__DOT__check_descriptor__5__expected_id);
            VL_STOP_MT("rtl/sim/tb_d5_mobilenet_05_xsim.sv", 71, "", false);
        }
        if (VL_UNLIKELY((((IData)(vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__layer_id) 
                          != (0x0000001fU & __Vtask_tb_d5_mobilenet_05_xsim__DOT__check_descriptor__5__expected_id))))) {
            VL_WRITEF_NX("[%0t] %%Fatal: tb_d5_mobilenet_05_xsim.sv:73: Assertion failed in %m: expected descriptor %0d, got %0d\n",5, 'M',vlSymsp->name(),"tb_d5_mobilenet_05_xsim.check_descriptor", 'T',-9
                         , '#',64,VL_TIME_UNITED_Q(1000)
                         , '~',32,__Vtask_tb_d5_mobilenet_05_xsim__DOT__check_descriptor__5__expected_id
                         , '#',5,(IData)(vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__layer_id));
            VL_STOP_MT("rtl/sim/tb_d5_mobilenet_05_xsim.sv", 73, "", false);
        }
        __Vfunc_tb_d5_mobilenet_05_xsim__DOT__expected_offset__6__id 
            = __Vtask_tb_d5_mobilenet_05_xsim__DOT__check_descriptor__5__expected_id;
        __Vfunc_tb_d5_mobilenet_05_xsim__DOT__expected_offset__6__Vfuncout = 0;
        __Vfunc_tb_d5_mobilenet_05_xsim__DOT__expected_offset__6__i = 0;
        __Vfunc_tb_d5_mobilenet_05_xsim__DOT__expected_offset__6__Vfuncout = 0U;
        __Vfunc_tb_d5_mobilenet_05_xsim__DOT__expected_offset__6__i = 0U;
        while (VL_LTS_III(32, __Vfunc_tb_d5_mobilenet_05_xsim__DOT__expected_offset__6__i, __Vfunc_tb_d5_mobilenet_05_xsim__DOT__expected_offset__6__id)) {
            __Vfunc_tb_d5_mobilenet_05_xsim__DOT__weight_bytes__7__id 
                = __Vfunc_tb_d5_mobilenet_05_xsim__DOT__expected_offset__6__i;
            __Vfunc_tb_d5_mobilenet_05_xsim__DOT__weight_bytes__7__Vfuncout = 0;
            __Vfunc_tb_d5_mobilenet_05_xsim__DOT__weight_bytes__7__Vfuncout 
                = (((((((((0U == __Vfunc_tb_d5_mobilenet_05_xsim__DOT__weight_bytes__7__id) 
                          | (1U == __Vfunc_tb_d5_mobilenet_05_xsim__DOT__weight_bytes__7__id)) 
                         | (2U == __Vfunc_tb_d5_mobilenet_05_xsim__DOT__weight_bytes__7__id)) 
                        | (3U == __Vfunc_tb_d5_mobilenet_05_xsim__DOT__weight_bytes__7__id)) 
                       | (4U == __Vfunc_tb_d5_mobilenet_05_xsim__DOT__weight_bytes__7__id)) 
                      | (5U == __Vfunc_tb_d5_mobilenet_05_xsim__DOT__weight_bytes__7__id)) 
                     | (6U == __Vfunc_tb_d5_mobilenet_05_xsim__DOT__weight_bytes__7__id)) 
                    | (7U == __Vfunc_tb_d5_mobilenet_05_xsim__DOT__weight_bytes__7__id))
                    ? ((0U == __Vfunc_tb_d5_mobilenet_05_xsim__DOT__weight_bytes__7__id)
                        ? 0x000001b0U : ((1U == __Vfunc_tb_d5_mobilenet_05_xsim__DOT__weight_bytes__7__id)
                                          ? 0x00000090U
                                          : ((2U == __Vfunc_tb_d5_mobilenet_05_xsim__DOT__weight_bytes__7__id)
                                              ? 0x00000200U
                                              : ((3U 
                                                  == __Vfunc_tb_d5_mobilenet_05_xsim__DOT__weight_bytes__7__id)
                                                  ? 0x00000120U
                                                  : 
                                                 ((4U 
                                                   == __Vfunc_tb_d5_mobilenet_05_xsim__DOT__weight_bytes__7__id)
                                                   ? 0x00000800U
                                                   : 
                                                  ((5U 
                                                    == __Vfunc_tb_d5_mobilenet_05_xsim__DOT__weight_bytes__7__id)
                                                    ? 0x00000240U
                                                    : 
                                                   ((6U 
                                                     == __Vfunc_tb_d5_mobilenet_05_xsim__DOT__weight_bytes__7__id)
                                                     ? 0x00001000U
                                                     : 0x00000240U)))))))
                    : (((((((((8U == __Vfunc_tb_d5_mobilenet_05_xsim__DOT__weight_bytes__7__id) 
                              | (9U == __Vfunc_tb_d5_mobilenet_05_xsim__DOT__weight_bytes__7__id)) 
                             | (0x0000000aU == __Vfunc_tb_d5_mobilenet_05_xsim__DOT__weight_bytes__7__id)) 
                            | (0x0000000bU == __Vfunc_tb_d5_mobilenet_05_xsim__DOT__weight_bytes__7__id)) 
                           | (0x0000000cU == __Vfunc_tb_d5_mobilenet_05_xsim__DOT__weight_bytes__7__id)) 
                          | (0x0000000dU == __Vfunc_tb_d5_mobilenet_05_xsim__DOT__weight_bytes__7__id)) 
                         | (0x0000000eU == __Vfunc_tb_d5_mobilenet_05_xsim__DOT__weight_bytes__7__id)) 
                        | (0x0000000fU == __Vfunc_tb_d5_mobilenet_05_xsim__DOT__weight_bytes__7__id))
                        ? ((8U == __Vfunc_tb_d5_mobilenet_05_xsim__DOT__weight_bytes__7__id)
                            ? 0x00002000U : ((9U == __Vfunc_tb_d5_mobilenet_05_xsim__DOT__weight_bytes__7__id)
                                              ? 0x00000480U
                                              : ((0x0000000aU 
                                                  == __Vfunc_tb_d5_mobilenet_05_xsim__DOT__weight_bytes__7__id)
                                                  ? 0x00004000U
                                                  : 
                                                 ((0x0000000bU 
                                                   == __Vfunc_tb_d5_mobilenet_05_xsim__DOT__weight_bytes__7__id)
                                                   ? 0x00000480U
                                                   : 
                                                  ((0x0000000cU 
                                                    == __Vfunc_tb_d5_mobilenet_05_xsim__DOT__weight_bytes__7__id)
                                                    ? 0x00004000U
                                                    : 
                                                   ((0x0000000dU 
                                                     == __Vfunc_tb_d5_mobilenet_05_xsim__DOT__weight_bytes__7__id)
                                                     ? 0x00000900U
                                                     : 
                                                    ((0x0000000eU 
                                                      == __Vfunc_tb_d5_mobilenet_05_xsim__DOT__weight_bytes__7__id)
                                                      ? 0x00008000U
                                                      : 0x00000900U)))))))
                        : ((0x00000010U == __Vfunc_tb_d5_mobilenet_05_xsim__DOT__weight_bytes__7__id)
                            ? 0x00010000U : ((0x00000011U 
                                              == __Vfunc_tb_d5_mobilenet_05_xsim__DOT__weight_bytes__7__id)
                                              ? 0x00000900U
                                              : ((0x00000012U 
                                                  == __Vfunc_tb_d5_mobilenet_05_xsim__DOT__weight_bytes__7__id)
                                                  ? 0x00020000U
                                                  : 
                                                 ((0x00000013U 
                                                   == __Vfunc_tb_d5_mobilenet_05_xsim__DOT__weight_bytes__7__id)
                                                   ? 0x00001200U
                                                   : 
                                                  ((0x00000014U 
                                                    == __Vfunc_tb_d5_mobilenet_05_xsim__DOT__weight_bytes__7__id)
                                                    ? 0x00040000U
                                                    : 
                                                   ((0x00000016U 
                                                     == __Vfunc_tb_d5_mobilenet_05_xsim__DOT__weight_bytes__7__id)
                                                     ? 0x00001400U
                                                     : 0U))))))));
            __Vfunc_tb_d5_mobilenet_05_xsim__DOT__expected_offset__6____VlefCall_0__weight_bytes 
                = __Vfunc_tb_d5_mobilenet_05_xsim__DOT__weight_bytes__7__Vfuncout;
            __Vfunc_tb_d5_mobilenet_05_xsim__DOT__expected_offset__6__Vfuncout 
                = (__Vfunc_tb_d5_mobilenet_05_xsim__DOT__expected_offset__6__Vfuncout 
                   + __Vfunc_tb_d5_mobilenet_05_xsim__DOT__expected_offset__6____VlefCall_0__weight_bytes);
            __Vfunc_tb_d5_mobilenet_05_xsim__DOT__expected_offset__6__i 
                = ((IData)(1U) + __Vfunc_tb_d5_mobilenet_05_xsim__DOT__expected_offset__6__i);
        }
        __Vtask_tb_d5_mobilenet_05_xsim__DOT__check_descriptor__5____VlefCall_0__expected_offset 
            = __Vfunc_tb_d5_mobilenet_05_xsim__DOT__expected_offset__6__Vfuncout;
        if (VL_UNLIKELY(((vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__weight_offset 
                          != __Vtask_tb_d5_mobilenet_05_xsim__DOT__check_descriptor__5____VlefCall_0__expected_offset)))) {
            __Vfunc_tb_d5_mobilenet_05_xsim__DOT__expected_offset__8__id 
                = __Vtask_tb_d5_mobilenet_05_xsim__DOT__check_descriptor__5__expected_id;
            __Vfunc_tb_d5_mobilenet_05_xsim__DOT__expected_offset__8__Vfuncout = 0;
            __Vfunc_tb_d5_mobilenet_05_xsim__DOT__expected_offset__8__i = 0;
            __Vfunc_tb_d5_mobilenet_05_xsim__DOT__expected_offset__8__Vfuncout = 0U;
            __Vfunc_tb_d5_mobilenet_05_xsim__DOT__expected_offset__8__i = 0U;
            while (VL_LTS_III(32, __Vfunc_tb_d5_mobilenet_05_xsim__DOT__expected_offset__8__i, __Vfunc_tb_d5_mobilenet_05_xsim__DOT__expected_offset__8__id)) {
                __Vfunc_tb_d5_mobilenet_05_xsim__DOT__weight_bytes__9__id 
                    = __Vfunc_tb_d5_mobilenet_05_xsim__DOT__expected_offset__8__i;
                __Vfunc_tb_d5_mobilenet_05_xsim__DOT__weight_bytes__9__Vfuncout = 0;
                __Vfunc_tb_d5_mobilenet_05_xsim__DOT__weight_bytes__9__Vfuncout 
                    = (((((((((0U == __Vfunc_tb_d5_mobilenet_05_xsim__DOT__weight_bytes__9__id) 
                              | (1U == __Vfunc_tb_d5_mobilenet_05_xsim__DOT__weight_bytes__9__id)) 
                             | (2U == __Vfunc_tb_d5_mobilenet_05_xsim__DOT__weight_bytes__9__id)) 
                            | (3U == __Vfunc_tb_d5_mobilenet_05_xsim__DOT__weight_bytes__9__id)) 
                           | (4U == __Vfunc_tb_d5_mobilenet_05_xsim__DOT__weight_bytes__9__id)) 
                          | (5U == __Vfunc_tb_d5_mobilenet_05_xsim__DOT__weight_bytes__9__id)) 
                         | (6U == __Vfunc_tb_d5_mobilenet_05_xsim__DOT__weight_bytes__9__id)) 
                        | (7U == __Vfunc_tb_d5_mobilenet_05_xsim__DOT__weight_bytes__9__id))
                        ? ((0U == __Vfunc_tb_d5_mobilenet_05_xsim__DOT__weight_bytes__9__id)
                            ? 0x000001b0U : ((1U == __Vfunc_tb_d5_mobilenet_05_xsim__DOT__weight_bytes__9__id)
                                              ? 0x00000090U
                                              : ((2U 
                                                  == __Vfunc_tb_d5_mobilenet_05_xsim__DOT__weight_bytes__9__id)
                                                  ? 0x00000200U
                                                  : 
                                                 ((3U 
                                                   == __Vfunc_tb_d5_mobilenet_05_xsim__DOT__weight_bytes__9__id)
                                                   ? 0x00000120U
                                                   : 
                                                  ((4U 
                                                    == __Vfunc_tb_d5_mobilenet_05_xsim__DOT__weight_bytes__9__id)
                                                    ? 0x00000800U
                                                    : 
                                                   ((5U 
                                                     == __Vfunc_tb_d5_mobilenet_05_xsim__DOT__weight_bytes__9__id)
                                                     ? 0x00000240U
                                                     : 
                                                    ((6U 
                                                      == __Vfunc_tb_d5_mobilenet_05_xsim__DOT__weight_bytes__9__id)
                                                      ? 0x00001000U
                                                      : 0x00000240U)))))))
                        : (((((((((8U == __Vfunc_tb_d5_mobilenet_05_xsim__DOT__weight_bytes__9__id) 
                                  | (9U == __Vfunc_tb_d5_mobilenet_05_xsim__DOT__weight_bytes__9__id)) 
                                 | (0x0000000aU == __Vfunc_tb_d5_mobilenet_05_xsim__DOT__weight_bytes__9__id)) 
                                | (0x0000000bU == __Vfunc_tb_d5_mobilenet_05_xsim__DOT__weight_bytes__9__id)) 
                               | (0x0000000cU == __Vfunc_tb_d5_mobilenet_05_xsim__DOT__weight_bytes__9__id)) 
                              | (0x0000000dU == __Vfunc_tb_d5_mobilenet_05_xsim__DOT__weight_bytes__9__id)) 
                             | (0x0000000eU == __Vfunc_tb_d5_mobilenet_05_xsim__DOT__weight_bytes__9__id)) 
                            | (0x0000000fU == __Vfunc_tb_d5_mobilenet_05_xsim__DOT__weight_bytes__9__id))
                            ? ((8U == __Vfunc_tb_d5_mobilenet_05_xsim__DOT__weight_bytes__9__id)
                                ? 0x00002000U : ((9U 
                                                  == __Vfunc_tb_d5_mobilenet_05_xsim__DOT__weight_bytes__9__id)
                                                  ? 0x00000480U
                                                  : 
                                                 ((0x0000000aU 
                                                   == __Vfunc_tb_d5_mobilenet_05_xsim__DOT__weight_bytes__9__id)
                                                   ? 0x00004000U
                                                   : 
                                                  ((0x0000000bU 
                                                    == __Vfunc_tb_d5_mobilenet_05_xsim__DOT__weight_bytes__9__id)
                                                    ? 0x00000480U
                                                    : 
                                                   ((0x0000000cU 
                                                     == __Vfunc_tb_d5_mobilenet_05_xsim__DOT__weight_bytes__9__id)
                                                     ? 0x00004000U
                                                     : 
                                                    ((0x0000000dU 
                                                      == __Vfunc_tb_d5_mobilenet_05_xsim__DOT__weight_bytes__9__id)
                                                      ? 0x00000900U
                                                      : 
                                                     ((0x0000000eU 
                                                       == __Vfunc_tb_d5_mobilenet_05_xsim__DOT__weight_bytes__9__id)
                                                       ? 0x00008000U
                                                       : 0x00000900U)))))))
                            : ((0x00000010U == __Vfunc_tb_d5_mobilenet_05_xsim__DOT__weight_bytes__9__id)
                                ? 0x00010000U : ((0x00000011U 
                                                  == __Vfunc_tb_d5_mobilenet_05_xsim__DOT__weight_bytes__9__id)
                                                  ? 0x00000900U
                                                  : 
                                                 ((0x00000012U 
                                                   == __Vfunc_tb_d5_mobilenet_05_xsim__DOT__weight_bytes__9__id)
                                                   ? 0x00020000U
                                                   : 
                                                  ((0x00000013U 
                                                    == __Vfunc_tb_d5_mobilenet_05_xsim__DOT__weight_bytes__9__id)
                                                    ? 0x00001200U
                                                    : 
                                                   ((0x00000014U 
                                                     == __Vfunc_tb_d5_mobilenet_05_xsim__DOT__weight_bytes__9__id)
                                                     ? 0x00040000U
                                                     : 
                                                    ((0x00000016U 
                                                      == __Vfunc_tb_d5_mobilenet_05_xsim__DOT__weight_bytes__9__id)
                                                      ? 0x00001400U
                                                      : 0U))))))));
                __Vfunc_tb_d5_mobilenet_05_xsim__DOT__expected_offset__8____VlefCall_0__weight_bytes 
                    = __Vfunc_tb_d5_mobilenet_05_xsim__DOT__weight_bytes__9__Vfuncout;
                __Vfunc_tb_d5_mobilenet_05_xsim__DOT__expected_offset__8__Vfuncout 
                    = (__Vfunc_tb_d5_mobilenet_05_xsim__DOT__expected_offset__8__Vfuncout 
                       + __Vfunc_tb_d5_mobilenet_05_xsim__DOT__expected_offset__8____VlefCall_0__weight_bytes);
                __Vfunc_tb_d5_mobilenet_05_xsim__DOT__expected_offset__8__i 
                    = ((IData)(1U) + __Vfunc_tb_d5_mobilenet_05_xsim__DOT__expected_offset__8__i);
            }
            __Vtask_tb_d5_mobilenet_05_xsim__DOT__check_descriptor__5____VlefCall_1__expected_offset 
                = __Vfunc_tb_d5_mobilenet_05_xsim__DOT__expected_offset__8__Vfuncout;
            VL_WRITEF_NX("[%0t] %%Fatal: tb_d5_mobilenet_05_xsim.sv:75: Assertion failed in %m: descriptor %0d offset %0d, expected %0d\n",6, 'M',vlSymsp->name(),"tb_d5_mobilenet_05_xsim.check_descriptor", 'T',-9
                         , '#',64,VL_TIME_UNITED_Q(1000)
                         , '~',32,__Vtask_tb_d5_mobilenet_05_xsim__DOT__check_descriptor__5__expected_id
                         , '#',32,vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__weight_offset
                         , '#',32,__Vtask_tb_d5_mobilenet_05_xsim__DOT__check_descriptor__5____VlefCall_1__expected_offset);
            VL_STOP_MT("rtl/sim/tb_d5_mobilenet_05_xsim.sv", 75, "", false);
        }
        if ((0U == __Vtask_tb_d5_mobilenet_05_xsim__DOT__check_descriptor__5__expected_id)) {
            if (VL_UNLIKELY((((((((0U != (IData)(vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__op_kind)) 
                                  | (3U != (IData)(vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__in_channels))) 
                                 | (0x0010U != (IData)(vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__out_channels))) 
                                | (0x20U != (IData)(vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__height))) 
                               | (0x20U != (IData)(vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__width))) 
                              | (1U != (IData)(vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__stride)))))) {
                VL_WRITEF_NX("[%0t] %%Fatal: tb_d5_mobilenet_05_xsim.sv:83: Assertion failed in %m: invalid stem descriptor\n",3, 'M',vlSymsp->name(),"tb_d5_mobilenet_05_xsim.check_descriptor", 'T',-9
                             , '#',64,VL_TIME_UNITED_Q(1000));
                VL_STOP_MT("rtl/sim/tb_d5_mobilenet_05_xsim.sv", 83, "", false);
            }
        } else if ((1U == __Vtask_tb_d5_mobilenet_05_xsim__DOT__check_descriptor__5__expected_id)) {
            if (VL_UNLIKELY(((((1U != (IData)(vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__op_kind)) 
                               | (0x0010U != (IData)(vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__in_channels))) 
                              | (0x0010U != (IData)(vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__out_channels)))))) {
                VL_WRITEF_NX("[%0t] %%Fatal: tb_d5_mobilenet_05_xsim.sv:85: Assertion failed in %m: invalid first depthwise descriptor\n",3, 'M',vlSymsp->name(),"tb_d5_mobilenet_05_xsim.check_descriptor", 'T',-9
                             , '#',64,VL_TIME_UNITED_Q(1000));
                VL_STOP_MT("rtl/sim/tb_d5_mobilenet_05_xsim.sv", 85, "", false);
            }
        } else if ((2U == __Vtask_tb_d5_mobilenet_05_xsim__DOT__check_descriptor__5__expected_id)) {
            if (VL_UNLIKELY(((((2U != (IData)(vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__op_kind)) 
                               | (0x0010U != (IData)(vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__in_channels))) 
                              | (0x0020U != (IData)(vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__out_channels)))))) {
                VL_WRITEF_NX("[%0t] %%Fatal: tb_d5_mobilenet_05_xsim.sv:87: Assertion failed in %m: invalid first pointwise descriptor\n",3, 'M',vlSymsp->name(),"tb_d5_mobilenet_05_xsim.check_descriptor", 'T',-9
                             , '#',64,VL_TIME_UNITED_Q(1000));
                VL_STOP_MT("rtl/sim/tb_d5_mobilenet_05_xsim.sv", 87, "", false);
            }
        } else if (((0x0000000dU == __Vtask_tb_d5_mobilenet_05_xsim__DOT__check_descriptor__5__expected_id) 
                    || (0x0000000fU == __Vtask_tb_d5_mobilenet_05_xsim__DOT__check_descriptor__5__expected_id))) {
            if (VL_UNLIKELY(((((1U != (IData)(vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__op_kind)) 
                               | (0x0100U != (IData)(vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__in_channels))) 
                              | (0x0100U != (IData)(vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__out_channels)))))) {
                VL_WRITEF_NX("[%0t] %%Fatal: tb_d5_mobilenet_05_xsim.sv:89: Assertion failed in %m: invalid 256-channel depthwise descriptor\n",3, 'M',vlSymsp->name(),"tb_d5_mobilenet_05_xsim.check_descriptor", 'T',-9
                             , '#',64,VL_TIME_UNITED_Q(1000));
                VL_STOP_MT("rtl/sim/tb_d5_mobilenet_05_xsim.sv", 89, "", false);
            }
        } else if (((0x0000000eU == __Vtask_tb_d5_mobilenet_05_xsim__DOT__check_descriptor__5__expected_id) 
                    || (0x00000010U == __Vtask_tb_d5_mobilenet_05_xsim__DOT__check_descriptor__5__expected_id))) {
            if (VL_UNLIKELY(((((2U != (IData)(vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__op_kind)) 
                               | (0x0100U != (IData)(vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__in_channels))) 
                              | (0x0100U != (IData)(vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__out_channels)))))) {
                VL_WRITEF_NX("[%0t] %%Fatal: tb_d5_mobilenet_05_xsim.sv:91: Assertion failed in %m: invalid retained 256-to-256 pointwise descriptor\n",3, 'M',vlSymsp->name(),"tb_d5_mobilenet_05_xsim.check_descriptor", 'T',-9
                             , '#',64,VL_TIME_UNITED_Q(1000));
                VL_STOP_MT("rtl/sim/tb_d5_mobilenet_05_xsim.sv", 91, "", false);
            }
        } else if ((0x00000015U == __Vtask_tb_d5_mobilenet_05_xsim__DOT__check_descriptor__5__expected_id)) {
            if (VL_UNLIKELY(((((((3U != (IData)(vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__op_kind)) 
                                 | (0x0200U != (IData)(vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__in_channels))) 
                                | (0x0200U != (IData)(vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__out_channels))) 
                               | (2U != (IData)(vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__height))) 
                              | (2U != (IData)(vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__width)))))) {
                VL_WRITEF_NX("[%0t] %%Fatal: tb_d5_mobilenet_05_xsim.sv:94: Assertion failed in %m: invalid GAP descriptor\n",3, 'M',vlSymsp->name(),"tb_d5_mobilenet_05_xsim.check_descriptor", 'T',-9
                             , '#',64,VL_TIME_UNITED_Q(1000));
                VL_STOP_MT("rtl/sim/tb_d5_mobilenet_05_xsim.sv", 94, "", false);
            }
        } else if ((0x00000016U == __Vtask_tb_d5_mobilenet_05_xsim__DOT__check_descriptor__5__expected_id)) {
            if (VL_UNLIKELY(((((((4U != (IData)(vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__op_kind)) 
                                 | (0x0200U != (IData)(vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__in_channels))) 
                                | (0x000aU != (IData)(vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__out_channels))) 
                               | (1U != (IData)(vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__height))) 
                              | (1U != (IData)(vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__width)))))) {
                VL_WRITEF_NX("[%0t] %%Fatal: tb_d5_mobilenet_05_xsim.sv:97: Assertion failed in %m: invalid FC descriptor\n",3, 'M',vlSymsp->name(),"tb_d5_mobilenet_05_xsim.check_descriptor", 'T',-9
                             , '#',64,VL_TIME_UNITED_Q(1000));
                VL_STOP_MT("rtl/sim/tb_d5_mobilenet_05_xsim.sv", 97, "", false);
            }
        }
        tb_d5_mobilenet_05_xsim__DOT__id = ((IData)(1U) 
                                            + tb_d5_mobilenet_05_xsim__DOT__id);
    }
    Vtb_d5_mobilenet_05_xsim___024root____VbeforeTrig_hc3011a3e__0(vlSelf, 
                                                                   "@(negedge tb_d5_mobilenet_05_xsim.clk)");
    co_await vlSelfRef.__VtrigSched_hc3011a3e__0.trigger(0U, 
                                                         nullptr, 
                                                         "@(negedge tb_d5_mobilenet_05_xsim.clk)", 
                                                         "rtl/sim/tb_d5_mobilenet_05_xsim.sv", 
                                                         127);
    vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__layer_done = 1U;
    Vtb_d5_mobilenet_05_xsim___024root____VbeforeTrig_hc3011dec__0(vlSelf, 
                                                                   "@(posedge tb_d5_mobilenet_05_xsim.clk)");
    co_await vlSelfRef.__VtrigSched_hc3011dec__0.trigger(0U, 
                                                         nullptr, 
                                                         "@(posedge tb_d5_mobilenet_05_xsim.clk)", 
                                                         "rtl/sim/tb_d5_mobilenet_05_xsim.sv", 
                                                         129);
    co_await vlSelfRef.__VdlySched.delay(0x00000000000003e8ULL, 
                                         nullptr, "rtl/sim/tb_d5_mobilenet_05_xsim.sv", 
                                         129);
    vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__layer_done = 0U;
    if (VL_UNLIKELY(((1U & ((~ (IData)(vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__done)) 
                            | (IData)(vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__busy)))))) {
        VL_WRITEF_NX("[%0t] %%Fatal: tb_d5_mobilenet_05_xsim.sv:131: Assertion failed in %m: sequencer did not complete after FC\n",3, 'M',vlSymsp->name(),"tb_d5_mobilenet_05_xsim", 'T',-9
                     , '#',64,VL_TIME_UNITED_Q(1000));
        VL_STOP_MT("rtl/sim/tb_d5_mobilenet_05_xsim.sv", 131, "", false);
    }
    VL_WRITEF_NX("D5 XSim descriptor sequencer PASS: stem + 10 blocks + GAP + FC\n",0);
    VL_FINISH_MT("rtl/sim/tb_d5_mobilenet_05_xsim.sv", 133, "");
    co_return;
}

VlCoroutine Vtb_d5_mobilenet_05_xsim___024root___eval_initial__TOP__Vtiming__1(Vtb_d5_mobilenet_05_xsim___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtb_d5_mobilenet_05_xsim___024root___eval_initial__TOP__Vtiming__1\n"); );
    Vtb_d5_mobilenet_05_xsim__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    while (VL_LIKELY(!vlSymsp->_vm_contextp__->gotFinish())) {
        co_await vlSelfRef.__VdlySched.delay(0x0000000000001388ULL, 
                                             nullptr, 
                                             "rtl/sim/tb_d5_mobilenet_05_xsim.sv", 
                                             38);
        vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__clk 
            = (1U & (~ (IData)(vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__clk)));
    }
    co_return;
}

void Vtb_d5_mobilenet_05_xsim___024root___eval_triggers_vec__act(Vtb_d5_mobilenet_05_xsim___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtb_d5_mobilenet_05_xsim___024root___eval_triggers_vec__act\n"); );
    Vtb_d5_mobilenet_05_xsim__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    vlSelfRef.__VactTriggered[0U] = (QData)((IData)(
                                                    (((vlSelfRef.__VdlySched.awaitingCurrentTime() 
                                                       << 3U) 
                                                      | (((~ (IData)(vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__clk)) 
                                                          & (IData)(vlSelfRef.__Vtrigprevexpr___TOP__tb_d5_mobilenet_05_xsim__DOT__clk__0)) 
                                                         << 2U)) 
                                                     | ((((~ (IData)(vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__rst_n)) 
                                                          & (IData)(vlSelfRef.__Vtrigprevexpr___TOP__tb_d5_mobilenet_05_xsim__DOT__rst_n__0)) 
                                                         << 1U) 
                                                        | ((IData)(vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__clk) 
                                                           & (~ (IData)(vlSelfRef.__Vtrigprevexpr___TOP__tb_d5_mobilenet_05_xsim__DOT__clk__0)))))));
    vlSelfRef.__Vtrigprevexpr___TOP__tb_d5_mobilenet_05_xsim__DOT__clk__0 
        = vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__clk;
    vlSelfRef.__Vtrigprevexpr___TOP__tb_d5_mobilenet_05_xsim__DOT__rst_n__0 
        = vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__rst_n;
}

bool Vtb_d5_mobilenet_05_xsim___024root___trigger_anySet__act(const VlUnpacked<QData/*63:0*/, 1> &in) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtb_d5_mobilenet_05_xsim___024root___trigger_anySet__act\n"); );
    // Locals
    IData/*31:0*/ n;
    // Body
    n = 0U;
    do {
        if (in[n]) {
            return (1U);
        }
        n = ((IData)(1U) + n);
    } while ((1U > n));
    return (0U);
}

void Vtb_d5_mobilenet_05_xsim___024root___nba_sequent__TOP__0(Vtb_d5_mobilenet_05_xsim___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtb_d5_mobilenet_05_xsim___024root___nba_sequent__TOP__0\n"); );
    Vtb_d5_mobilenet_05_xsim__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Locals
    CData/*4:0*/ __Vfunc_tb_d5_mobilenet_05_xsim__DOT__dut__DOT__layer_weight_bytes__10__id;
    __Vfunc_tb_d5_mobilenet_05_xsim__DOT__dut__DOT__layer_weight_bytes__10__id = 0;
    CData/*4:0*/ __Vtask_tb_d5_mobilenet_05_xsim__DOT__dut__DOT__set_descriptor__11__id;
    __Vtask_tb_d5_mobilenet_05_xsim__DOT__dut__DOT__set_descriptor__11__id = 0;
    // Body
    if (vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__rst_n) {
        vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__layer_start = 0U;
        vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__done = 0U;
        if (vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__dut__DOT__active) {
            if (vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__layer_done) {
                if ((0x16U == (IData)(vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__layer_id))) {
                    vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__dut__DOT__active = 0U;
                    vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__busy = 0U;
                    vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__done = 1U;
                } else {
                    __Vfunc_tb_d5_mobilenet_05_xsim__DOT__dut__DOT__layer_weight_bytes__10__id 
                        = vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__layer_id;
                    vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__dut__DOT____VlemCall_0__layer_weight_bytes 
                        = (((((((((0U == (IData)(__Vfunc_tb_d5_mobilenet_05_xsim__DOT__dut__DOT__layer_weight_bytes__10__id)) 
                                  | (1U == (IData)(__Vfunc_tb_d5_mobilenet_05_xsim__DOT__dut__DOT__layer_weight_bytes__10__id))) 
                                 | (2U == (IData)(__Vfunc_tb_d5_mobilenet_05_xsim__DOT__dut__DOT__layer_weight_bytes__10__id))) 
                                | (3U == (IData)(__Vfunc_tb_d5_mobilenet_05_xsim__DOT__dut__DOT__layer_weight_bytes__10__id))) 
                               | (4U == (IData)(__Vfunc_tb_d5_mobilenet_05_xsim__DOT__dut__DOT__layer_weight_bytes__10__id))) 
                              | (5U == (IData)(__Vfunc_tb_d5_mobilenet_05_xsim__DOT__dut__DOT__layer_weight_bytes__10__id))) 
                             | (6U == (IData)(__Vfunc_tb_d5_mobilenet_05_xsim__DOT__dut__DOT__layer_weight_bytes__10__id))) 
                            | (7U == (IData)(__Vfunc_tb_d5_mobilenet_05_xsim__DOT__dut__DOT__layer_weight_bytes__10__id)))
                            ? ((0U == (IData)(__Vfunc_tb_d5_mobilenet_05_xsim__DOT__dut__DOT__layer_weight_bytes__10__id))
                                ? 0x000001b0U : ((1U 
                                                  == (IData)(__Vfunc_tb_d5_mobilenet_05_xsim__DOT__dut__DOT__layer_weight_bytes__10__id))
                                                  ? 0x00000090U
                                                  : 
                                                 ((2U 
                                                   == (IData)(__Vfunc_tb_d5_mobilenet_05_xsim__DOT__dut__DOT__layer_weight_bytes__10__id))
                                                   ? 0x00000200U
                                                   : 
                                                  ((3U 
                                                    == (IData)(__Vfunc_tb_d5_mobilenet_05_xsim__DOT__dut__DOT__layer_weight_bytes__10__id))
                                                    ? 0x00000120U
                                                    : 
                                                   ((4U 
                                                     == (IData)(__Vfunc_tb_d5_mobilenet_05_xsim__DOT__dut__DOT__layer_weight_bytes__10__id))
                                                     ? 0x00000800U
                                                     : 
                                                    ((5U 
                                                      == (IData)(__Vfunc_tb_d5_mobilenet_05_xsim__DOT__dut__DOT__layer_weight_bytes__10__id))
                                                      ? 0x00000240U
                                                      : 
                                                     ((6U 
                                                       == (IData)(__Vfunc_tb_d5_mobilenet_05_xsim__DOT__dut__DOT__layer_weight_bytes__10__id))
                                                       ? 0x00001000U
                                                       : 0x00000240U)))))))
                            : (((((((((8U == (IData)(__Vfunc_tb_d5_mobilenet_05_xsim__DOT__dut__DOT__layer_weight_bytes__10__id)) 
                                      | (9U == (IData)(__Vfunc_tb_d5_mobilenet_05_xsim__DOT__dut__DOT__layer_weight_bytes__10__id))) 
                                     | (0x0aU == (IData)(__Vfunc_tb_d5_mobilenet_05_xsim__DOT__dut__DOT__layer_weight_bytes__10__id))) 
                                    | (0x0bU == (IData)(__Vfunc_tb_d5_mobilenet_05_xsim__DOT__dut__DOT__layer_weight_bytes__10__id))) 
                                   | (0x0cU == (IData)(__Vfunc_tb_d5_mobilenet_05_xsim__DOT__dut__DOT__layer_weight_bytes__10__id))) 
                                  | (0x0dU == (IData)(__Vfunc_tb_d5_mobilenet_05_xsim__DOT__dut__DOT__layer_weight_bytes__10__id))) 
                                 | (0x0eU == (IData)(__Vfunc_tb_d5_mobilenet_05_xsim__DOT__dut__DOT__layer_weight_bytes__10__id))) 
                                | (0x0fU == (IData)(__Vfunc_tb_d5_mobilenet_05_xsim__DOT__dut__DOT__layer_weight_bytes__10__id)))
                                ? ((8U == (IData)(__Vfunc_tb_d5_mobilenet_05_xsim__DOT__dut__DOT__layer_weight_bytes__10__id))
                                    ? 0x00002000U : 
                                   ((9U == (IData)(__Vfunc_tb_d5_mobilenet_05_xsim__DOT__dut__DOT__layer_weight_bytes__10__id))
                                     ? 0x00000480U : 
                                    ((0x0aU == (IData)(__Vfunc_tb_d5_mobilenet_05_xsim__DOT__dut__DOT__layer_weight_bytes__10__id))
                                      ? 0x00004000U
                                      : ((0x0bU == (IData)(__Vfunc_tb_d5_mobilenet_05_xsim__DOT__dut__DOT__layer_weight_bytes__10__id))
                                          ? 0x00000480U
                                          : ((0x0cU 
                                              == (IData)(__Vfunc_tb_d5_mobilenet_05_xsim__DOT__dut__DOT__layer_weight_bytes__10__id))
                                              ? 0x00004000U
                                              : ((0x0dU 
                                                  == (IData)(__Vfunc_tb_d5_mobilenet_05_xsim__DOT__dut__DOT__layer_weight_bytes__10__id))
                                                  ? 0x00000900U
                                                  : 
                                                 ((0x0eU 
                                                   == (IData)(__Vfunc_tb_d5_mobilenet_05_xsim__DOT__dut__DOT__layer_weight_bytes__10__id))
                                                   ? 0x00008000U
                                                   : 0x00000900U)))))))
                                : ((0x10U == (IData)(__Vfunc_tb_d5_mobilenet_05_xsim__DOT__dut__DOT__layer_weight_bytes__10__id))
                                    ? 0x00010000U : 
                                   ((0x11U == (IData)(__Vfunc_tb_d5_mobilenet_05_xsim__DOT__dut__DOT__layer_weight_bytes__10__id))
                                     ? 0x00000900U : 
                                    ((0x12U == (IData)(__Vfunc_tb_d5_mobilenet_05_xsim__DOT__dut__DOT__layer_weight_bytes__10__id))
                                      ? 0x00020000U
                                      : ((0x13U == (IData)(__Vfunc_tb_d5_mobilenet_05_xsim__DOT__dut__DOT__layer_weight_bytes__10__id))
                                          ? 0x00001200U
                                          : ((0x14U 
                                              == (IData)(__Vfunc_tb_d5_mobilenet_05_xsim__DOT__dut__DOT__layer_weight_bytes__10__id))
                                              ? 0x00040000U
                                              : ((0x16U 
                                                  == (IData)(__Vfunc_tb_d5_mobilenet_05_xsim__DOT__dut__DOT__layer_weight_bytes__10__id))
                                                  ? 0x00001400U
                                                  : 0U))))))));
                    vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__weight_offset 
                        = (vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__weight_offset 
                           + vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__dut__DOT____VlemCall_0__layer_weight_bytes);
                    vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__layer_start = 1U;
                    __Vtask_tb_d5_mobilenet_05_xsim__DOT__dut__DOT__set_descriptor__11__id 
                        = (0x0000001fU & ((IData)(1U) 
                                          + (IData)(vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__layer_id)));
                    vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__layer_id 
                        = __Vtask_tb_d5_mobilenet_05_xsim__DOT__dut__DOT__set_descriptor__11__id;
                    vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__stride = 0U;
                    vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__in_channels = 0U;
                    vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__out_channels = 0U;
                    vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__height = 0U;
                    vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__width = 0U;
                    vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__op_kind = 0U;
                    if (((((((((0U == (IData)(__Vtask_tb_d5_mobilenet_05_xsim__DOT__dut__DOT__set_descriptor__11__id)) 
                               | (1U == (IData)(__Vtask_tb_d5_mobilenet_05_xsim__DOT__dut__DOT__set_descriptor__11__id))) 
                              | (2U == (IData)(__Vtask_tb_d5_mobilenet_05_xsim__DOT__dut__DOT__set_descriptor__11__id))) 
                             | (3U == (IData)(__Vtask_tb_d5_mobilenet_05_xsim__DOT__dut__DOT__set_descriptor__11__id))) 
                            | (4U == (IData)(__Vtask_tb_d5_mobilenet_05_xsim__DOT__dut__DOT__set_descriptor__11__id))) 
                           | (5U == (IData)(__Vtask_tb_d5_mobilenet_05_xsim__DOT__dut__DOT__set_descriptor__11__id))) 
                          | (6U == (IData)(__Vtask_tb_d5_mobilenet_05_xsim__DOT__dut__DOT__set_descriptor__11__id))) 
                         | (7U == (IData)(__Vtask_tb_d5_mobilenet_05_xsim__DOT__dut__DOT__set_descriptor__11__id)))) {
                        if ((0U == (IData)(__Vtask_tb_d5_mobilenet_05_xsim__DOT__dut__DOT__set_descriptor__11__id))) {
                            vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__op_kind = 0U;
                            vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__in_channels = 3U;
                            vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__out_channels = 0x0010U;
                            vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__height = 0x20U;
                            vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__width = 0x20U;
                            vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__stride = 1U;
                        } else if ((1U == (IData)(__Vtask_tb_d5_mobilenet_05_xsim__DOT__dut__DOT__set_descriptor__11__id))) {
                            vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__op_kind = 1U;
                            vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__in_channels = 0x0010U;
                            vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__out_channels = 0x0010U;
                            vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__height = 0x20U;
                            vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__width = 0x20U;
                            vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__stride = 1U;
                        } else if ((2U == (IData)(__Vtask_tb_d5_mobilenet_05_xsim__DOT__dut__DOT__set_descriptor__11__id))) {
                            vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__op_kind = 2U;
                            vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__in_channels = 0x0010U;
                            vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__out_channels = 0x0020U;
                            vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__height = 0x20U;
                            vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__width = 0x20U;
                            vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__stride = 1U;
                        } else if ((3U == (IData)(__Vtask_tb_d5_mobilenet_05_xsim__DOT__dut__DOT__set_descriptor__11__id))) {
                            vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__op_kind = 1U;
                            vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__in_channels = 0x0020U;
                            vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__out_channels = 0x0020U;
                            vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__height = 0x20U;
                            vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__width = 0x20U;
                            vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__stride = 2U;
                        } else if ((4U == (IData)(__Vtask_tb_d5_mobilenet_05_xsim__DOT__dut__DOT__set_descriptor__11__id))) {
                            vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__op_kind = 2U;
                            vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__in_channels = 0x0020U;
                            vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__out_channels = 0x0040U;
                            vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__height = 0x10U;
                            vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__width = 0x10U;
                            vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__stride = 1U;
                        } else if ((5U == (IData)(__Vtask_tb_d5_mobilenet_05_xsim__DOT__dut__DOT__set_descriptor__11__id))) {
                            vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__op_kind = 1U;
                            vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__in_channels = 0x0040U;
                            vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__out_channels = 0x0040U;
                            vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__height = 0x10U;
                            vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__width = 0x10U;
                            vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__stride = 1U;
                        } else if ((6U == (IData)(__Vtask_tb_d5_mobilenet_05_xsim__DOT__dut__DOT__set_descriptor__11__id))) {
                            vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__op_kind = 2U;
                            vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__in_channels = 0x0040U;
                            vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__out_channels = 0x0040U;
                            vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__height = 0x10U;
                            vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__width = 0x10U;
                            vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__stride = 1U;
                        } else {
                            vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__op_kind = 1U;
                            vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__in_channels = 0x0040U;
                            vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__out_channels = 0x0040U;
                            vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__height = 0x10U;
                            vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__width = 0x10U;
                            vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__stride = 2U;
                        }
                    } else if (((((((((8U == (IData)(__Vtask_tb_d5_mobilenet_05_xsim__DOT__dut__DOT__set_descriptor__11__id)) 
                                      | (9U == (IData)(__Vtask_tb_d5_mobilenet_05_xsim__DOT__dut__DOT__set_descriptor__11__id))) 
                                     | (0x0aU == (IData)(__Vtask_tb_d5_mobilenet_05_xsim__DOT__dut__DOT__set_descriptor__11__id))) 
                                    | (0x0bU == (IData)(__Vtask_tb_d5_mobilenet_05_xsim__DOT__dut__DOT__set_descriptor__11__id))) 
                                   | (0x0cU == (IData)(__Vtask_tb_d5_mobilenet_05_xsim__DOT__dut__DOT__set_descriptor__11__id))) 
                                  | ((0x0dU == (IData)(__Vtask_tb_d5_mobilenet_05_xsim__DOT__dut__DOT__set_descriptor__11__id)) 
                                     || (0x0fU == (IData)(__Vtask_tb_d5_mobilenet_05_xsim__DOT__dut__DOT__set_descriptor__11__id)))) 
                                 | ((0x0eU == (IData)(__Vtask_tb_d5_mobilenet_05_xsim__DOT__dut__DOT__set_descriptor__11__id)) 
                                    || (0x10U == (IData)(__Vtask_tb_d5_mobilenet_05_xsim__DOT__dut__DOT__set_descriptor__11__id)))) 
                                | (0x11U == (IData)(__Vtask_tb_d5_mobilenet_05_xsim__DOT__dut__DOT__set_descriptor__11__id)))) {
                        if ((8U == (IData)(__Vtask_tb_d5_mobilenet_05_xsim__DOT__dut__DOT__set_descriptor__11__id))) {
                            vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__op_kind = 2U;
                            vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__in_channels = 0x0040U;
                            vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__out_channels = 0x0080U;
                            vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__height = 8U;
                            vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__width = 8U;
                            vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__stride = 1U;
                        } else if ((9U == (IData)(__Vtask_tb_d5_mobilenet_05_xsim__DOT__dut__DOT__set_descriptor__11__id))) {
                            vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__op_kind = 1U;
                            vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__in_channels = 0x0080U;
                            vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__out_channels = 0x0080U;
                            vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__height = 8U;
                            vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__width = 8U;
                            vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__stride = 1U;
                        } else if ((0x0aU == (IData)(__Vtask_tb_d5_mobilenet_05_xsim__DOT__dut__DOT__set_descriptor__11__id))) {
                            vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__op_kind = 2U;
                            vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__in_channels = 0x0080U;
                            vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__out_channels = 0x0080U;
                            vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__height = 8U;
                            vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__width = 8U;
                            vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__stride = 1U;
                        } else if ((0x0bU == (IData)(__Vtask_tb_d5_mobilenet_05_xsim__DOT__dut__DOT__set_descriptor__11__id))) {
                            vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__op_kind = 1U;
                            vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__in_channels = 0x0080U;
                            vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__out_channels = 0x0080U;
                            vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__height = 8U;
                            vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__width = 8U;
                            vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__stride = 2U;
                        } else if ((0x0cU == (IData)(__Vtask_tb_d5_mobilenet_05_xsim__DOT__dut__DOT__set_descriptor__11__id))) {
                            vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__op_kind = 2U;
                            vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__in_channels = 0x0080U;
                            vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__out_channels = 0x0100U;
                            vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__height = 4U;
                            vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__width = 4U;
                            vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__stride = 1U;
                        } else if (((0x0dU == (IData)(__Vtask_tb_d5_mobilenet_05_xsim__DOT__dut__DOT__set_descriptor__11__id)) 
                                    || (0x0fU == (IData)(__Vtask_tb_d5_mobilenet_05_xsim__DOT__dut__DOT__set_descriptor__11__id)))) {
                            vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__op_kind = 1U;
                            vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__in_channels = 0x0100U;
                            vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__out_channels = 0x0100U;
                            vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__height = 4U;
                            vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__width = 4U;
                            vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__stride = 1U;
                        } else if (((0x0eU == (IData)(__Vtask_tb_d5_mobilenet_05_xsim__DOT__dut__DOT__set_descriptor__11__id)) 
                                    || (0x10U == (IData)(__Vtask_tb_d5_mobilenet_05_xsim__DOT__dut__DOT__set_descriptor__11__id)))) {
                            vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__op_kind = 2U;
                            vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__in_channels = 0x0100U;
                            vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__out_channels = 0x0100U;
                            vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__height = 4U;
                            vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__width = 4U;
                            vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__stride = 1U;
                        } else {
                            vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__op_kind = 1U;
                            vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__in_channels = 0x0100U;
                            vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__out_channels = 0x0100U;
                            vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__height = 4U;
                            vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__width = 4U;
                            vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__stride = 2U;
                        }
                    } else if ((0x12U == (IData)(__Vtask_tb_d5_mobilenet_05_xsim__DOT__dut__DOT__set_descriptor__11__id))) {
                        vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__op_kind = 2U;
                        vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__in_channels = 0x0100U;
                        vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__out_channels = 0x0200U;
                        vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__height = 2U;
                        vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__width = 2U;
                        vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__stride = 1U;
                    } else if ((0x13U == (IData)(__Vtask_tb_d5_mobilenet_05_xsim__DOT__dut__DOT__set_descriptor__11__id))) {
                        vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__op_kind = 1U;
                        vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__in_channels = 0x0200U;
                        vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__out_channels = 0x0200U;
                        vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__height = 2U;
                        vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__width = 2U;
                        vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__stride = 1U;
                    } else if ((0x14U == (IData)(__Vtask_tb_d5_mobilenet_05_xsim__DOT__dut__DOT__set_descriptor__11__id))) {
                        vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__op_kind = 2U;
                        vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__in_channels = 0x0200U;
                        vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__out_channels = 0x0200U;
                        vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__height = 2U;
                        vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__width = 2U;
                        vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__stride = 1U;
                    } else if ((0x15U == (IData)(__Vtask_tb_d5_mobilenet_05_xsim__DOT__dut__DOT__set_descriptor__11__id))) {
                        vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__op_kind = 3U;
                        vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__in_channels = 0x0200U;
                        vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__out_channels = 0x0200U;
                        vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__height = 2U;
                        vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__width = 2U;
                        vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__stride = 1U;
                    } else if ((0x16U == (IData)(__Vtask_tb_d5_mobilenet_05_xsim__DOT__dut__DOT__set_descriptor__11__id))) {
                        vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__op_kind = 4U;
                        vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__in_channels = 0x0200U;
                        vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__out_channels = 0x000aU;
                        vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__height = 1U;
                        vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__width = 1U;
                        vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__stride = 1U;
                    }
                }
            }
        } else {
            vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__busy = 0U;
            if (vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__start) {
                vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__dut__DOT__active = 1U;
                vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__busy = 1U;
                vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__weight_offset = 0U;
                vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__layer_id = 0U;
                vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__stride = 0U;
                vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__layer_start = 1U;
                vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__in_channels = 0U;
                vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__out_channels = 0U;
                vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__height = 0U;
                vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__width = 0U;
                vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__op_kind = 0U;
                vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__op_kind = 0U;
                vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__in_channels = 3U;
                vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__out_channels = 0x0010U;
                vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__height = 0x20U;
                vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__width = 0x20U;
                vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__stride = 1U;
            }
        }
    } else {
        vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__dut__DOT__active = 0U;
        vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__layer_start = 0U;
        vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__layer_id = 0U;
        vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__op_kind = 0U;
        vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__in_channels = 0U;
        vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__out_channels = 0U;
        vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__height = 0U;
        vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__width = 0U;
        vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__stride = 0U;
        vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__weight_offset = 0U;
        vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__busy = 0U;
        vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__done = 0U;
    }
}

void Vtb_d5_mobilenet_05_xsim___024root___eval_nba(Vtb_d5_mobilenet_05_xsim___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtb_d5_mobilenet_05_xsim___024root___eval_nba\n"); );
    Vtb_d5_mobilenet_05_xsim__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    if ((3ULL & vlSelfRef.__VnbaTriggered[0U])) {
        Vtb_d5_mobilenet_05_xsim___024root___nba_sequent__TOP__0(vlSelf);
    }
}

void Vtb_d5_mobilenet_05_xsim___024root___timing_ready(Vtb_d5_mobilenet_05_xsim___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtb_d5_mobilenet_05_xsim___024root___timing_ready\n"); );
    Vtb_d5_mobilenet_05_xsim__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    if ((1ULL & vlSelfRef.__VactTriggered[0U])) {
        vlSelfRef.__VtrigSched_hc3011dec__0.ready("@(posedge tb_d5_mobilenet_05_xsim.clk)");
    }
    if ((4ULL & vlSelfRef.__VactTriggered[0U])) {
        vlSelfRef.__VtrigSched_hc3011a3e__0.ready("@(negedge tb_d5_mobilenet_05_xsim.clk)");
    }
}

void Vtb_d5_mobilenet_05_xsim___024root___timing_resume(Vtb_d5_mobilenet_05_xsim___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtb_d5_mobilenet_05_xsim___024root___timing_resume\n"); );
    Vtb_d5_mobilenet_05_xsim__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    vlSelfRef.__VtrigSched_hc3011dec__0.moveToResumeQueue(
                                                          "@(posedge tb_d5_mobilenet_05_xsim.clk)");
    vlSelfRef.__VtrigSched_hc3011a3e__0.moveToResumeQueue(
                                                          "@(negedge tb_d5_mobilenet_05_xsim.clk)");
    vlSelfRef.__VtrigSched_hc3011dec__0.resume("@(posedge tb_d5_mobilenet_05_xsim.clk)");
    vlSelfRef.__VtrigSched_hc3011a3e__0.resume("@(negedge tb_d5_mobilenet_05_xsim.clk)");
    if ((8ULL & vlSelfRef.__VactTriggered[0U])) {
        vlSelfRef.__VdlySched.resume();
    }
}

void Vtb_d5_mobilenet_05_xsim___024root___trigger_orInto__act_vec_vec(VlUnpacked<QData/*63:0*/, 1> &out, const VlUnpacked<QData/*63:0*/, 1> &in) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtb_d5_mobilenet_05_xsim___024root___trigger_orInto__act_vec_vec\n"); );
    // Locals
    IData/*31:0*/ n;
    // Body
    n = 0U;
    do {
        out[n] = (out[n] | in[n]);
        n = ((IData)(1U) + n);
    } while ((0U >= n));
}

#ifdef VL_DEBUG
VL_ATTR_COLD void Vtb_d5_mobilenet_05_xsim___024root___dump_triggers__act(const VlUnpacked<QData/*63:0*/, 1> &triggers, const std::string &tag);
#endif  // VL_DEBUG

bool Vtb_d5_mobilenet_05_xsim___024root___eval_phase__act(Vtb_d5_mobilenet_05_xsim___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtb_d5_mobilenet_05_xsim___024root___eval_phase__act\n"); );
    Vtb_d5_mobilenet_05_xsim__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Locals
    CData/*0:0*/ __VactExecute;
    // Body
    Vtb_d5_mobilenet_05_xsim___024root___eval_triggers_vec__act(vlSelf);
    Vtb_d5_mobilenet_05_xsim___024root___timing_ready(vlSelf);
    Vtb_d5_mobilenet_05_xsim___024root___trigger_orInto__act_vec_vec(vlSelfRef.__VactTriggered, vlSelfRef.__VactTriggeredAcc);
#ifdef VL_DEBUG
    if (VL_UNLIKELY(vlSymsp->_vm_contextp__->debug())) {
        Vtb_d5_mobilenet_05_xsim___024root___dump_triggers__act(vlSelfRef.__VactTriggered, "act"s);
    }
#endif
    Vtb_d5_mobilenet_05_xsim___024root___trigger_orInto__act_vec_vec(vlSelfRef.__VnbaTriggered, vlSelfRef.__VactTriggered);
    __VactExecute = Vtb_d5_mobilenet_05_xsim___024root___trigger_anySet__act(vlSelfRef.__VactTriggered);
    if (__VactExecute) {
        vlSelfRef.__VactTriggeredAcc.fill(0ULL);
        Vtb_d5_mobilenet_05_xsim___024root___timing_resume(vlSelf);
    }
    return (__VactExecute);
}

bool Vtb_d5_mobilenet_05_xsim___024root___eval_phase__inact(Vtb_d5_mobilenet_05_xsim___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtb_d5_mobilenet_05_xsim___024root___eval_phase__inact\n"); );
    Vtb_d5_mobilenet_05_xsim__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Locals
    CData/*0:0*/ __VinactExecute;
    // Body
    __VinactExecute = vlSelfRef.__VdlySched.awaitingZeroDelay();
    if (__VinactExecute) {
        VL_FATAL_MT("rtl/sim/tb_d5_mobilenet_05_xsim.sv", 7, "", "ZERODLY: Design Verilated with '--no-sched-zero-delay', but #0 delay executed at runtime");
    }
    return (__VinactExecute);
}

void Vtb_d5_mobilenet_05_xsim___024root___trigger_clear__act(VlUnpacked<QData/*63:0*/, 1> &out) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtb_d5_mobilenet_05_xsim___024root___trigger_clear__act\n"); );
    // Locals
    IData/*31:0*/ n;
    // Body
    n = 0U;
    do {
        out[n] = 0ULL;
        n = ((IData)(1U) + n);
    } while ((1U > n));
}

bool Vtb_d5_mobilenet_05_xsim___024root___eval_phase__nba(Vtb_d5_mobilenet_05_xsim___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtb_d5_mobilenet_05_xsim___024root___eval_phase__nba\n"); );
    Vtb_d5_mobilenet_05_xsim__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Locals
    CData/*0:0*/ __VnbaExecute;
    // Body
    __VnbaExecute = Vtb_d5_mobilenet_05_xsim___024root___trigger_anySet__act(vlSelfRef.__VnbaTriggered);
    if (__VnbaExecute) {
        Vtb_d5_mobilenet_05_xsim___024root___eval_nba(vlSelf);
        Vtb_d5_mobilenet_05_xsim___024root___trigger_clear__act(vlSelfRef.__VnbaTriggered);
    }
    return (__VnbaExecute);
}

void Vtb_d5_mobilenet_05_xsim___024root___eval(Vtb_d5_mobilenet_05_xsim___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtb_d5_mobilenet_05_xsim___024root___eval\n"); );
    Vtb_d5_mobilenet_05_xsim__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Locals
    IData/*31:0*/ __VnbaIterCount;
    // Body
    __VnbaIterCount = 0U;
    do {
        if (VL_UNLIKELY(((0x00002710U < __VnbaIterCount)))) {
#ifdef VL_DEBUG
            Vtb_d5_mobilenet_05_xsim___024root___dump_triggers__act(vlSelfRef.__VnbaTriggered, "nba"s);
#endif
            VL_FATAL_MT("rtl/sim/tb_d5_mobilenet_05_xsim.sv", 7, "", "DIDNOTCONVERGE: NBA region did not converge after '--converge-limit' of 10000 tries");
        }
        __VnbaIterCount = ((IData)(1U) + __VnbaIterCount);
        vlSelfRef.__VinactIterCount = 0U;
        do {
            if (VL_UNLIKELY(((0x00002710U < vlSelfRef.__VinactIterCount)))) {
                VL_FATAL_MT("rtl/sim/tb_d5_mobilenet_05_xsim.sv", 7, "", "DIDNOTCONVERGE: Inactive region did not converge after '--converge-limit' of 10000 tries");
            }
            vlSelfRef.__VinactIterCount = ((IData)(1U) 
                                           + vlSelfRef.__VinactIterCount);
            vlSelfRef.__VactIterCount = 0U;
            do {
                if (VL_UNLIKELY(((0x00002710U < vlSelfRef.__VactIterCount)))) {
#ifdef VL_DEBUG
                    Vtb_d5_mobilenet_05_xsim___024root___dump_triggers__act(vlSelfRef.__VactTriggered, "act"s);
#endif
                    VL_FATAL_MT("rtl/sim/tb_d5_mobilenet_05_xsim.sv", 7, "", "DIDNOTCONVERGE: Active region did not converge after '--converge-limit' of 10000 tries");
                }
                vlSelfRef.__VactIterCount = ((IData)(1U) 
                                             + vlSelfRef.__VactIterCount);
                vlSelfRef.__VactPhaseResult = Vtb_d5_mobilenet_05_xsim___024root___eval_phase__act(vlSelf);
            } while (vlSelfRef.__VactPhaseResult);
            vlSelfRef.__VinactPhaseResult = Vtb_d5_mobilenet_05_xsim___024root___eval_phase__inact(vlSelf);
        } while (vlSelfRef.__VinactPhaseResult);
        vlSelfRef.__VnbaPhaseResult = Vtb_d5_mobilenet_05_xsim___024root___eval_phase__nba(vlSelf);
    } while (vlSelfRef.__VnbaPhaseResult);
}

void Vtb_d5_mobilenet_05_xsim___024root____VbeforeTrig_hc3011dec__0(Vtb_d5_mobilenet_05_xsim___024root* vlSelf, const char* __VeventDescription) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtb_d5_mobilenet_05_xsim___024root____VbeforeTrig_hc3011dec__0\n"); );
    Vtb_d5_mobilenet_05_xsim__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Locals
    VlUnpacked<QData/*63:0*/, 1> __VTmp;
    // Body
    __VTmp[0U] = (QData)((IData)(((((~ (IData)(vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__clk)) 
                                    & (IData)(vlSelfRef.__Vtrigprevexpr___TOP__tb_d5_mobilenet_05_xsim__DOT__clk__0)) 
                                   << 2U) | ((IData)(vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__clk) 
                                             & (~ (IData)(vlSelfRef.__Vtrigprevexpr___TOP__tb_d5_mobilenet_05_xsim__DOT__clk__0))))));
    vlSelfRef.__Vtrigprevexpr___TOP__tb_d5_mobilenet_05_xsim__DOT__clk__0 
        = vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__clk;
    if ((1ULL & __VTmp[0U])) {
        vlSelfRef.__VtrigSched_hc3011dec__0.ready(__VeventDescription);
        vlSelfRef.__VtrigSched_hc3011dec__0.ready(__VeventDescription);
        vlSelfRef.__VtrigSched_hc3011dec__0.ready(__VeventDescription);
        vlSelfRef.__VtrigSched_hc3011dec__0.ready(__VeventDescription);
    }
    if ((4ULL & __VTmp[0U])) {
        vlSelfRef.__VtrigSched_hc3011a3e__0.ready(__VeventDescription);
        vlSelfRef.__VtrigSched_hc3011a3e__0.ready(__VeventDescription);
        vlSelfRef.__VtrigSched_hc3011a3e__0.ready(__VeventDescription);
    }
    vlSelfRef.__VactTriggeredAcc[0U] = (vlSelfRef.__VactTriggeredAcc[0U] 
                                        | __VTmp[0U]);
}

void Vtb_d5_mobilenet_05_xsim___024root____VbeforeTrig_hc3011a3e__0(Vtb_d5_mobilenet_05_xsim___024root* vlSelf, const char* __VeventDescription) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtb_d5_mobilenet_05_xsim___024root____VbeforeTrig_hc3011a3e__0\n"); );
    Vtb_d5_mobilenet_05_xsim__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Locals
    VlUnpacked<QData/*63:0*/, 1> __VTmp;
    // Body
    __VTmp[0U] = (QData)((IData)(((((~ (IData)(vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__clk)) 
                                    & (IData)(vlSelfRef.__Vtrigprevexpr___TOP__tb_d5_mobilenet_05_xsim__DOT__clk__0)) 
                                   << 2U) | ((IData)(vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__clk) 
                                             & (~ (IData)(vlSelfRef.__Vtrigprevexpr___TOP__tb_d5_mobilenet_05_xsim__DOT__clk__0))))));
    vlSelfRef.__Vtrigprevexpr___TOP__tb_d5_mobilenet_05_xsim__DOT__clk__0 
        = vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__clk;
    if ((1ULL & __VTmp[0U])) {
        vlSelfRef.__VtrigSched_hc3011dec__0.ready(__VeventDescription);
        vlSelfRef.__VtrigSched_hc3011dec__0.ready(__VeventDescription);
        vlSelfRef.__VtrigSched_hc3011dec__0.ready(__VeventDescription);
        vlSelfRef.__VtrigSched_hc3011dec__0.ready(__VeventDescription);
    }
    if ((4ULL & __VTmp[0U])) {
        vlSelfRef.__VtrigSched_hc3011a3e__0.ready(__VeventDescription);
        vlSelfRef.__VtrigSched_hc3011a3e__0.ready(__VeventDescription);
        vlSelfRef.__VtrigSched_hc3011a3e__0.ready(__VeventDescription);
    }
    vlSelfRef.__VactTriggeredAcc[0U] = (vlSelfRef.__VactTriggeredAcc[0U] 
                                        | __VTmp[0U]);
}

#ifdef VL_DEBUG
void Vtb_d5_mobilenet_05_xsim___024root___eval_debug_assertions(Vtb_d5_mobilenet_05_xsim___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtb_d5_mobilenet_05_xsim___024root___eval_debug_assertions\n"); );
    Vtb_d5_mobilenet_05_xsim__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
}
#endif  // VL_DEBUG
