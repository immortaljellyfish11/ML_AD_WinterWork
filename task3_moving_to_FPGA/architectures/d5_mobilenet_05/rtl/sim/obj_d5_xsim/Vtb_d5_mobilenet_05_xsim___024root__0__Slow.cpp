// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design implementation internals
// See Vtb_d5_mobilenet_05_xsim.h for the primary calling header

#include "Vtb_d5_mobilenet_05_xsim__pch.h"

void Vtb_d5_mobilenet_05_xsim___024root___timing_ready(Vtb_d5_mobilenet_05_xsim___024root* vlSelf);

VL_ATTR_COLD void Vtb_d5_mobilenet_05_xsim___024root___eval_static(Vtb_d5_mobilenet_05_xsim___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtb_d5_mobilenet_05_xsim___024root___eval_static\n"); );
    Vtb_d5_mobilenet_05_xsim__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__clk = 0U;
    vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__rst_n = 0U;
    vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__start = 0U;
    vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__layer_done = 0U;
    vlSelfRef.__Vtrigprevexpr___TOP__tb_d5_mobilenet_05_xsim__DOT__clk__0 = 0U;
    vlSelfRef.__Vtrigprevexpr___TOP__tb_d5_mobilenet_05_xsim__DOT__rst_n__0 = 0U;
    Vtb_d5_mobilenet_05_xsim___024root___timing_ready(vlSelf);
    do {
        vlSelfRef.__VactTriggeredAcc[vlSelfRef.__Vi] 
            = vlSelfRef.__VactTriggered[vlSelfRef.__Vi];
        vlSelfRef.__Vi = ((IData)(1U) + vlSelfRef.__Vi);
    } while ((0U >= vlSelfRef.__Vi));
}

VL_ATTR_COLD void Vtb_d5_mobilenet_05_xsim___024root___eval_static__TOP(Vtb_d5_mobilenet_05_xsim___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtb_d5_mobilenet_05_xsim___024root___eval_static__TOP\n"); );
    Vtb_d5_mobilenet_05_xsim__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__clk = 0U;
    vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__rst_n = 0U;
    vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__start = 0U;
    vlSelfRef.tb_d5_mobilenet_05_xsim__DOT__layer_done = 0U;
}

VL_ATTR_COLD void Vtb_d5_mobilenet_05_xsim___024root___eval_final(Vtb_d5_mobilenet_05_xsim___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtb_d5_mobilenet_05_xsim___024root___eval_final\n"); );
    Vtb_d5_mobilenet_05_xsim__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
}

VL_ATTR_COLD void Vtb_d5_mobilenet_05_xsim___024root___eval_settle(Vtb_d5_mobilenet_05_xsim___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtb_d5_mobilenet_05_xsim___024root___eval_settle\n"); );
    Vtb_d5_mobilenet_05_xsim__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
}

bool Vtb_d5_mobilenet_05_xsim___024root___trigger_anySet__act(const VlUnpacked<QData/*63:0*/, 1> &in);

#ifdef VL_DEBUG
VL_ATTR_COLD void Vtb_d5_mobilenet_05_xsim___024root___dump_triggers__act(const VlUnpacked<QData/*63:0*/, 1> &triggers, const std::string &tag) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtb_d5_mobilenet_05_xsim___024root___dump_triggers__act\n"); );
    // Body
    if ((1U & (~ (IData)(Vtb_d5_mobilenet_05_xsim___024root___trigger_anySet__act(triggers))))) {
        VL_DBG_MSGS("         No '" + tag + "' region triggers active\n");
    }
    if ((1U & (IData)(triggers[0U]))) {
        VL_DBG_MSGS("         '" + tag + "' region trigger index 0 is active: @(posedge tb_d5_mobilenet_05_xsim.clk)\n");
    }
    if ((1U & (IData)((triggers[0U] >> 1U)))) {
        VL_DBG_MSGS("         '" + tag + "' region trigger index 1 is active: @(negedge tb_d5_mobilenet_05_xsim.rst_n)\n");
    }
    if ((1U & (IData)((triggers[0U] >> 2U)))) {
        VL_DBG_MSGS("         '" + tag + "' region trigger index 2 is active: @(negedge tb_d5_mobilenet_05_xsim.clk)\n");
    }
    if ((1U & (IData)((triggers[0U] >> 3U)))) {
        VL_DBG_MSGS("         '" + tag + "' region trigger index 3 is active: @([true] __VdlySched.awaitingCurrentTime())\n");
    }
}
#endif  // VL_DEBUG

VL_ATTR_COLD void Vtb_d5_mobilenet_05_xsim___024root___ctor_var_reset(Vtb_d5_mobilenet_05_xsim___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtb_d5_mobilenet_05_xsim___024root___ctor_var_reset\n"); );
    Vtb_d5_mobilenet_05_xsim__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    const uint64_t __VscopeHash = VL_MURMUR64_HASH(vlSelf->vlNamep);
    vlSelf->tb_d5_mobilenet_05_xsim__DOT__layer_start = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 3198987873753999182ull);
    vlSelf->tb_d5_mobilenet_05_xsim__DOT__layer_id = VL_SCOPED_RAND_RESET_I(5, __VscopeHash, 6869739829733951982ull);
    vlSelf->tb_d5_mobilenet_05_xsim__DOT__op_kind = VL_SCOPED_RAND_RESET_I(3, __VscopeHash, 16891811493591291800ull);
    vlSelf->tb_d5_mobilenet_05_xsim__DOT__in_channels = VL_SCOPED_RAND_RESET_I(10, __VscopeHash, 11473835228418432830ull);
    vlSelf->tb_d5_mobilenet_05_xsim__DOT__out_channels = VL_SCOPED_RAND_RESET_I(10, __VscopeHash, 12932118970096586951ull);
    vlSelf->tb_d5_mobilenet_05_xsim__DOT__height = VL_SCOPED_RAND_RESET_I(6, __VscopeHash, 10843551956785783300ull);
    vlSelf->tb_d5_mobilenet_05_xsim__DOT__width = VL_SCOPED_RAND_RESET_I(6, __VscopeHash, 10219011098797252306ull);
    vlSelf->tb_d5_mobilenet_05_xsim__DOT__stride = VL_SCOPED_RAND_RESET_I(2, __VscopeHash, 1304463343993415128ull);
    vlSelf->tb_d5_mobilenet_05_xsim__DOT__weight_offset = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 7106845653478427173ull);
    vlSelf->tb_d5_mobilenet_05_xsim__DOT__busy = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 7249592644341074963ull);
    vlSelf->tb_d5_mobilenet_05_xsim__DOT__done = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 12233505839935534978ull);
    vlSelf->tb_d5_mobilenet_05_xsim__DOT__dut__DOT__active = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 9627629093793569303ull);
    for (int __Vi0 = 0; __Vi0 < 1; ++__Vi0) {
        vlSelf->__VactTriggered[__Vi0] = 0;
    }
    for (int __Vi0 = 0; __Vi0 < 1; ++__Vi0) {
        vlSelf->__VactTriggeredAcc[__Vi0] = 0;
    }
    vlSelf->__Vtrigprevexpr___TOP__tb_d5_mobilenet_05_xsim__DOT__clk__0 = 0;
    vlSelf->__Vtrigprevexpr___TOP__tb_d5_mobilenet_05_xsim__DOT__rst_n__0 = 0;
    for (int __Vi0 = 0; __Vi0 < 1; ++__Vi0) {
        vlSelf->__VnbaTriggered[__Vi0] = 0;
    }
    vlSelf->__Vi = 0;
}
