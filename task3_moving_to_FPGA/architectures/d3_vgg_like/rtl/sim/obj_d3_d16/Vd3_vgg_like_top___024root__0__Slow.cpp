// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design implementation internals
// See Vd3_vgg_like_top.h for the primary calling header

#include "Vd3_vgg_like_top__pch.h"

VL_ATTR_COLD void Vd3_vgg_like_top___024root___eval_static(Vd3_vgg_like_top___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vd3_vgg_like_top___024root___eval_static\n"); );
    Vd3_vgg_like_top__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    vlSelfRef.__Vtrigprevexpr___TOP__clk__0 = vlSelfRef.clk;
    vlSelfRef.__Vtrigprevexpr___TOP__rst_n__0 = vlSelfRef.rst_n;
}

VL_ATTR_COLD void Vd3_vgg_like_top___024root___eval_initial(Vd3_vgg_like_top___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vd3_vgg_like_top___024root___eval_initial\n"); );
    Vd3_vgg_like_top__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Locals
    IData/*31:0*/ __Vinline__eval_initial__TOP_d3_vgg_like_top__DOT__rw4__DOT__index;
    __Vinline__eval_initial__TOP_d3_vgg_like_top__DOT__rw4__DOT__index = 0;
    // Body
    VL_READMEM_N(true, 8, 432, 0, "artifacts/int8_params/conv1_weight.mem"s
                 ,  &(vlSelfRef.d3_vgg_like_top__DOT__rw1__DOT__rom)
                 , 0, ~0ULL);
    VL_READMEM_N(true, 8, 4608, 0, "artifacts/int8_params/conv2_weight.mem"s
                 ,  &(vlSelfRef.d3_vgg_like_top__DOT__rw2__DOT__rom)
                 , 0, ~0ULL);
    VL_READMEM_N(true, 8, 18432, 0, "artifacts/int8_params/conv3_weight.mem"s
                 ,  &(vlSelfRef.d3_vgg_like_top__DOT__rw3__DOT__rom)
                 , 0, ~0ULL);
    __Vinline__eval_initial__TOP_d3_vgg_like_top__DOT__rw4__DOT__index = 0U;
    while (VL_GTS_III(32, 0x00012000U, __Vinline__eval_initial__TOP_d3_vgg_like_top__DOT__rw4__DOT__index)) {
        vlSelfRef.d3_vgg_like_top__DOT__rw4__DOT____Vlvbound_h1d1de31c__0 = 0U;
        if (VL_LIKELY(((0x00011fffU >= (0x0001ffffU 
                                        & __Vinline__eval_initial__TOP_d3_vgg_like_top__DOT__rw4__DOT__index))))) {
            vlSelfRef.d3_vgg_like_top__DOT__rw4__DOT__rom[(0x0001ffffU 
                                                           & __Vinline__eval_initial__TOP_d3_vgg_like_top__DOT__rw4__DOT__index)] 
                = vlSelfRef.d3_vgg_like_top__DOT__rw4__DOT____Vlvbound_h1d1de31c__0;
        }
        __Vinline__eval_initial__TOP_d3_vgg_like_top__DOT__rw4__DOT__index 
            = ((IData)(1U) + __Vinline__eval_initial__TOP_d3_vgg_like_top__DOT__rw4__DOT__index);
    }
    VL_READMEM_N(true, 8, 73728, 0, "artifacts/int8_params/conv4_weight.mem"s
                 ,  &(vlSelfRef.d3_vgg_like_top__DOT__rw4__DOT__rom)
                 , 0, ~0ULL);
    VL_READMEM_N(true, 8, 1280, 0, "artifacts/int8_params/linear_weight.mem"s
                 ,  &(vlSelfRef.d3_vgg_like_top__DOT__rwf__DOT__rom)
                 , 0, ~0ULL);
    VL_READMEM_N(true, 32, 16, 0, "artifacts/int8_params/conv1_bias.mem"s
                 ,  &(vlSelfRef.d3_vgg_like_top__DOT__rb1__DOT__rom)
                 , 0, ~0ULL);
    VL_READMEM_N(true, 32, 32, 0, "artifacts/int8_params/conv2_bias.mem"s
                 ,  &(vlSelfRef.d3_vgg_like_top__DOT__rb2__DOT__rom)
                 , 0, ~0ULL);
    VL_READMEM_N(true, 32, 64, 0, "artifacts/int8_params/conv3_bias.mem"s
                 ,  &(vlSelfRef.d3_vgg_like_top__DOT__rb3__DOT__rom)
                 , 0, ~0ULL);
    VL_READMEM_N(true, 32, 128, 0, "artifacts/int8_params/conv4_bias.mem"s
                 ,  &(vlSelfRef.d3_vgg_like_top__DOT__rb4__DOT__rom)
                 , 0, ~0ULL);
    VL_READMEM_N(true, 32, 10, 0, "artifacts/int8_params/linear_bias.mem"s
                 ,  &(vlSelfRef.d3_vgg_like_top__DOT__rbf__DOT__rom)
                 , 0, ~0ULL);
}

VL_ATTR_COLD void Vd3_vgg_like_top___024root___eval_initial__TOP(Vd3_vgg_like_top___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vd3_vgg_like_top___024root___eval_initial__TOP\n"); );
    Vd3_vgg_like_top__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Locals
    IData/*31:0*/ d3_vgg_like_top__DOT__rw4__DOT__index;
    d3_vgg_like_top__DOT__rw4__DOT__index = 0;
    // Body
    VL_READMEM_N(true, 8, 432, 0, "artifacts/int8_params/conv1_weight.mem"s
                 ,  &(vlSelfRef.d3_vgg_like_top__DOT__rw1__DOT__rom)
                 , 0, ~0ULL);
    VL_READMEM_N(true, 8, 4608, 0, "artifacts/int8_params/conv2_weight.mem"s
                 ,  &(vlSelfRef.d3_vgg_like_top__DOT__rw2__DOT__rom)
                 , 0, ~0ULL);
    VL_READMEM_N(true, 8, 18432, 0, "artifacts/int8_params/conv3_weight.mem"s
                 ,  &(vlSelfRef.d3_vgg_like_top__DOT__rw3__DOT__rom)
                 , 0, ~0ULL);
    d3_vgg_like_top__DOT__rw4__DOT__index = 0U;
    while (VL_GTS_III(32, 0x00012000U, d3_vgg_like_top__DOT__rw4__DOT__index)) {
        vlSelfRef.d3_vgg_like_top__DOT__rw4__DOT____Vlvbound_h1d1de31c__0 = 0U;
        if (VL_LIKELY(((0x00011fffU >= (0x0001ffffU 
                                        & d3_vgg_like_top__DOT__rw4__DOT__index))))) {
            vlSelfRef.d3_vgg_like_top__DOT__rw4__DOT__rom[(0x0001ffffU 
                                                           & d3_vgg_like_top__DOT__rw4__DOT__index)] 
                = vlSelfRef.d3_vgg_like_top__DOT__rw4__DOT____Vlvbound_h1d1de31c__0;
        }
        d3_vgg_like_top__DOT__rw4__DOT__index = ((IData)(1U) 
                                                 + d3_vgg_like_top__DOT__rw4__DOT__index);
    }
    VL_READMEM_N(true, 8, 73728, 0, "artifacts/int8_params/conv4_weight.mem"s
                 ,  &(vlSelfRef.d3_vgg_like_top__DOT__rw4__DOT__rom)
                 , 0, ~0ULL);
    VL_READMEM_N(true, 8, 1280, 0, "artifacts/int8_params/linear_weight.mem"s
                 ,  &(vlSelfRef.d3_vgg_like_top__DOT__rwf__DOT__rom)
                 , 0, ~0ULL);
    VL_READMEM_N(true, 32, 16, 0, "artifacts/int8_params/conv1_bias.mem"s
                 ,  &(vlSelfRef.d3_vgg_like_top__DOT__rb1__DOT__rom)
                 , 0, ~0ULL);
    VL_READMEM_N(true, 32, 32, 0, "artifacts/int8_params/conv2_bias.mem"s
                 ,  &(vlSelfRef.d3_vgg_like_top__DOT__rb2__DOT__rom)
                 , 0, ~0ULL);
    VL_READMEM_N(true, 32, 64, 0, "artifacts/int8_params/conv3_bias.mem"s
                 ,  &(vlSelfRef.d3_vgg_like_top__DOT__rb3__DOT__rom)
                 , 0, ~0ULL);
    VL_READMEM_N(true, 32, 128, 0, "artifacts/int8_params/conv4_bias.mem"s
                 ,  &(vlSelfRef.d3_vgg_like_top__DOT__rb4__DOT__rom)
                 , 0, ~0ULL);
    VL_READMEM_N(true, 32, 10, 0, "artifacts/int8_params/linear_bias.mem"s
                 ,  &(vlSelfRef.d3_vgg_like_top__DOT__rbf__DOT__rom)
                 , 0, ~0ULL);
}

VL_ATTR_COLD void Vd3_vgg_like_top___024root___eval_final(Vd3_vgg_like_top___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vd3_vgg_like_top___024root___eval_final\n"); );
    Vd3_vgg_like_top__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
}

#ifdef VL_DEBUG
VL_ATTR_COLD void Vd3_vgg_like_top___024root___dump_triggers__stl(const VlUnpacked<QData/*63:0*/, 1> &triggers, const std::string &tag);
#endif  // VL_DEBUG
VL_ATTR_COLD bool Vd3_vgg_like_top___024root___eval_phase__stl(Vd3_vgg_like_top___024root* vlSelf);

VL_ATTR_COLD void Vd3_vgg_like_top___024root___eval_settle(Vd3_vgg_like_top___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vd3_vgg_like_top___024root___eval_settle\n"); );
    Vd3_vgg_like_top__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Locals
    IData/*31:0*/ __VstlIterCount;
    // Body
    __VstlIterCount = 0U;
    vlSelfRef.__VstlFirstIteration = 1U;
    do {
        if (VL_UNLIKELY(((0x00002710U < __VstlIterCount)))) {
#ifdef VL_DEBUG
            Vd3_vgg_like_top___024root___dump_triggers__stl(vlSelfRef.__VstlTriggered, "stl"s);
#endif
            VL_FATAL_MT("rtl/src/d3_vgg_like_top.v", 4, "", "DIDNOTCONVERGE: Settle region did not converge after '--converge-limit' of 10000 tries");
        }
        __VstlIterCount = ((IData)(1U) + __VstlIterCount);
        vlSelfRef.__VstlPhaseResult = Vd3_vgg_like_top___024root___eval_phase__stl(vlSelf);
        vlSelfRef.__VstlFirstIteration = 0U;
    } while (vlSelfRef.__VstlPhaseResult);
}

VL_ATTR_COLD void Vd3_vgg_like_top___024root___eval_triggers_vec__stl(Vd3_vgg_like_top___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vd3_vgg_like_top___024root___eval_triggers_vec__stl\n"); );
    Vd3_vgg_like_top__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    vlSelfRef.__VstlTriggered[0U] = ((0xfffffffffffffffeULL 
                                      & vlSelfRef.__VstlTriggered[0U]) 
                                     | (IData)((IData)(vlSelfRef.__VstlFirstIteration)));
}

VL_ATTR_COLD bool Vd3_vgg_like_top___024root___trigger_anySet__stl(const VlUnpacked<QData/*63:0*/, 1> &in);

#ifdef VL_DEBUG
VL_ATTR_COLD void Vd3_vgg_like_top___024root___dump_triggers__stl(const VlUnpacked<QData/*63:0*/, 1> &triggers, const std::string &tag) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vd3_vgg_like_top___024root___dump_triggers__stl\n"); );
    // Body
    if ((1U & (~ (IData)(Vd3_vgg_like_top___024root___trigger_anySet__stl(triggers))))) {
        VL_DBG_MSGS("         No '" + tag + "' region triggers active\n");
    }
    if ((1U & (IData)(triggers[0U]))) {
        VL_DBG_MSGS("         '" + tag + "' region trigger index 0 is active: Internal 'stl' trigger - first iteration\n");
    }
}
#endif  // VL_DEBUG

VL_ATTR_COLD bool Vd3_vgg_like_top___024root___trigger_anySet__stl(const VlUnpacked<QData/*63:0*/, 1> &in) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vd3_vgg_like_top___024root___trigger_anySet__stl\n"); );
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

VL_ATTR_COLD void Vd3_vgg_like_top___024root___stl_sequent__TOP__0(Vd3_vgg_like_top___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vd3_vgg_like_top___024root___stl_sequent__TOP__0\n"); );
    Vd3_vgg_like_top__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Locals
    QData/*42:0*/ d3_vgg_like_top__DOT__conv1__DOT__quant__DOT__result43;
    d3_vgg_like_top__DOT__conv1__DOT__quant__DOT__result43 = 0;
    QData/*42:0*/ d3_vgg_like_top__DOT__conv1__DOT__quant__DOT__gen_round_shift__DOT__shifted_magnitude;
    d3_vgg_like_top__DOT__conv1__DOT__quant__DOT__gen_round_shift__DOT__shifted_magnitude = 0;
    QData/*42:0*/ d3_vgg_like_top__DOT__conv2__DOT__quant__DOT__result43;
    d3_vgg_like_top__DOT__conv2__DOT__quant__DOT__result43 = 0;
    QData/*42:0*/ d3_vgg_like_top__DOT__conv2__DOT__quant__DOT__gen_round_shift__DOT__shifted_magnitude;
    d3_vgg_like_top__DOT__conv2__DOT__quant__DOT__gen_round_shift__DOT__shifted_magnitude = 0;
    QData/*42:0*/ d3_vgg_like_top__DOT__conv3__DOT__quant__DOT__result43;
    d3_vgg_like_top__DOT__conv3__DOT__quant__DOT__result43 = 0;
    QData/*42:0*/ d3_vgg_like_top__DOT__conv3__DOT__quant__DOT__gen_round_shift__DOT__shifted_magnitude;
    d3_vgg_like_top__DOT__conv3__DOT__quant__DOT__gen_round_shift__DOT__shifted_magnitude = 0;
    QData/*42:0*/ d3_vgg_like_top__DOT__conv4__DOT__quant__DOT__result43;
    d3_vgg_like_top__DOT__conv4__DOT__quant__DOT__result43 = 0;
    QData/*42:0*/ d3_vgg_like_top__DOT__conv4__DOT__quant__DOT__gen_round_shift__DOT__shifted_magnitude;
    d3_vgg_like_top__DOT__conv4__DOT__quant__DOT__gen_round_shift__DOT__shifted_magnitude = 0;
    // Body
    vlSelfRef.d3_vgg_like_top__DOT__rwf__DOT__rom_addr 
        = (0x000007ffU & ((IData)(vlSelfRef.d3_vgg_like_top__DOT__linear__DOT__feature_idx) 
                          + ((IData)(vlSelfRef.d3_vgg_like_top__DOT__linear__DOT__class_idx) 
                             << 7U)));
    vlSelfRef.d3_vgg_like_top__DOT__conv1__DOT__engine__DOT__last_item 
        = ((2U == (IData)(vlSelfRef.d3_vgg_like_top__DOT__conv1__DOT__engine__DOT__ic)) 
           & ((2U == (IData)(vlSelfRef.d3_vgg_like_top__DOT__conv1__DOT__engine__DOT__kr)) 
              & (2U == (IData)(vlSelfRef.d3_vgg_like_top__DOT__conv1__DOT__engine__DOT__kc))));
    vlSelfRef.d3_vgg_like_top__DOT__conv2__DOT__engine__DOT__last_item 
        = ((0x000fU == (IData)(vlSelfRef.d3_vgg_like_top__DOT__conv2__DOT__engine__DOT__ic)) 
           & ((2U == (IData)(vlSelfRef.d3_vgg_like_top__DOT__conv2__DOT__engine__DOT__kr)) 
              & (2U == (IData)(vlSelfRef.d3_vgg_like_top__DOT__conv2__DOT__engine__DOT__kc))));
    vlSelfRef.d3_vgg_like_top__DOT__conv3__DOT__engine__DOT__last_item 
        = ((0x001fU == (IData)(vlSelfRef.d3_vgg_like_top__DOT__conv3__DOT__engine__DOT__ic)) 
           & ((2U == (IData)(vlSelfRef.d3_vgg_like_top__DOT__conv3__DOT__engine__DOT__kr)) 
              & (2U == (IData)(vlSelfRef.d3_vgg_like_top__DOT__conv3__DOT__engine__DOT__kc))));
    vlSelfRef.d3_vgg_like_top__DOT__conv4__DOT__engine__DOT__last_item 
        = ((0x003fU == (IData)(vlSelfRef.d3_vgg_like_top__DOT__conv4__DOT__engine__DOT__ic)) 
           & ((2U == (IData)(vlSelfRef.d3_vgg_like_top__DOT__conv4__DOT__engine__DOT__kr)) 
              & (2U == (IData)(vlSelfRef.d3_vgg_like_top__DOT__conv4__DOT__engine__DOT__kc))));
    vlSelfRef.d3_vgg_like_top__DOT__c4_wa = (0x0001ffffU 
                                             & ((IData)(vlSelfRef.d3_vgg_like_top__DOT__conv4__DOT__engine__DOT__kc) 
                                                + (
                                                   ((IData)(9U) 
                                                    * 
                                                    VL_SHIFTL_III(32,32,32, (IData)(vlSelfRef.d3_vgg_like_top__DOT__conv4__DOT__out_channel), 6U)) 
                                                   + 
                                                   (((IData)(9U) 
                                                     * (IData)(vlSelfRef.d3_vgg_like_top__DOT__conv4__DOT__engine__DOT__ic)) 
                                                    + 
                                                    ((IData)(3U) 
                                                     * (IData)(vlSelfRef.d3_vgg_like_top__DOT__conv4__DOT__engine__DOT__kr))))));
    vlSelfRef.d3_vgg_like_top__DOT__rw1__DOT__rom_addr 
        = (0x000001ffU & ((IData)(vlSelfRef.d3_vgg_like_top__DOT__conv1__DOT__engine__DOT__kc) 
                          + (((IData)(0x0000001bU) 
                              * (IData)(vlSelfRef.d3_vgg_like_top__DOT__conv1__DOT__out_channel)) 
                             + (((IData)(9U) * (IData)(vlSelfRef.d3_vgg_like_top__DOT__conv1__DOT__engine__DOT__ic)) 
                                + ((IData)(3U) * (IData)(vlSelfRef.d3_vgg_like_top__DOT__conv1__DOT__engine__DOT__kr))))));
    vlSelfRef.d3_vgg_like_top__DOT__rw2__DOT__rom_addr 
        = (0x00001fffU & ((IData)(vlSelfRef.d3_vgg_like_top__DOT__conv2__DOT__engine__DOT__kc) 
                          + (((IData)(9U) * VL_SHIFTL_III(32,32,32, (IData)(vlSelfRef.d3_vgg_like_top__DOT__conv2__DOT__out_channel), 4U)) 
                             + (((IData)(9U) * (IData)(vlSelfRef.d3_vgg_like_top__DOT__conv2__DOT__engine__DOT__ic)) 
                                + ((IData)(3U) * (IData)(vlSelfRef.d3_vgg_like_top__DOT__conv2__DOT__engine__DOT__kr))))));
    vlSelfRef.d3_vgg_like_top__DOT__rw3__DOT__rom_addr 
        = (0x00007fffU & ((IData)(vlSelfRef.d3_vgg_like_top__DOT__conv3__DOT__engine__DOT__kc) 
                          + (((IData)(9U) * VL_SHIFTL_III(32,32,32, (IData)(vlSelfRef.d3_vgg_like_top__DOT__conv3__DOT__out_channel), 5U)) 
                             + (((IData)(9U) * (IData)(vlSelfRef.d3_vgg_like_top__DOT__conv3__DOT__engine__DOT__ic)) 
                                + ((IData)(3U) * (IData)(vlSelfRef.d3_vgg_like_top__DOT__conv3__DOT__engine__DOT__kr))))));
    vlSelfRef.__VdfgRegularize_h6e95ff9d_0_1 = (((IData)(vlSelfRef.d3_vgg_like_top__DOT__conv1__DOT__engine__DOT__kr) 
                                                 + (IData)(vlSelfRef.d3_vgg_like_top__DOT__conv1__DOT__out_row)) 
                                                - (IData)(1U));
    vlSelfRef.__VdfgRegularize_h6e95ff9d_0_2 = (((IData)(vlSelfRef.d3_vgg_like_top__DOT__conv1__DOT__engine__DOT__kc) 
                                                 + (IData)(vlSelfRef.d3_vgg_like_top__DOT__conv1__DOT__out_col)) 
                                                - (IData)(1U));
    vlSelfRef.__VdfgRegularize_h6e95ff9d_0_3 = (((IData)(vlSelfRef.d3_vgg_like_top__DOT__conv2__DOT__engine__DOT__kr) 
                                                 + (IData)(vlSelfRef.d3_vgg_like_top__DOT__conv2__DOT__out_row)) 
                                                - (IData)(1U));
    vlSelfRef.__VdfgRegularize_h6e95ff9d_0_4 = (((IData)(vlSelfRef.d3_vgg_like_top__DOT__conv2__DOT__engine__DOT__kc) 
                                                 + (IData)(vlSelfRef.d3_vgg_like_top__DOT__conv2__DOT__out_col)) 
                                                - (IData)(1U));
    vlSelfRef.__VdfgRegularize_h6e95ff9d_0_5 = (((IData)(vlSelfRef.d3_vgg_like_top__DOT__conv3__DOT__engine__DOT__kr) 
                                                 + (IData)(vlSelfRef.d3_vgg_like_top__DOT__conv3__DOT__out_row)) 
                                                - (IData)(1U));
    vlSelfRef.__VdfgRegularize_h6e95ff9d_0_6 = (((IData)(vlSelfRef.d3_vgg_like_top__DOT__conv3__DOT__engine__DOT__kc) 
                                                 + (IData)(vlSelfRef.d3_vgg_like_top__DOT__conv3__DOT__out_col)) 
                                                - (IData)(1U));
    vlSelfRef.__VdfgRegularize_h6e95ff9d_0_7 = (((IData)(vlSelfRef.d3_vgg_like_top__DOT__conv4__DOT__engine__DOT__kr) 
                                                 + (IData)(vlSelfRef.d3_vgg_like_top__DOT__conv4__DOT__out_row)) 
                                                - (IData)(1U));
    vlSelfRef.__VdfgRegularize_h6e95ff9d_0_8 = (((IData)(vlSelfRef.d3_vgg_like_top__DOT__conv4__DOT__engine__DOT__kc) 
                                                 + (IData)(vlSelfRef.d3_vgg_like_top__DOT__conv4__DOT__out_col)) 
                                                - (IData)(1U));
    d3_vgg_like_top__DOT__conv1__DOT__quant__DOT__gen_round_shift__DOT__shifted_magnitude 
        = (0x000007ffffffffffULL & VL_SHIFTRS_QQI(43,43,32, 
                                                  (0x000007ffffffffffULL 
                                                   & ((1U 
                                                       & (IData)(
                                                                 (vlSelfRef.d3_vgg_like_top__DOT__conv1__DOT__product_reg 
                                                                  >> 0x0000002aU)))
                                                       ? 
                                                      (0x0000000000800000ULL 
                                                       + 
                                                       (- vlSelfRef.d3_vgg_like_top__DOT__conv1__DOT__product_reg))
                                                       : 
                                                      (0x0000000000800000ULL 
                                                       + vlSelfRef.d3_vgg_like_top__DOT__conv1__DOT__product_reg))), 0x00000018U));
    d3_vgg_like_top__DOT__conv2__DOT__quant__DOT__gen_round_shift__DOT__shifted_magnitude 
        = (0x000007ffffffffffULL & VL_SHIFTRS_QQI(43,43,32, 
                                                  (0x000007ffffffffffULL 
                                                   & ((1U 
                                                       & (IData)(
                                                                 (vlSelfRef.d3_vgg_like_top__DOT__conv2__DOT__product_reg 
                                                                  >> 0x0000002aU)))
                                                       ? 
                                                      (0x0000000000800000ULL 
                                                       + 
                                                       (- vlSelfRef.d3_vgg_like_top__DOT__conv2__DOT__product_reg))
                                                       : 
                                                      (0x0000000000800000ULL 
                                                       + vlSelfRef.d3_vgg_like_top__DOT__conv2__DOT__product_reg))), 0x00000018U));
    d3_vgg_like_top__DOT__conv3__DOT__quant__DOT__gen_round_shift__DOT__shifted_magnitude 
        = (0x000007ffffffffffULL & VL_SHIFTRS_QQI(43,43,32, 
                                                  (0x000007ffffffffffULL 
                                                   & ((1U 
                                                       & (IData)(
                                                                 (vlSelfRef.d3_vgg_like_top__DOT__conv3__DOT__product_reg 
                                                                  >> 0x0000002aU)))
                                                       ? 
                                                      (0x0000000000800000ULL 
                                                       + 
                                                       (- vlSelfRef.d3_vgg_like_top__DOT__conv3__DOT__product_reg))
                                                       : 
                                                      (0x0000000000800000ULL 
                                                       + vlSelfRef.d3_vgg_like_top__DOT__conv3__DOT__product_reg))), 0x00000018U));
    d3_vgg_like_top__DOT__conv4__DOT__quant__DOT__gen_round_shift__DOT__shifted_magnitude 
        = (0x000007ffffffffffULL & VL_SHIFTRS_QQI(43,43,32, 
                                                  (0x000007ffffffffffULL 
                                                   & ((1U 
                                                       & (IData)(
                                                                 (vlSelfRef.d3_vgg_like_top__DOT__conv4__DOT__product_reg 
                                                                  >> 0x0000002aU)))
                                                       ? 
                                                      (0x0000000000400000ULL 
                                                       + 
                                                       (- vlSelfRef.d3_vgg_like_top__DOT__conv4__DOT__product_reg))
                                                       : 
                                                      (0x0000000000400000ULL 
                                                       + vlSelfRef.d3_vgg_like_top__DOT__conv4__DOT__product_reg))), 0x00000017U));
    vlSelfRef.class_id = 0U;
    vlSelfRef.max_logit = vlSelfRef.d3_vgg_like_top__DOT__l0;
    if (VL_GTS_III(32, vlSelfRef.d3_vgg_like_top__DOT__l1, vlSelfRef.max_logit)) {
        vlSelfRef.max_logit = vlSelfRef.d3_vgg_like_top__DOT__l1;
        vlSelfRef.class_id = 1U;
    }
    if (VL_GTS_III(32, vlSelfRef.d3_vgg_like_top__DOT__l2, vlSelfRef.max_logit)) {
        vlSelfRef.max_logit = vlSelfRef.d3_vgg_like_top__DOT__l2;
        vlSelfRef.class_id = 2U;
    }
    if (VL_GTS_III(32, vlSelfRef.d3_vgg_like_top__DOT__l3, vlSelfRef.max_logit)) {
        vlSelfRef.max_logit = vlSelfRef.d3_vgg_like_top__DOT__l3;
        vlSelfRef.class_id = 3U;
    }
    if (VL_GTS_III(32, vlSelfRef.d3_vgg_like_top__DOT__l4, vlSelfRef.max_logit)) {
        vlSelfRef.max_logit = vlSelfRef.d3_vgg_like_top__DOT__l4;
        vlSelfRef.class_id = 4U;
    }
    if (VL_GTS_III(32, vlSelfRef.d3_vgg_like_top__DOT__l5, vlSelfRef.max_logit)) {
        vlSelfRef.max_logit = vlSelfRef.d3_vgg_like_top__DOT__l5;
        vlSelfRef.class_id = 5U;
    }
    if (VL_GTS_III(32, vlSelfRef.d3_vgg_like_top__DOT__l6, vlSelfRef.max_logit)) {
        vlSelfRef.max_logit = vlSelfRef.d3_vgg_like_top__DOT__l6;
        vlSelfRef.class_id = 6U;
    }
    if (VL_GTS_III(32, vlSelfRef.d3_vgg_like_top__DOT__l7, vlSelfRef.max_logit)) {
        vlSelfRef.max_logit = vlSelfRef.d3_vgg_like_top__DOT__l7;
        vlSelfRef.class_id = 7U;
    }
    if (VL_GTS_III(32, vlSelfRef.d3_vgg_like_top__DOT__l8, vlSelfRef.max_logit)) {
        vlSelfRef.max_logit = vlSelfRef.d3_vgg_like_top__DOT__l8;
        vlSelfRef.class_id = 8U;
    }
    if (VL_GTS_III(32, vlSelfRef.d3_vgg_like_top__DOT__l9, vlSelfRef.max_logit)) {
        vlSelfRef.max_logit = vlSelfRef.d3_vgg_like_top__DOT__l9;
        vlSelfRef.class_id = 9U;
    }
    vlSelfRef.__VdfgRegularize_h6e95ff9d_0_0 = ((~ (IData)(vlSelfRef.busy)) 
                                                & ((~ 
                                                    ((IData)(vlSelfRef.d3_vgg_like_top__DOT__conv1__DOT__postprocess_pending) 
                                                     | ((IData)(vlSelfRef.d3_vgg_like_top__DOT__conv1__DOT__product_valid) 
                                                        | ((IData)(vlSelfRef.d3_vgg_like_top__DOT__conv1__DOT__postprocess_valid) 
                                                           | ((IData)(vlSelfRef.d3_vgg_like_top__DOT__conv2__DOT__postprocess_pending) 
                                                              | ((IData)(vlSelfRef.d3_vgg_like_top__DOT__conv2__DOT__product_valid) 
                                                                 | ((IData)(vlSelfRef.d3_vgg_like_top__DOT__conv2__DOT__postprocess_valid) 
                                                                    | ((IData)(vlSelfRef.d3_vgg_like_top__DOT__conv3__DOT__postprocess_pending) 
                                                                       | ((IData)(vlSelfRef.d3_vgg_like_top__DOT__conv3__DOT__product_valid) 
                                                                          | ((IData)(vlSelfRef.d3_vgg_like_top__DOT__conv3__DOT__postprocess_valid) 
                                                                             | ((IData)(vlSelfRef.d3_vgg_like_top__DOT__conv4__DOT__postprocess_pending) 
                                                                                | ((IData)(vlSelfRef.d3_vgg_like_top__DOT__conv4__DOT__product_valid) 
                                                                                | ((IData)(vlSelfRef.d3_vgg_like_top__DOT__conv4__DOT__postprocess_valid) 
                                                                                | ((IData)(vlSelfRef.d3_vgg_like_top__DOT__conv1__DOT__controller_busy) 
                                                                                | ((IData)(vlSelfRef.d3_vgg_like_top__DOT__conv2__DOT__controller_busy) 
                                                                                | ((IData)(vlSelfRef.d3_vgg_like_top__DOT__conv3__DOT__controller_busy) 
                                                                                | ((IData)(vlSelfRef.d3_vgg_like_top__DOT__conv4__DOT__controller_busy) 
                                                                                | ((IData)(vlSelfRef.d3_vgg_like_top__DOT__p1_busy) 
                                                                                | ((IData)(vlSelfRef.d3_vgg_like_top__DOT__p2_busy) 
                                                                                | ((IData)(vlSelfRef.d3_vgg_like_top__DOT__gap_busy) 
                                                                                | (IData)(vlSelfRef.d3_vgg_like_top__DOT__fc_busy))))))))))))))))))))) 
                                                   & (IData)(vlSelfRef.input_we)));
    vlSelfRef.d3_vgg_like_top__DOT__conv1__DOT__engine__DOT__valid_pixel 
        = (VL_LTES_III(32, 0U, vlSelfRef.__VdfgRegularize_h6e95ff9d_0_1) 
           & (VL_GTS_III(32, 0x00000020U, vlSelfRef.__VdfgRegularize_h6e95ff9d_0_1) 
              & (VL_LTES_III(32, 0U, vlSelfRef.__VdfgRegularize_h6e95ff9d_0_2) 
                 & VL_GTS_III(32, 0x00000020U, vlSelfRef.__VdfgRegularize_h6e95ff9d_0_2))));
    vlSelfRef.d3_vgg_like_top__DOT__conv2__DOT__engine__DOT__valid_pixel 
        = (VL_LTES_III(32, 0U, vlSelfRef.__VdfgRegularize_h6e95ff9d_0_3) 
           & (VL_GTS_III(32, 0x00000020U, vlSelfRef.__VdfgRegularize_h6e95ff9d_0_3) 
              & (VL_LTES_III(32, 0U, vlSelfRef.__VdfgRegularize_h6e95ff9d_0_4) 
                 & VL_GTS_III(32, 0x00000020U, vlSelfRef.__VdfgRegularize_h6e95ff9d_0_4))));
    vlSelfRef.d3_vgg_like_top__DOT__conv3__DOT__engine__DOT__valid_pixel 
        = (VL_LTES_III(32, 0U, vlSelfRef.__VdfgRegularize_h6e95ff9d_0_5) 
           & (VL_GTS_III(32, 0x00000010U, vlSelfRef.__VdfgRegularize_h6e95ff9d_0_5) 
              & (VL_LTES_III(32, 0U, vlSelfRef.__VdfgRegularize_h6e95ff9d_0_6) 
                 & VL_GTS_III(32, 0x00000010U, vlSelfRef.__VdfgRegularize_h6e95ff9d_0_6))));
    vlSelfRef.d3_vgg_like_top__DOT__conv4__DOT__engine__DOT__valid_pixel 
        = (VL_LTES_III(32, 0U, vlSelfRef.__VdfgRegularize_h6e95ff9d_0_7) 
           & (VL_GTS_III(32, 0x00000010U, vlSelfRef.__VdfgRegularize_h6e95ff9d_0_7) 
              & (VL_LTES_III(32, 0U, vlSelfRef.__VdfgRegularize_h6e95ff9d_0_8) 
                 & VL_GTS_III(32, 0x00000010U, vlSelfRef.__VdfgRegularize_h6e95ff9d_0_8))));
    d3_vgg_like_top__DOT__conv1__DOT__quant__DOT__result43 
        = (0x000007ffffffffffULL & ((1U & (IData)((vlSelfRef.d3_vgg_like_top__DOT__conv1__DOT__product_reg 
                                                   >> 0x0000002aU)))
                                     ? (- d3_vgg_like_top__DOT__conv1__DOT__quant__DOT__gen_round_shift__DOT__shifted_magnitude)
                                     : d3_vgg_like_top__DOT__conv1__DOT__quant__DOT__gen_round_shift__DOT__shifted_magnitude));
    d3_vgg_like_top__DOT__conv2__DOT__quant__DOT__result43 
        = (0x000007ffffffffffULL & ((1U & (IData)((vlSelfRef.d3_vgg_like_top__DOT__conv2__DOT__product_reg 
                                                   >> 0x0000002aU)))
                                     ? (- d3_vgg_like_top__DOT__conv2__DOT__quant__DOT__gen_round_shift__DOT__shifted_magnitude)
                                     : d3_vgg_like_top__DOT__conv2__DOT__quant__DOT__gen_round_shift__DOT__shifted_magnitude));
    d3_vgg_like_top__DOT__conv3__DOT__quant__DOT__result43 
        = (0x000007ffffffffffULL & ((1U & (IData)((vlSelfRef.d3_vgg_like_top__DOT__conv3__DOT__product_reg 
                                                   >> 0x0000002aU)))
                                     ? (- d3_vgg_like_top__DOT__conv3__DOT__quant__DOT__gen_round_shift__DOT__shifted_magnitude)
                                     : d3_vgg_like_top__DOT__conv3__DOT__quant__DOT__gen_round_shift__DOT__shifted_magnitude));
    d3_vgg_like_top__DOT__conv4__DOT__quant__DOT__result43 
        = (0x000007ffffffffffULL & ((1U & (IData)((vlSelfRef.d3_vgg_like_top__DOT__conv4__DOT__product_reg 
                                                   >> 0x0000002aU)))
                                     ? (- d3_vgg_like_top__DOT__conv4__DOT__quant__DOT__gen_round_shift__DOT__shifted_magnitude)
                                     : d3_vgg_like_top__DOT__conv4__DOT__quant__DOT__gen_round_shift__DOT__shifted_magnitude));
    vlSelfRef.d3_vgg_like_top__DOT__conv1__DOT__activated 
        = ((IData)(d3_vgg_like_top__DOT__conv1__DOT__quant__DOT__result43) 
           & (- (IData)((1U & (~ (IData)((d3_vgg_like_top__DOT__conv1__DOT__quant__DOT__result43 
                                          >> 0x0000001fU)))))));
    vlSelfRef.d3_vgg_like_top__DOT__conv2__DOT__activated 
        = ((IData)(d3_vgg_like_top__DOT__conv2__DOT__quant__DOT__result43) 
           & (- (IData)((1U & (~ (IData)((d3_vgg_like_top__DOT__conv2__DOT__quant__DOT__result43 
                                          >> 0x0000001fU)))))));
    vlSelfRef.d3_vgg_like_top__DOT__conv3__DOT__activated 
        = ((IData)(d3_vgg_like_top__DOT__conv3__DOT__quant__DOT__result43) 
           & (- (IData)((1U & (~ (IData)((d3_vgg_like_top__DOT__conv3__DOT__quant__DOT__result43 
                                          >> 0x0000001fU)))))));
    vlSelfRef.d3_vgg_like_top__DOT__conv4__DOT__activated 
        = ((IData)(d3_vgg_like_top__DOT__conv4__DOT__quant__DOT__result43) 
           & (- (IData)((1U & (~ (IData)((d3_vgg_like_top__DOT__conv4__DOT__quant__DOT__result43 
                                          >> 0x0000001fU)))))));
}

VL_ATTR_COLD void Vd3_vgg_like_top___024root___eval_stl(Vd3_vgg_like_top___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vd3_vgg_like_top___024root___eval_stl\n"); );
    Vd3_vgg_like_top__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    if ((1ULL & vlSelfRef.__VstlTriggered[0U])) {
        Vd3_vgg_like_top___024root___stl_sequent__TOP__0(vlSelf);
    }
}

VL_ATTR_COLD bool Vd3_vgg_like_top___024root___eval_phase__stl(Vd3_vgg_like_top___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vd3_vgg_like_top___024root___eval_phase__stl\n"); );
    Vd3_vgg_like_top__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Locals
    CData/*0:0*/ __VstlExecute;
    // Body
    Vd3_vgg_like_top___024root___eval_triggers_vec__stl(vlSelf);
#ifdef VL_DEBUG
    if (VL_UNLIKELY(vlSymsp->_vm_contextp__->debug())) {
        Vd3_vgg_like_top___024root___dump_triggers__stl(vlSelfRef.__VstlTriggered, "stl"s);
    }
#endif
    __VstlExecute = Vd3_vgg_like_top___024root___trigger_anySet__stl(vlSelfRef.__VstlTriggered);
    if (__VstlExecute) {
        Vd3_vgg_like_top___024root___eval_stl(vlSelf);
    }
    return (__VstlExecute);
}

bool Vd3_vgg_like_top___024root___trigger_anySet__ico(const VlUnpacked<QData/*63:0*/, 1> &in);

#ifdef VL_DEBUG
VL_ATTR_COLD void Vd3_vgg_like_top___024root___dump_triggers__ico(const VlUnpacked<QData/*63:0*/, 1> &triggers, const std::string &tag) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vd3_vgg_like_top___024root___dump_triggers__ico\n"); );
    // Body
    if ((1U & (~ (IData)(Vd3_vgg_like_top___024root___trigger_anySet__ico(triggers))))) {
        VL_DBG_MSGS("         No '" + tag + "' region triggers active\n");
    }
    if ((1U & (IData)(triggers[0U]))) {
        VL_DBG_MSGS("         '" + tag + "' region trigger index 0 is active: Internal 'ico' trigger - first iteration\n");
    }
}
#endif  // VL_DEBUG

bool Vd3_vgg_like_top___024root___trigger_anySet__act(const VlUnpacked<QData/*63:0*/, 1> &in);

#ifdef VL_DEBUG
VL_ATTR_COLD void Vd3_vgg_like_top___024root___dump_triggers__act(const VlUnpacked<QData/*63:0*/, 1> &triggers, const std::string &tag) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vd3_vgg_like_top___024root___dump_triggers__act\n"); );
    // Body
    if ((1U & (~ (IData)(Vd3_vgg_like_top___024root___trigger_anySet__act(triggers))))) {
        VL_DBG_MSGS("         No '" + tag + "' region triggers active\n");
    }
    if ((1U & (IData)(triggers[0U]))) {
        VL_DBG_MSGS("         '" + tag + "' region trigger index 0 is active: @(posedge clk)\n");
    }
    if ((1U & (IData)((triggers[0U] >> 1U)))) {
        VL_DBG_MSGS("         '" + tag + "' region trigger index 1 is active: @(negedge rst_n)\n");
    }
}
#endif  // VL_DEBUG

VL_ATTR_COLD void Vd3_vgg_like_top___024root___ctor_var_reset(Vd3_vgg_like_top___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vd3_vgg_like_top___024root___ctor_var_reset\n"); );
    Vd3_vgg_like_top__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    const uint64_t __VscopeHash = VL_MURMUR64_HASH(vlSelf->vlNamep);
    vlSelf->clk = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 16707436170211756652ull);
    vlSelf->rst_n = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 1638864771569018232ull);
    vlSelf->start = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 9867861323841650631ull);
    vlSelf->input_we = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 7025069179568517235ull);
    vlSelf->input_addr = VL_SCOPED_RAND_RESET_I(17, __VscopeHash, 13892080179392794878ull);
    vlSelf->input_data = VL_SCOPED_RAND_RESET_I(8, __VscopeHash, 1923588759227995539ull);
    vlSelf->busy = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 6386567572483775230ull);
    vlSelf->done = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 10296494685231209730ull);
    vlSelf->class_id = VL_SCOPED_RAND_RESET_I(4, __VscopeHash, 137756330502096589ull);
    vlSelf->max_logit = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 12814312630002674517ull);
    vlSelf->argmax_valid = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 17584079252057304459ull);
    vlSelf->state_dbg = VL_SCOPED_RAND_RESET_I(4, __VscopeHash, 6042213990076480022ull);
    vlSelf->d3_vgg_like_top__DOT__c1_start = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 9169415532392180098ull);
    vlSelf->d3_vgg_like_top__DOT__c2_start = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 7704058902680820283ull);
    vlSelf->d3_vgg_like_top__DOT__p1_start = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 9219442820028040143ull);
    vlSelf->d3_vgg_like_top__DOT__c3_start = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 12968788107485162035ull);
    vlSelf->d3_vgg_like_top__DOT__c4_start = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 17195128085653016124ull);
    vlSelf->d3_vgg_like_top__DOT__p2_start = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 4712066649777132129ull);
    vlSelf->d3_vgg_like_top__DOT__gap_start = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 11457491626067809981ull);
    vlSelf->d3_vgg_like_top__DOT__fc_start = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 7248776351659814445ull);
    vlSelf->d3_vgg_like_top__DOT__p1_done = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 14786669468537012445ull);
    vlSelf->d3_vgg_like_top__DOT__p2_done = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 356097878056338587ull);
    vlSelf->d3_vgg_like_top__DOT__gap_done = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 7835386719450611851ull);
    vlSelf->d3_vgg_like_top__DOT__fc_done = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 7688652398172338301ull);
    vlSelf->d3_vgg_like_top__DOT__p1_busy = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 12704926160227088862ull);
    vlSelf->d3_vgg_like_top__DOT__p2_busy = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 6337282224058531861ull);
    vlSelf->d3_vgg_like_top__DOT__gap_busy = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 4337641819983102152ull);
    vlSelf->d3_vgg_like_top__DOT__fc_busy = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 1484127945347136264ull);
    vlSelf->d3_vgg_like_top__DOT__a_rd = VL_SCOPED_RAND_RESET_I(8, __VscopeHash, 8205619159437346193ull);
    vlSelf->d3_vgg_like_top__DOT__b_rd = VL_SCOPED_RAND_RESET_I(8, __VscopeHash, 7358099708953953124ull);
    vlSelf->d3_vgg_like_top__DOT__c4_wa = VL_SCOPED_RAND_RESET_I(17, __VscopeHash, 14238916523615772523ull);
    vlSelf->d3_vgg_like_top__DOT__p1_oa = VL_SCOPED_RAND_RESET_I(17, __VscopeHash, 3211480710785944060ull);
    vlSelf->d3_vgg_like_top__DOT__p2_oa = VL_SCOPED_RAND_RESET_I(17, __VscopeHash, 7296282535270514816ull);
    vlSelf->d3_vgg_like_top__DOT__gap_oa = VL_SCOPED_RAND_RESET_I(17, __VscopeHash, 10556579304992683822ull);
    vlSelf->d3_vgg_like_top__DOT__p1_owe = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 14559099891664484072ull);
    vlSelf->d3_vgg_like_top__DOT__p2_owe = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 5259051771702740132ull);
    vlSelf->d3_vgg_like_top__DOT__gap_owe = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 17280259441076057870ull);
    vlSelf->d3_vgg_like_top__DOT__p1_od = VL_SCOPED_RAND_RESET_I(8, __VscopeHash, 9480377557826405519ull);
    vlSelf->d3_vgg_like_top__DOT__p2_od = VL_SCOPED_RAND_RESET_I(8, __VscopeHash, 6195966967140421482ull);
    vlSelf->d3_vgg_like_top__DOT__gap_od = VL_SCOPED_RAND_RESET_I(8, __VscopeHash, 6288186071383807303ull);
    vlSelf->d3_vgg_like_top__DOT__logit_index = VL_SCOPED_RAND_RESET_I(4, __VscopeHash, 8255330057723707744ull);
    vlSelf->d3_vgg_like_top__DOT__logit_data = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 13754424066668157123ull);
    vlSelf->d3_vgg_like_top__DOT__logit_valid = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 11872315984554133749ull);
    vlSelf->d3_vgg_like_top__DOT__c1_wd = VL_SCOPED_RAND_RESET_I(8, __VscopeHash, 9974019181173384811ull);
    vlSelf->d3_vgg_like_top__DOT__c2_wd = VL_SCOPED_RAND_RESET_I(8, __VscopeHash, 10565436961673387014ull);
    vlSelf->d3_vgg_like_top__DOT__c3_wd = VL_SCOPED_RAND_RESET_I(8, __VscopeHash, 6585261580981371507ull);
    vlSelf->d3_vgg_like_top__DOT__c4_wd = VL_SCOPED_RAND_RESET_I(8, __VscopeHash, 3486736891571842025ull);
    vlSelf->d3_vgg_like_top__DOT__fc_wd = VL_SCOPED_RAND_RESET_I(8, __VscopeHash, 6445610659605231315ull);
    vlSelf->d3_vgg_like_top__DOT__l0 = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 3801444665722353556ull);
    vlSelf->d3_vgg_like_top__DOT__l1 = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 17032401656923594671ull);
    vlSelf->d3_vgg_like_top__DOT__l2 = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 6114650819972948411ull);
    vlSelf->d3_vgg_like_top__DOT__l3 = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 15151511471790115206ull);
    vlSelf->d3_vgg_like_top__DOT__l4 = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 4651756793264477462ull);
    vlSelf->d3_vgg_like_top__DOT__l5 = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 6899750709844196484ull);
    vlSelf->d3_vgg_like_top__DOT__l6 = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 9877753481501332924ull);
    vlSelf->d3_vgg_like_top__DOT__l7 = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 4996066498149202491ull);
    vlSelf->d3_vgg_like_top__DOT__l8 = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 423246995349361036ull);
    vlSelf->d3_vgg_like_top__DOT__l9 = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 13655688124469846059ull);
    vlSelf->d3_vgg_like_top__DOT__fsm__DOT__state = VL_SCOPED_RAND_RESET_I(4, __VscopeHash, 8625892955550058400ull);
    for (int __Vi0 = 0; __Vi0 < 32768; ++__Vi0) {
        vlSelf->d3_vgg_like_top__DOT__buffer_a__DOT__mem[__Vi0] = VL_SCOPED_RAND_RESET_I(8, __VscopeHash, 3897266144835911151ull);
    }
    for (int __Vi0 = 0; __Vi0 < 32768; ++__Vi0) {
        vlSelf->d3_vgg_like_top__DOT__buffer_b__DOT__mem[__Vi0] = VL_SCOPED_RAND_RESET_I(8, __VscopeHash, 11365504369469241460ull);
    }
    for (int __Vi0 = 0; __Vi0 < 432; ++__Vi0) {
        vlSelf->d3_vgg_like_top__DOT__rw1__DOT__rom[__Vi0] = VL_SCOPED_RAND_RESET_I(8, __VscopeHash, 12987244274730187141ull);
    }
    vlSelf->d3_vgg_like_top__DOT__rw1__DOT__rom_addr = VL_SCOPED_RAND_RESET_I(9, __VscopeHash, 8179324819715535985ull);
    for (int __Vi0 = 0; __Vi0 < 4608; ++__Vi0) {
        vlSelf->d3_vgg_like_top__DOT__rw2__DOT__rom[__Vi0] = VL_SCOPED_RAND_RESET_I(8, __VscopeHash, 1991653480595812295ull);
    }
    vlSelf->d3_vgg_like_top__DOT__rw2__DOT__rom_addr = VL_SCOPED_RAND_RESET_I(13, __VscopeHash, 15228019249655110280ull);
    for (int __Vi0 = 0; __Vi0 < 18432; ++__Vi0) {
        vlSelf->d3_vgg_like_top__DOT__rw3__DOT__rom[__Vi0] = VL_SCOPED_RAND_RESET_I(8, __VscopeHash, 1964571717585661684ull);
    }
    vlSelf->d3_vgg_like_top__DOT__rw3__DOT__rom_addr = VL_SCOPED_RAND_RESET_I(15, __VscopeHash, 795812801219591809ull);
    vlSelf->d3_vgg_like_top__DOT__rw4__DOT____Vlvbound_h1d1de31c__0 = 0;
    for (int __Vi0 = 0; __Vi0 < 73728; ++__Vi0) {
        vlSelf->d3_vgg_like_top__DOT__rw4__DOT__rom[__Vi0] = VL_SCOPED_RAND_RESET_I(8, __VscopeHash, 13841989420209889337ull);
    }
    for (int __Vi0 = 0; __Vi0 < 1280; ++__Vi0) {
        vlSelf->d3_vgg_like_top__DOT__rwf__DOT__rom[__Vi0] = VL_SCOPED_RAND_RESET_I(8, __VscopeHash, 17316075748289458725ull);
    }
    vlSelf->d3_vgg_like_top__DOT__rwf__DOT__rom_addr = VL_SCOPED_RAND_RESET_I(11, __VscopeHash, 8997527802775812333ull);
    for (int __Vi0 = 0; __Vi0 < 16; ++__Vi0) {
        vlSelf->d3_vgg_like_top__DOT__rb1__DOT__rom[__Vi0] = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 13519374966623464308ull);
    }
    for (int __Vi0 = 0; __Vi0 < 32; ++__Vi0) {
        vlSelf->d3_vgg_like_top__DOT__rb2__DOT__rom[__Vi0] = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 8315050854080954738ull);
    }
    for (int __Vi0 = 0; __Vi0 < 64; ++__Vi0) {
        vlSelf->d3_vgg_like_top__DOT__rb3__DOT__rom[__Vi0] = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 10378479566991847110ull);
    }
    for (int __Vi0 = 0; __Vi0 < 128; ++__Vi0) {
        vlSelf->d3_vgg_like_top__DOT__rb4__DOT__rom[__Vi0] = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 3923511801398107976ull);
    }
    for (int __Vi0 = 0; __Vi0 < 10; ++__Vi0) {
        vlSelf->d3_vgg_like_top__DOT__rbf__DOT__rom[__Vi0] = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 1666655966731743571ull);
    }
    vlSelf->d3_vgg_like_top__DOT__conv1__DOT__engine_start = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 6714537588562807718ull);
    vlSelf->d3_vgg_like_top__DOT__conv1__DOT__engine_done = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 4144794512151426619ull);
    vlSelf->d3_vgg_like_top__DOT__conv1__DOT__controller_busy = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 3376355192895250603ull);
    vlSelf->d3_vgg_like_top__DOT__conv1__DOT__controller_done = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 3383790684242834737ull);
    vlSelf->d3_vgg_like_top__DOT__conv1__DOT__raw_sum = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 15528487421017238973ull);
    vlSelf->d3_vgg_like_top__DOT__conv1__DOT__biased_sum_reg = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 6294246008979834012ull);
    vlSelf->d3_vgg_like_top__DOT__conv1__DOT__product_reg = VL_SCOPED_RAND_RESET_Q(43, __VscopeHash, 5820182049913809623ull);
    vlSelf->d3_vgg_like_top__DOT__conv1__DOT__activated = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 10084574018367065753ull);
    vlSelf->d3_vgg_like_top__DOT__conv1__DOT__out_row = VL_SCOPED_RAND_RESET_I(16, __VscopeHash, 18198323435681514534ull);
    vlSelf->d3_vgg_like_top__DOT__conv1__DOT__out_col = VL_SCOPED_RAND_RESET_I(16, __VscopeHash, 17494823848481041614ull);
    vlSelf->d3_vgg_like_top__DOT__conv1__DOT__out_channel = VL_SCOPED_RAND_RESET_I(16, __VscopeHash, 12764408836092046749ull);
    vlSelf->d3_vgg_like_top__DOT__conv1__DOT__controller_out_addr = VL_SCOPED_RAND_RESET_I(17, __VscopeHash, 13008841342266395512ull);
    vlSelf->d3_vgg_like_top__DOT__conv1__DOT__postprocess_pending = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 16883848108237577901ull);
    vlSelf->d3_vgg_like_top__DOT__conv1__DOT__product_valid = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 14288281783895024682ull);
    vlSelf->d3_vgg_like_top__DOT__conv1__DOT__postprocess_valid = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 5798776107137957519ull);
    vlSelf->d3_vgg_like_top__DOT__conv1__DOT__product_addr_reg = VL_SCOPED_RAND_RESET_I(17, __VscopeHash, 3775049428359889340ull);
    vlSelf->d3_vgg_like_top__DOT__conv1__DOT__out_data_reg = VL_SCOPED_RAND_RESET_I(8, __VscopeHash, 5490188574125063744ull);
    vlSelf->d3_vgg_like_top__DOT__conv1__DOT__out_addr_reg = VL_SCOPED_RAND_RESET_I(17, __VscopeHash, 11739165258255780740ull);
    vlSelf->d3_vgg_like_top__DOT__conv1__DOT__controller_done_d1 = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 5743947768326197771ull);
    vlSelf->d3_vgg_like_top__DOT__conv1__DOT__controller_done_d2 = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 4687280542552600225ull);
    vlSelf->d3_vgg_like_top__DOT__conv1__DOT__controller__DOT__state = VL_SCOPED_RAND_RESET_I(2, __VscopeHash, 14383149163542826229ull);
    vlSelf->d3_vgg_like_top__DOT__conv1__DOT__engine__DOT__state = VL_SCOPED_RAND_RESET_I(2, __VscopeHash, 9979778796694882262ull);
    vlSelf->d3_vgg_like_top__DOT__conv1__DOT__engine__DOT__ic = VL_SCOPED_RAND_RESET_I(16, __VscopeHash, 8927661119413928058ull);
    vlSelf->d3_vgg_like_top__DOT__conv1__DOT__engine__DOT__kr = VL_SCOPED_RAND_RESET_I(2, __VscopeHash, 616001489380499512ull);
    vlSelf->d3_vgg_like_top__DOT__conv1__DOT__engine__DOT__kc = VL_SCOPED_RAND_RESET_I(2, __VscopeHash, 15351578058216066265ull);
    vlSelf->d3_vgg_like_top__DOT__conv1__DOT__engine__DOT__valid_pixel = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 12156481796924102122ull);
    vlSelf->d3_vgg_like_top__DOT__conv1__DOT__engine__DOT__product_reg = VL_SCOPED_RAND_RESET_I(16, __VscopeHash, 12175648421677542014ull);
    vlSelf->d3_vgg_like_top__DOT__conv1__DOT__engine__DOT__product_valid = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 8786223074272269233ull);
    vlSelf->d3_vgg_like_top__DOT__conv1__DOT__engine__DOT__product_last = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 13996354410175002109ull);
    vlSelf->d3_vgg_like_top__DOT__conv1__DOT__engine__DOT__last_item = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 6867562011204659774ull);
    vlSelf->d3_vgg_like_top__DOT__conv2__DOT__engine_start = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 6816385830439641959ull);
    vlSelf->d3_vgg_like_top__DOT__conv2__DOT__engine_done = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 1764167252943821201ull);
    vlSelf->d3_vgg_like_top__DOT__conv2__DOT__controller_busy = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 17511680758117457025ull);
    vlSelf->d3_vgg_like_top__DOT__conv2__DOT__controller_done = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 10587270428880806620ull);
    vlSelf->d3_vgg_like_top__DOT__conv2__DOT__raw_sum = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 15066811869515950209ull);
    vlSelf->d3_vgg_like_top__DOT__conv2__DOT__biased_sum_reg = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 8313231827983971449ull);
    vlSelf->d3_vgg_like_top__DOT__conv2__DOT__product_reg = VL_SCOPED_RAND_RESET_Q(43, __VscopeHash, 5559179616137764953ull);
    vlSelf->d3_vgg_like_top__DOT__conv2__DOT__activated = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 4633839986446389011ull);
    vlSelf->d3_vgg_like_top__DOT__conv2__DOT__out_row = VL_SCOPED_RAND_RESET_I(16, __VscopeHash, 8749773945207412594ull);
    vlSelf->d3_vgg_like_top__DOT__conv2__DOT__out_col = VL_SCOPED_RAND_RESET_I(16, __VscopeHash, 9160686996196194809ull);
    vlSelf->d3_vgg_like_top__DOT__conv2__DOT__out_channel = VL_SCOPED_RAND_RESET_I(16, __VscopeHash, 3689268639630167546ull);
    vlSelf->d3_vgg_like_top__DOT__conv2__DOT__controller_out_addr = VL_SCOPED_RAND_RESET_I(17, __VscopeHash, 7607366751339158848ull);
    vlSelf->d3_vgg_like_top__DOT__conv2__DOT__postprocess_pending = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 11784473791054638756ull);
    vlSelf->d3_vgg_like_top__DOT__conv2__DOT__product_valid = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 620174967627777862ull);
    vlSelf->d3_vgg_like_top__DOT__conv2__DOT__postprocess_valid = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 10228165828639426367ull);
    vlSelf->d3_vgg_like_top__DOT__conv2__DOT__product_addr_reg = VL_SCOPED_RAND_RESET_I(17, __VscopeHash, 15108794668745157207ull);
    vlSelf->d3_vgg_like_top__DOT__conv2__DOT__out_data_reg = VL_SCOPED_RAND_RESET_I(8, __VscopeHash, 1121355899596203089ull);
    vlSelf->d3_vgg_like_top__DOT__conv2__DOT__out_addr_reg = VL_SCOPED_RAND_RESET_I(17, __VscopeHash, 4204815945607221857ull);
    vlSelf->d3_vgg_like_top__DOT__conv2__DOT__controller_done_d1 = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 9488885777047039318ull);
    vlSelf->d3_vgg_like_top__DOT__conv2__DOT__controller_done_d2 = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 12279372869580794386ull);
    vlSelf->d3_vgg_like_top__DOT__conv2__DOT__controller__DOT__state = VL_SCOPED_RAND_RESET_I(2, __VscopeHash, 9671452004354876422ull);
    vlSelf->d3_vgg_like_top__DOT__conv2__DOT__engine__DOT__state = VL_SCOPED_RAND_RESET_I(2, __VscopeHash, 1197804865302962412ull);
    vlSelf->d3_vgg_like_top__DOT__conv2__DOT__engine__DOT__ic = VL_SCOPED_RAND_RESET_I(16, __VscopeHash, 12836745791747802095ull);
    vlSelf->d3_vgg_like_top__DOT__conv2__DOT__engine__DOT__kr = VL_SCOPED_RAND_RESET_I(2, __VscopeHash, 17149734756059128277ull);
    vlSelf->d3_vgg_like_top__DOT__conv2__DOT__engine__DOT__kc = VL_SCOPED_RAND_RESET_I(2, __VscopeHash, 13277109884780526676ull);
    vlSelf->d3_vgg_like_top__DOT__conv2__DOT__engine__DOT__valid_pixel = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 5219767716057504490ull);
    vlSelf->d3_vgg_like_top__DOT__conv2__DOT__engine__DOT__product_reg = VL_SCOPED_RAND_RESET_I(16, __VscopeHash, 18256043240653473011ull);
    vlSelf->d3_vgg_like_top__DOT__conv2__DOT__engine__DOT__product_valid = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 16281134795517790496ull);
    vlSelf->d3_vgg_like_top__DOT__conv2__DOT__engine__DOT__product_last = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 15513616631813383808ull);
    vlSelf->d3_vgg_like_top__DOT__conv2__DOT__engine__DOT__last_item = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 17042617294582875172ull);
    vlSelf->d3_vgg_like_top__DOT__pool1__DOT__state = VL_SCOPED_RAND_RESET_I(3, __VscopeHash, 11127324990514361295ull);
    vlSelf->d3_vgg_like_top__DOT__pool1__DOT__ch = VL_SCOPED_RAND_RESET_I(16, __VscopeHash, 11477635922743381635ull);
    vlSelf->d3_vgg_like_top__DOT__pool1__DOT__row = VL_SCOPED_RAND_RESET_I(16, __VscopeHash, 16546850683333607446ull);
    vlSelf->d3_vgg_like_top__DOT__pool1__DOT__col = VL_SCOPED_RAND_RESET_I(16, __VscopeHash, 16501285433132426613ull);
    vlSelf->d3_vgg_like_top__DOT__pool1__DOT__tap = VL_SCOPED_RAND_RESET_I(2, __VscopeHash, 902048312431286452ull);
    vlSelf->d3_vgg_like_top__DOT__pool1__DOT__current_max = VL_SCOPED_RAND_RESET_I(8, __VscopeHash, 16931537216041768806ull);
    vlSelf->d3_vgg_like_top__DOT__conv3__DOT__engine_start = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 1623118407546414136ull);
    vlSelf->d3_vgg_like_top__DOT__conv3__DOT__engine_done = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 1260829458996388638ull);
    vlSelf->d3_vgg_like_top__DOT__conv3__DOT__controller_busy = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 4259271473546182234ull);
    vlSelf->d3_vgg_like_top__DOT__conv3__DOT__controller_done = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 17150221567578590283ull);
    vlSelf->d3_vgg_like_top__DOT__conv3__DOT__raw_sum = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 2829684520493082261ull);
    vlSelf->d3_vgg_like_top__DOT__conv3__DOT__biased_sum_reg = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 6944614090181650394ull);
    vlSelf->d3_vgg_like_top__DOT__conv3__DOT__product_reg = VL_SCOPED_RAND_RESET_Q(43, __VscopeHash, 4221841362633087329ull);
    vlSelf->d3_vgg_like_top__DOT__conv3__DOT__activated = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 17657197799028406857ull);
    vlSelf->d3_vgg_like_top__DOT__conv3__DOT__out_row = VL_SCOPED_RAND_RESET_I(16, __VscopeHash, 6410112299749596367ull);
    vlSelf->d3_vgg_like_top__DOT__conv3__DOT__out_col = VL_SCOPED_RAND_RESET_I(16, __VscopeHash, 4853215615885135353ull);
    vlSelf->d3_vgg_like_top__DOT__conv3__DOT__out_channel = VL_SCOPED_RAND_RESET_I(16, __VscopeHash, 15604368169234183761ull);
    vlSelf->d3_vgg_like_top__DOT__conv3__DOT__controller_out_addr = VL_SCOPED_RAND_RESET_I(17, __VscopeHash, 14134829435917388773ull);
    vlSelf->d3_vgg_like_top__DOT__conv3__DOT__postprocess_pending = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 2961856035911111250ull);
    vlSelf->d3_vgg_like_top__DOT__conv3__DOT__product_valid = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 6298269200711394810ull);
    vlSelf->d3_vgg_like_top__DOT__conv3__DOT__postprocess_valid = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 8386971048491304069ull);
    vlSelf->d3_vgg_like_top__DOT__conv3__DOT__product_addr_reg = VL_SCOPED_RAND_RESET_I(17, __VscopeHash, 11193077902916905361ull);
    vlSelf->d3_vgg_like_top__DOT__conv3__DOT__out_data_reg = VL_SCOPED_RAND_RESET_I(8, __VscopeHash, 989522266853633323ull);
    vlSelf->d3_vgg_like_top__DOT__conv3__DOT__out_addr_reg = VL_SCOPED_RAND_RESET_I(17, __VscopeHash, 9195976390726853089ull);
    vlSelf->d3_vgg_like_top__DOT__conv3__DOT__controller_done_d1 = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 10679217984532000293ull);
    vlSelf->d3_vgg_like_top__DOT__conv3__DOT__controller_done_d2 = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 6941421682369346044ull);
    vlSelf->d3_vgg_like_top__DOT__conv3__DOT__controller__DOT__state = VL_SCOPED_RAND_RESET_I(2, __VscopeHash, 8142669675156913490ull);
    vlSelf->d3_vgg_like_top__DOT__conv3__DOT__engine__DOT__state = VL_SCOPED_RAND_RESET_I(2, __VscopeHash, 12034921090480943610ull);
    vlSelf->d3_vgg_like_top__DOT__conv3__DOT__engine__DOT__ic = VL_SCOPED_RAND_RESET_I(16, __VscopeHash, 10852167931943103076ull);
    vlSelf->d3_vgg_like_top__DOT__conv3__DOT__engine__DOT__kr = VL_SCOPED_RAND_RESET_I(2, __VscopeHash, 17082795127915515745ull);
    vlSelf->d3_vgg_like_top__DOT__conv3__DOT__engine__DOT__kc = VL_SCOPED_RAND_RESET_I(2, __VscopeHash, 14039827821354012509ull);
    vlSelf->d3_vgg_like_top__DOT__conv3__DOT__engine__DOT__valid_pixel = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 266986535410050785ull);
    vlSelf->d3_vgg_like_top__DOT__conv3__DOT__engine__DOT__product_reg = VL_SCOPED_RAND_RESET_I(16, __VscopeHash, 18148763412965507134ull);
    vlSelf->d3_vgg_like_top__DOT__conv3__DOT__engine__DOT__product_valid = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 3723223065939370293ull);
    vlSelf->d3_vgg_like_top__DOT__conv3__DOT__engine__DOT__product_last = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 853132190110647723ull);
    vlSelf->d3_vgg_like_top__DOT__conv3__DOT__engine__DOT__last_item = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 13304384063840040738ull);
    vlSelf->d3_vgg_like_top__DOT__conv4__DOT__engine_start = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 609334783864828160ull);
    vlSelf->d3_vgg_like_top__DOT__conv4__DOT__engine_done = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 8723034199813334268ull);
    vlSelf->d3_vgg_like_top__DOT__conv4__DOT__controller_busy = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 3081586884767180123ull);
    vlSelf->d3_vgg_like_top__DOT__conv4__DOT__controller_done = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 9724355685795588619ull);
    vlSelf->d3_vgg_like_top__DOT__conv4__DOT__raw_sum = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 5845225254212703315ull);
    vlSelf->d3_vgg_like_top__DOT__conv4__DOT__biased_sum_reg = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 6696193130907921422ull);
    vlSelf->d3_vgg_like_top__DOT__conv4__DOT__product_reg = VL_SCOPED_RAND_RESET_Q(43, __VscopeHash, 16903015335746482211ull);
    vlSelf->d3_vgg_like_top__DOT__conv4__DOT__activated = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 16233296679283260749ull);
    vlSelf->d3_vgg_like_top__DOT__conv4__DOT__out_row = VL_SCOPED_RAND_RESET_I(16, __VscopeHash, 215426464076289459ull);
    vlSelf->d3_vgg_like_top__DOT__conv4__DOT__out_col = VL_SCOPED_RAND_RESET_I(16, __VscopeHash, 17158316741772136501ull);
    vlSelf->d3_vgg_like_top__DOT__conv4__DOT__out_channel = VL_SCOPED_RAND_RESET_I(16, __VscopeHash, 13105098659558764154ull);
    vlSelf->d3_vgg_like_top__DOT__conv4__DOT__controller_out_addr = VL_SCOPED_RAND_RESET_I(17, __VscopeHash, 1833422193494803088ull);
    vlSelf->d3_vgg_like_top__DOT__conv4__DOT__postprocess_pending = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 14834110637324729165ull);
    vlSelf->d3_vgg_like_top__DOT__conv4__DOT__product_valid = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 17252224221307364995ull);
    vlSelf->d3_vgg_like_top__DOT__conv4__DOT__postprocess_valid = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 16367054456241635656ull);
    vlSelf->d3_vgg_like_top__DOT__conv4__DOT__product_addr_reg = VL_SCOPED_RAND_RESET_I(17, __VscopeHash, 10515451209370897261ull);
    vlSelf->d3_vgg_like_top__DOT__conv4__DOT__out_data_reg = VL_SCOPED_RAND_RESET_I(8, __VscopeHash, 5030585741183868593ull);
    vlSelf->d3_vgg_like_top__DOT__conv4__DOT__out_addr_reg = VL_SCOPED_RAND_RESET_I(17, __VscopeHash, 15737763686563204997ull);
    vlSelf->d3_vgg_like_top__DOT__conv4__DOT__controller_done_d1 = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 12482857456915141441ull);
    vlSelf->d3_vgg_like_top__DOT__conv4__DOT__controller_done_d2 = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 14972060968058578636ull);
    vlSelf->d3_vgg_like_top__DOT__conv4__DOT__controller__DOT__state = VL_SCOPED_RAND_RESET_I(2, __VscopeHash, 3636640757385981996ull);
    vlSelf->d3_vgg_like_top__DOT__conv4__DOT__engine__DOT__state = VL_SCOPED_RAND_RESET_I(2, __VscopeHash, 17823952655915135045ull);
    vlSelf->d3_vgg_like_top__DOT__conv4__DOT__engine__DOT__ic = VL_SCOPED_RAND_RESET_I(16, __VscopeHash, 1202489502227831751ull);
    vlSelf->d3_vgg_like_top__DOT__conv4__DOT__engine__DOT__kr = VL_SCOPED_RAND_RESET_I(2, __VscopeHash, 747194259093276083ull);
    vlSelf->d3_vgg_like_top__DOT__conv4__DOT__engine__DOT__kc = VL_SCOPED_RAND_RESET_I(2, __VscopeHash, 3087357325831183340ull);
    vlSelf->d3_vgg_like_top__DOT__conv4__DOT__engine__DOT__valid_pixel = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 3400527662097963526ull);
    vlSelf->d3_vgg_like_top__DOT__conv4__DOT__engine__DOT__product_reg = VL_SCOPED_RAND_RESET_I(16, __VscopeHash, 10370639329676438424ull);
    vlSelf->d3_vgg_like_top__DOT__conv4__DOT__engine__DOT__product_valid = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 4627952036408083889ull);
    vlSelf->d3_vgg_like_top__DOT__conv4__DOT__engine__DOT__product_last = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 2637566006173836245ull);
    vlSelf->d3_vgg_like_top__DOT__conv4__DOT__engine__DOT__last_item = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 18418931351233103612ull);
    vlSelf->d3_vgg_like_top__DOT__pool2__DOT__state = VL_SCOPED_RAND_RESET_I(3, __VscopeHash, 18235620333185461064ull);
    vlSelf->d3_vgg_like_top__DOT__pool2__DOT__ch = VL_SCOPED_RAND_RESET_I(16, __VscopeHash, 3667912592836408018ull);
    vlSelf->d3_vgg_like_top__DOT__pool2__DOT__row = VL_SCOPED_RAND_RESET_I(16, __VscopeHash, 17912605355262358686ull);
    vlSelf->d3_vgg_like_top__DOT__pool2__DOT__col = VL_SCOPED_RAND_RESET_I(16, __VscopeHash, 8484806689433644525ull);
    vlSelf->d3_vgg_like_top__DOT__pool2__DOT__tap = VL_SCOPED_RAND_RESET_I(2, __VscopeHash, 18133117689867484715ull);
    vlSelf->d3_vgg_like_top__DOT__pool2__DOT__current_max = VL_SCOPED_RAND_RESET_I(8, __VscopeHash, 17535523474949280571ull);
    vlSelf->d3_vgg_like_top__DOT__gap__DOT__state = VL_SCOPED_RAND_RESET_I(3, __VscopeHash, 15750596226106090027ull);
    vlSelf->d3_vgg_like_top__DOT__gap__DOT__ch = VL_SCOPED_RAND_RESET_I(16, __VscopeHash, 9885474010675783142ull);
    vlSelf->d3_vgg_like_top__DOT__gap__DOT__idx = VL_SCOPED_RAND_RESET_I(16, __VscopeHash, 8961994894590687462ull);
    vlSelf->d3_vgg_like_top__DOT__gap__DOT__sum = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 7713719715904631527ull);
    vlSelf->d3_vgg_like_top__DOT__linear__DOT__state = VL_SCOPED_RAND_RESET_I(3, __VscopeHash, 14365402974507041433ull);
    vlSelf->d3_vgg_like_top__DOT__linear__DOT__class_idx = VL_SCOPED_RAND_RESET_I(4, __VscopeHash, 7317228221296093262ull);
    vlSelf->d3_vgg_like_top__DOT__linear__DOT__feature_idx = VL_SCOPED_RAND_RESET_I(7, __VscopeHash, 8343239336700198931ull);
    vlSelf->d3_vgg_like_top__DOT__linear__DOT__acc = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 10159850214984601865ull);
    vlSelf->__VdfgRegularize_h6e95ff9d_0_0 = 0;
    vlSelf->__VdfgRegularize_h6e95ff9d_0_1 = 0;
    vlSelf->__VdfgRegularize_h6e95ff9d_0_2 = 0;
    vlSelf->__VdfgRegularize_h6e95ff9d_0_3 = 0;
    vlSelf->__VdfgRegularize_h6e95ff9d_0_4 = 0;
    vlSelf->__VdfgRegularize_h6e95ff9d_0_5 = 0;
    vlSelf->__VdfgRegularize_h6e95ff9d_0_6 = 0;
    vlSelf->__VdfgRegularize_h6e95ff9d_0_7 = 0;
    vlSelf->__VdfgRegularize_h6e95ff9d_0_8 = 0;
    vlSelf->__Vdly__d3_vgg_like_top__DOT__c1_start = 0;
    vlSelf->__Vdly__d3_vgg_like_top__DOT__c2_start = 0;
    vlSelf->__Vdly__d3_vgg_like_top__DOT__p1_start = 0;
    vlSelf->__Vdly__d3_vgg_like_top__DOT__c3_start = 0;
    vlSelf->__Vdly__d3_vgg_like_top__DOT__c4_start = 0;
    vlSelf->__Vdly__d3_vgg_like_top__DOT__p2_start = 0;
    vlSelf->__Vdly__d3_vgg_like_top__DOT__gap_start = 0;
    vlSelf->__Vdly__state_dbg = 0;
    vlSelf->__Vdly__d3_vgg_like_top__DOT__conv1__DOT__engine_start = 0;
    vlSelf->__Vdly__d3_vgg_like_top__DOT__conv1__DOT__controller__DOT__state = 0;
    vlSelf->__Vdly__d3_vgg_like_top__DOT__conv1__DOT__engine__DOT__state = 0;
    vlSelf->__Vdly__d3_vgg_like_top__DOT__conv1__DOT__engine__DOT__ic = 0;
    vlSelf->__Vdly__d3_vgg_like_top__DOT__conv1__DOT__engine__DOT__product_reg = 0;
    vlSelf->__Vdly__d3_vgg_like_top__DOT__conv1__DOT__engine__DOT__product_valid = 0;
    vlSelf->__Vdly__d3_vgg_like_top__DOT__conv1__DOT__engine__DOT__product_last = 0;
    vlSelf->__Vdly__d3_vgg_like_top__DOT__conv2__DOT__engine_start = 0;
    vlSelf->__Vdly__d3_vgg_like_top__DOT__conv2__DOT__controller__DOT__state = 0;
    vlSelf->__Vdly__d3_vgg_like_top__DOT__conv2__DOT__engine__DOT__state = 0;
    vlSelf->__Vdly__d3_vgg_like_top__DOT__conv2__DOT__engine__DOT__ic = 0;
    vlSelf->__Vdly__d3_vgg_like_top__DOT__conv2__DOT__engine__DOT__product_reg = 0;
    vlSelf->__Vdly__d3_vgg_like_top__DOT__conv2__DOT__engine__DOT__product_valid = 0;
    vlSelf->__Vdly__d3_vgg_like_top__DOT__conv2__DOT__engine__DOT__product_last = 0;
    vlSelf->__Vdly__d3_vgg_like_top__DOT__pool1__DOT__state = 0;
    vlSelf->__Vdly__d3_vgg_like_top__DOT__pool1__DOT__tap = 0;
    vlSelf->__Vdly__d3_vgg_like_top__DOT__pool1__DOT__col = 0;
    vlSelf->__Vdly__d3_vgg_like_top__DOT__pool1__DOT__row = 0;
    vlSelf->__Vdly__d3_vgg_like_top__DOT__pool1__DOT__ch = 0;
    vlSelf->__Vdly__d3_vgg_like_top__DOT__pool1__DOT__current_max = 0;
    vlSelf->__Vdly__d3_vgg_like_top__DOT__conv3__DOT__engine_start = 0;
    vlSelf->__Vdly__d3_vgg_like_top__DOT__conv3__DOT__controller__DOT__state = 0;
    vlSelf->__Vdly__d3_vgg_like_top__DOT__conv3__DOT__engine__DOT__state = 0;
    vlSelf->__Vdly__d3_vgg_like_top__DOT__conv3__DOT__engine__DOT__ic = 0;
    vlSelf->__Vdly__d3_vgg_like_top__DOT__conv3__DOT__engine__DOT__product_reg = 0;
    vlSelf->__Vdly__d3_vgg_like_top__DOT__conv3__DOT__engine__DOT__product_valid = 0;
    vlSelf->__Vdly__d3_vgg_like_top__DOT__conv3__DOT__engine__DOT__product_last = 0;
    vlSelf->__Vdly__d3_vgg_like_top__DOT__conv4__DOT__engine_start = 0;
    vlSelf->__Vdly__d3_vgg_like_top__DOT__conv4__DOT__controller__DOT__state = 0;
    vlSelf->__Vdly__d3_vgg_like_top__DOT__conv4__DOT__engine__DOT__state = 0;
    vlSelf->__Vdly__d3_vgg_like_top__DOT__conv4__DOT__engine__DOT__ic = 0;
    vlSelf->__Vdly__d3_vgg_like_top__DOT__conv4__DOT__engine__DOT__product_reg = 0;
    vlSelf->__Vdly__d3_vgg_like_top__DOT__conv4__DOT__engine__DOT__product_valid = 0;
    vlSelf->__Vdly__d3_vgg_like_top__DOT__conv4__DOT__engine__DOT__product_last = 0;
    vlSelf->__Vdly__d3_vgg_like_top__DOT__pool2__DOT__state = 0;
    vlSelf->__Vdly__d3_vgg_like_top__DOT__pool2__DOT__tap = 0;
    vlSelf->__Vdly__d3_vgg_like_top__DOT__pool2__DOT__col = 0;
    vlSelf->__Vdly__d3_vgg_like_top__DOT__pool2__DOT__row = 0;
    vlSelf->__Vdly__d3_vgg_like_top__DOT__pool2__DOT__ch = 0;
    vlSelf->__Vdly__d3_vgg_like_top__DOT__pool2__DOT__current_max = 0;
    vlSelf->__Vdly__d3_vgg_like_top__DOT__gap__DOT__state = 0;
    vlSelf->__Vdly__d3_vgg_like_top__DOT__gap__DOT__idx = 0;
    vlSelf->__Vdly__d3_vgg_like_top__DOT__gap__DOT__ch = 0;
    vlSelf->__Vdly__d3_vgg_like_top__DOT__gap__DOT__sum = 0;
    vlSelf->__Vdly__d3_vgg_like_top__DOT__linear__DOT__feature_idx = 0;
    vlSelf->__VdlyVal__d3_vgg_like_top__DOT__buffer_a__DOT__mem__v0 = 0;
    vlSelf->__VdlyDim0__d3_vgg_like_top__DOT__buffer_a__DOT__mem__v0 = 0;
    vlSelf->__VdlySet__d3_vgg_like_top__DOT__buffer_a__DOT__mem__v0 = 0;
    vlSelf->__VdlyVal__d3_vgg_like_top__DOT__buffer_b__DOT__mem__v0 = 0;
    vlSelf->__VdlyDim0__d3_vgg_like_top__DOT__buffer_b__DOT__mem__v0 = 0;
    vlSelf->__VdlySet__d3_vgg_like_top__DOT__buffer_b__DOT__mem__v0 = 0;
    for (int __Vi0 = 0; __Vi0 < 1; ++__Vi0) {
        vlSelf->__VstlTriggered[__Vi0] = 0;
    }
    for (int __Vi0 = 0; __Vi0 < 1; ++__Vi0) {
        vlSelf->__VicoTriggered[__Vi0] = 0;
    }
    for (int __Vi0 = 0; __Vi0 < 1; ++__Vi0) {
        vlSelf->__VactTriggered[__Vi0] = 0;
    }
    vlSelf->__Vtrigprevexpr___TOP__clk__0 = 0;
    vlSelf->__Vtrigprevexpr___TOP__rst_n__0 = 0;
    for (int __Vi0 = 0; __Vi0 < 1; ++__Vi0) {
        vlSelf->__VnbaTriggered[__Vi0] = 0;
    }
}
