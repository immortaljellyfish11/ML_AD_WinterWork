// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design implementation internals
// See Vedge_cnn_top.h for the primary calling header

#include "Vedge_cnn_top__pch.h"

VL_ATTR_COLD void Vedge_cnn_top___024root___eval_static(Vedge_cnn_top___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vedge_cnn_top___024root___eval_static\n"); );
    Vedge_cnn_top__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    vlSelfRef.__Vtrigprevexpr___TOP__clk__0 = vlSelfRef.clk;
    vlSelfRef.__Vtrigprevexpr___TOP__rst_n__0 = vlSelfRef.rst_n;
}

VL_ATTR_COLD void Vedge_cnn_top___024root___eval_initial(Vedge_cnn_top___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vedge_cnn_top___024root___eval_initial\n"); );
    Vedge_cnn_top__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    VL_READMEM_N(true, 8, 864, 0, "../architectures/d1_baseline/artifacts/int8_params/conv1_weight.mem"s
                 ,  &(vlSelfRef.edge_cnn_top__DOT__conv1_weight_rom__DOT__rom)
                 , 0, ~0ULL);
    VL_READMEM_N(true, 8, 18432, 0, "../architectures/d1_baseline/artifacts/int8_params/conv2_weight.mem"s
                 ,  &(vlSelfRef.edge_cnn_top__DOT__conv2_weight_rom__DOT__rom)
                 , 0, ~0ULL);
    VL_READMEM_N(true, 8, 73728, 0, "../architectures/d1_baseline/artifacts/int8_params/conv3_weight.mem"s
                 ,  &(vlSelfRef.edge_cnn_top__DOT__conv3_weight_rom__DOT__rom)
                 , 0, ~0ULL);
    VL_READMEM_N(true, 8, 294912, 0, "../architectures/d1_baseline/artifacts/int8_params/conv4_weight.mem"s
                 ,  &(vlSelfRef.edge_cnn_top__DOT__conv4_weight_rom__DOT__rom)
                 , 0, ~0ULL);
    VL_READMEM_N(true, 8, 2560, 0, "../architectures/d1_baseline/artifacts/int8_params/linear_weight.mem"s
                 ,  &(vlSelfRef.edge_cnn_top__DOT__linear_weight_rom__DOT__rom)
                 , 0, ~0ULL);
    VL_READMEM_N(true, 32, 32, 0, "../architectures/d1_baseline/artifacts/int8_params/conv1_bias.mem"s
                 ,  &(vlSelfRef.edge_cnn_top__DOT__conv1_bias_rom__DOT__rom)
                 , 0, ~0ULL);
    VL_READMEM_N(true, 32, 64, 0, "../architectures/d1_baseline/artifacts/int8_params/conv2_bias.mem"s
                 ,  &(vlSelfRef.edge_cnn_top__DOT__conv2_bias_rom__DOT__rom)
                 , 0, ~0ULL);
    VL_READMEM_N(true, 32, 128, 0, "../architectures/d1_baseline/artifacts/int8_params/conv3_bias.mem"s
                 ,  &(vlSelfRef.edge_cnn_top__DOT__conv3_bias_rom__DOT__rom)
                 , 0, ~0ULL);
    VL_READMEM_N(true, 32, 256, 0, "../architectures/d1_baseline/artifacts/int8_params/conv4_bias.mem"s
                 ,  &(vlSelfRef.edge_cnn_top__DOT__conv4_bias_rom__DOT__rom)
                 , 0, ~0ULL);
    VL_READMEM_N(true, 32, 10, 0, "../architectures/d1_baseline/artifacts/int8_params/linear_bias.mem"s
                 ,  &(vlSelfRef.edge_cnn_top__DOT__linear_bias_rom__DOT__rom)
                 , 0, ~0ULL);
}

VL_ATTR_COLD void Vedge_cnn_top___024root___eval_initial__TOP(Vedge_cnn_top___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vedge_cnn_top___024root___eval_initial__TOP\n"); );
    Vedge_cnn_top__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    VL_READMEM_N(true, 8, 864, 0, "../architectures/d1_baseline/artifacts/int8_params/conv1_weight.mem"s
                 ,  &(vlSelfRef.edge_cnn_top__DOT__conv1_weight_rom__DOT__rom)
                 , 0, ~0ULL);
    VL_READMEM_N(true, 8, 18432, 0, "../architectures/d1_baseline/artifacts/int8_params/conv2_weight.mem"s
                 ,  &(vlSelfRef.edge_cnn_top__DOT__conv2_weight_rom__DOT__rom)
                 , 0, ~0ULL);
    VL_READMEM_N(true, 8, 73728, 0, "../architectures/d1_baseline/artifacts/int8_params/conv3_weight.mem"s
                 ,  &(vlSelfRef.edge_cnn_top__DOT__conv3_weight_rom__DOT__rom)
                 , 0, ~0ULL);
    VL_READMEM_N(true, 8, 294912, 0, "../architectures/d1_baseline/artifacts/int8_params/conv4_weight.mem"s
                 ,  &(vlSelfRef.edge_cnn_top__DOT__conv4_weight_rom__DOT__rom)
                 , 0, ~0ULL);
    VL_READMEM_N(true, 8, 2560, 0, "../architectures/d1_baseline/artifacts/int8_params/linear_weight.mem"s
                 ,  &(vlSelfRef.edge_cnn_top__DOT__linear_weight_rom__DOT__rom)
                 , 0, ~0ULL);
    VL_READMEM_N(true, 32, 32, 0, "../architectures/d1_baseline/artifacts/int8_params/conv1_bias.mem"s
                 ,  &(vlSelfRef.edge_cnn_top__DOT__conv1_bias_rom__DOT__rom)
                 , 0, ~0ULL);
    VL_READMEM_N(true, 32, 64, 0, "../architectures/d1_baseline/artifacts/int8_params/conv2_bias.mem"s
                 ,  &(vlSelfRef.edge_cnn_top__DOT__conv2_bias_rom__DOT__rom)
                 , 0, ~0ULL);
    VL_READMEM_N(true, 32, 128, 0, "../architectures/d1_baseline/artifacts/int8_params/conv3_bias.mem"s
                 ,  &(vlSelfRef.edge_cnn_top__DOT__conv3_bias_rom__DOT__rom)
                 , 0, ~0ULL);
    VL_READMEM_N(true, 32, 256, 0, "../architectures/d1_baseline/artifacts/int8_params/conv4_bias.mem"s
                 ,  &(vlSelfRef.edge_cnn_top__DOT__conv4_bias_rom__DOT__rom)
                 , 0, ~0ULL);
    VL_READMEM_N(true, 32, 10, 0, "../architectures/d1_baseline/artifacts/int8_params/linear_bias.mem"s
                 ,  &(vlSelfRef.edge_cnn_top__DOT__linear_bias_rom__DOT__rom)
                 , 0, ~0ULL);
}

VL_ATTR_COLD void Vedge_cnn_top___024root___eval_final(Vedge_cnn_top___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vedge_cnn_top___024root___eval_final\n"); );
    Vedge_cnn_top__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
}

#ifdef VL_DEBUG
VL_ATTR_COLD void Vedge_cnn_top___024root___dump_triggers__stl(const VlUnpacked<QData/*63:0*/, 1> &triggers, const std::string &tag);
#endif  // VL_DEBUG
VL_ATTR_COLD bool Vedge_cnn_top___024root___eval_phase__stl(Vedge_cnn_top___024root* vlSelf);

VL_ATTR_COLD void Vedge_cnn_top___024root___eval_settle(Vedge_cnn_top___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vedge_cnn_top___024root___eval_settle\n"); );
    Vedge_cnn_top__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Locals
    IData/*31:0*/ __VstlIterCount;
    // Body
    __VstlIterCount = 0U;
    vlSelfRef.__VstlFirstIteration = 1U;
    do {
        if (VL_UNLIKELY(((0x00002710U < __VstlIterCount)))) {
#ifdef VL_DEBUG
            Vedge_cnn_top___024root___dump_triggers__stl(vlSelfRef.__VstlTriggered, "stl"s);
#endif
            VL_FATAL_MT("src/edge_cnn_top.v", 20, "", "DIDNOTCONVERGE: Settle region did not converge after '--converge-limit' of 10000 tries");
        }
        __VstlIterCount = ((IData)(1U) + __VstlIterCount);
        vlSelfRef.__VstlPhaseResult = Vedge_cnn_top___024root___eval_phase__stl(vlSelf);
        vlSelfRef.__VstlFirstIteration = 0U;
    } while (vlSelfRef.__VstlPhaseResult);
}

VL_ATTR_COLD void Vedge_cnn_top___024root___eval_triggers_vec__stl(Vedge_cnn_top___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vedge_cnn_top___024root___eval_triggers_vec__stl\n"); );
    Vedge_cnn_top__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    vlSelfRef.__VstlTriggered[0U] = ((0xfffffffffffffffeULL 
                                      & vlSelfRef.__VstlTriggered[0U]) 
                                     | (IData)((IData)(vlSelfRef.__VstlFirstIteration)));
}

VL_ATTR_COLD bool Vedge_cnn_top___024root___trigger_anySet__stl(const VlUnpacked<QData/*63:0*/, 1> &in);

#ifdef VL_DEBUG
VL_ATTR_COLD void Vedge_cnn_top___024root___dump_triggers__stl(const VlUnpacked<QData/*63:0*/, 1> &triggers, const std::string &tag) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vedge_cnn_top___024root___dump_triggers__stl\n"); );
    // Body
    if ((1U & (~ (IData)(Vedge_cnn_top___024root___trigger_anySet__stl(triggers))))) {
        VL_DBG_MSGS("         No '" + tag + "' region triggers active\n");
    }
    if ((1U & (IData)(triggers[0U]))) {
        VL_DBG_MSGS("         '" + tag + "' region trigger index 0 is active: Internal 'stl' trigger - first iteration\n");
    }
}
#endif  // VL_DEBUG

VL_ATTR_COLD bool Vedge_cnn_top___024root___trigger_anySet__stl(const VlUnpacked<QData/*63:0*/, 1> &in) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vedge_cnn_top___024root___trigger_anySet__stl\n"); );
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

VL_ATTR_COLD void Vedge_cnn_top___024root___stl_sequent__TOP__0(Vedge_cnn_top___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vedge_cnn_top___024root___stl_sequent__TOP__0\n"); );
    Vedge_cnn_top__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Locals
    QData/*42:0*/ edge_cnn_top__DOT__conv1__DOT__quant__DOT__result43;
    edge_cnn_top__DOT__conv1__DOT__quant__DOT__result43 = 0;
    QData/*42:0*/ edge_cnn_top__DOT__conv1__DOT__quant__DOT__gen_round_shift__DOT__shifted_magnitude;
    edge_cnn_top__DOT__conv1__DOT__quant__DOT__gen_round_shift__DOT__shifted_magnitude = 0;
    QData/*42:0*/ edge_cnn_top__DOT__conv2__DOT__quant__DOT__result43;
    edge_cnn_top__DOT__conv2__DOT__quant__DOT__result43 = 0;
    QData/*42:0*/ edge_cnn_top__DOT__conv2__DOT__quant__DOT__gen_round_shift__DOT__shifted_magnitude;
    edge_cnn_top__DOT__conv2__DOT__quant__DOT__gen_round_shift__DOT__shifted_magnitude = 0;
    QData/*42:0*/ edge_cnn_top__DOT__conv3__DOT__quant__DOT__result43;
    edge_cnn_top__DOT__conv3__DOT__quant__DOT__result43 = 0;
    QData/*42:0*/ edge_cnn_top__DOT__conv3__DOT__quant__DOT__gen_round_shift__DOT__shifted_magnitude;
    edge_cnn_top__DOT__conv3__DOT__quant__DOT__gen_round_shift__DOT__shifted_magnitude = 0;
    QData/*42:0*/ edge_cnn_top__DOT__conv4__DOT__quant__DOT__result43;
    edge_cnn_top__DOT__conv4__DOT__quant__DOT__result43 = 0;
    QData/*42:0*/ edge_cnn_top__DOT__conv4__DOT__quant__DOT__gen_round_shift__DOT__shifted_magnitude;
    edge_cnn_top__DOT__conv4__DOT__quant__DOT__gen_round_shift__DOT__shifted_magnitude = 0;
    // Body
    vlSelfRef.__VdfgRegularize_h6e95ff9d_0_10 = ((IData)(vlSelfRef.edge_cnn_top__DOT__conv3__DOT__out_channel) 
                                                 << 6U);
    vlSelfRef.edge_cnn_top__DOT__linear_weight_rom__DOT__rom_addr 
        = (0x00000fffU & ((IData)(vlSelfRef.edge_cnn_top__DOT__linear__DOT__feature_idx) 
                          + ((IData)(vlSelfRef.edge_cnn_top__DOT__linear__DOT__class_idx) 
                             << 8U)));
    vlSelfRef.edge_cnn_top__DOT__conv1__DOT__engine__DOT__last_item 
        = ((2U == (IData)(vlSelfRef.edge_cnn_top__DOT__conv1__DOT__engine__DOT__ic)) 
           & ((2U == (IData)(vlSelfRef.edge_cnn_top__DOT__conv1__DOT__engine__DOT__kr)) 
              & (2U == (IData)(vlSelfRef.edge_cnn_top__DOT__conv1__DOT__engine__DOT__kc))));
    vlSelfRef.edge_cnn_top__DOT__conv2__DOT__engine__DOT__last_item 
        = ((0x001fU == (IData)(vlSelfRef.edge_cnn_top__DOT__conv2__DOT__engine__DOT__ic)) 
           & ((2U == (IData)(vlSelfRef.edge_cnn_top__DOT__conv2__DOT__engine__DOT__kr)) 
              & (2U == (IData)(vlSelfRef.edge_cnn_top__DOT__conv2__DOT__engine__DOT__kc))));
    vlSelfRef.edge_cnn_top__DOT__conv3__DOT__engine__DOT__last_item 
        = ((0x003fU == (IData)(vlSelfRef.edge_cnn_top__DOT__conv3__DOT__engine__DOT__ic)) 
           & ((2U == (IData)(vlSelfRef.edge_cnn_top__DOT__conv3__DOT__engine__DOT__kr)) 
              & (2U == (IData)(vlSelfRef.edge_cnn_top__DOT__conv3__DOT__engine__DOT__kc))));
    vlSelfRef.edge_cnn_top__DOT__conv4__DOT__engine__DOT__last_item 
        = ((0x007fU == (IData)(vlSelfRef.edge_cnn_top__DOT__conv4__DOT__engine__DOT__ic)) 
           & ((2U == (IData)(vlSelfRef.edge_cnn_top__DOT__conv4__DOT__engine__DOT__kr)) 
              & (2U == (IData)(vlSelfRef.edge_cnn_top__DOT__conv4__DOT__engine__DOT__kc))));
    vlSelfRef.edge_cnn_top__DOT__conv4_weight_addr 
        = (0x0007ffffU & ((IData)(vlSelfRef.edge_cnn_top__DOT__conv4__DOT__engine__DOT__kc) 
                          + (((IData)(9U) * VL_SHIFTL_III(32,32,32, (IData)(vlSelfRef.edge_cnn_top__DOT__conv4__DOT__out_channel), 7U)) 
                             + (((IData)(9U) * (IData)(vlSelfRef.edge_cnn_top__DOT__conv4__DOT__engine__DOT__ic)) 
                                + ((IData)(3U) * (IData)(vlSelfRef.edge_cnn_top__DOT__conv4__DOT__engine__DOT__kr))))));
    vlSelfRef.edge_cnn_top__DOT__conv1_weight_rom__DOT__rom_addr 
        = (0x000003ffU & ((IData)(vlSelfRef.edge_cnn_top__DOT__conv1__DOT__engine__DOT__kc) 
                          + (((IData)(0x0000001bU) 
                              * (IData)(vlSelfRef.edge_cnn_top__DOT__conv1__DOT__out_channel)) 
                             + (((IData)(9U) * (IData)(vlSelfRef.edge_cnn_top__DOT__conv1__DOT__engine__DOT__ic)) 
                                + ((IData)(3U) * (IData)(vlSelfRef.edge_cnn_top__DOT__conv1__DOT__engine__DOT__kr))))));
    vlSelfRef.edge_cnn_top__DOT__conv2_weight_rom__DOT__rom_addr 
        = (0x00007fffU & ((IData)(vlSelfRef.edge_cnn_top__DOT__conv2__DOT__engine__DOT__kc) 
                          + (((IData)(9U) * VL_SHIFTL_III(32,32,32, (IData)(vlSelfRef.edge_cnn_top__DOT__conv2__DOT__out_channel), 5U)) 
                             + (((IData)(9U) * (IData)(vlSelfRef.edge_cnn_top__DOT__conv2__DOT__engine__DOT__ic)) 
                                + ((IData)(3U) * (IData)(vlSelfRef.edge_cnn_top__DOT__conv2__DOT__engine__DOT__kr))))));
    vlSelfRef.edge_cnn_top__DOT__conv3_weight_rom__DOT__rom_addr 
        = (0x0001ffffU & ((IData)(vlSelfRef.edge_cnn_top__DOT__conv3__DOT__engine__DOT__kc) 
                          + (((IData)(9U) * ((IData)(vlSelfRef.edge_cnn_top__DOT__conv3__DOT__out_channel) 
                                             << 6U)) 
                             + (((IData)(9U) * (IData)(vlSelfRef.edge_cnn_top__DOT__conv3__DOT__engine__DOT__ic)) 
                                + ((IData)(3U) * (IData)(vlSelfRef.edge_cnn_top__DOT__conv3__DOT__engine__DOT__kr))))));
    vlSelfRef.__VdfgRegularize_h6e95ff9d_0_2 = (((IData)(vlSelfRef.edge_cnn_top__DOT__conv1__DOT__engine__DOT__kr) 
                                                 + (IData)(vlSelfRef.edge_cnn_top__DOT__conv1__DOT__out_row)) 
                                                - (IData)(1U));
    vlSelfRef.__VdfgRegularize_h6e95ff9d_0_3 = (((IData)(vlSelfRef.edge_cnn_top__DOT__conv1__DOT__engine__DOT__kc) 
                                                 + (IData)(vlSelfRef.edge_cnn_top__DOT__conv1__DOT__out_col)) 
                                                - (IData)(1U));
    vlSelfRef.__VdfgRegularize_h6e95ff9d_0_4 = (((IData)(vlSelfRef.edge_cnn_top__DOT__conv2__DOT__engine__DOT__kr) 
                                                 + (IData)(vlSelfRef.edge_cnn_top__DOT__conv2__DOT__out_row)) 
                                                - (IData)(1U));
    vlSelfRef.__VdfgRegularize_h6e95ff9d_0_5 = (((IData)(vlSelfRef.edge_cnn_top__DOT__conv2__DOT__engine__DOT__kc) 
                                                 + (IData)(vlSelfRef.edge_cnn_top__DOT__conv2__DOT__out_col)) 
                                                - (IData)(1U));
    vlSelfRef.__VdfgRegularize_h6e95ff9d_0_6 = (((IData)(vlSelfRef.edge_cnn_top__DOT__conv3__DOT__engine__DOT__kr) 
                                                 + (IData)(vlSelfRef.edge_cnn_top__DOT__conv3__DOT__out_row)) 
                                                - (IData)(1U));
    vlSelfRef.__VdfgRegularize_h6e95ff9d_0_7 = (((IData)(vlSelfRef.edge_cnn_top__DOT__conv3__DOT__engine__DOT__kc) 
                                                 + (IData)(vlSelfRef.edge_cnn_top__DOT__conv3__DOT__out_col)) 
                                                - (IData)(1U));
    vlSelfRef.__VdfgRegularize_h6e95ff9d_0_8 = (((IData)(vlSelfRef.edge_cnn_top__DOT__conv4__DOT__engine__DOT__kr) 
                                                 + (IData)(vlSelfRef.edge_cnn_top__DOT__conv4__DOT__out_row)) 
                                                - (IData)(1U));
    vlSelfRef.__VdfgRegularize_h6e95ff9d_0_9 = (((IData)(vlSelfRef.edge_cnn_top__DOT__conv4__DOT__engine__DOT__kc) 
                                                 + (IData)(vlSelfRef.edge_cnn_top__DOT__conv4__DOT__out_col)) 
                                                - (IData)(1U));
    edge_cnn_top__DOT__conv1__DOT__quant__DOT__gen_round_shift__DOT__shifted_magnitude 
        = (0x000007ffffffffffULL & VL_SHIFTRS_QQI(43,43,32, 
                                                  (0x000007ffffffffffULL 
                                                   & ((1U 
                                                       & (IData)(
                                                                 (vlSelfRef.edge_cnn_top__DOT__conv1__DOT__product_reg 
                                                                  >> 0x0000002aU)))
                                                       ? 
                                                      (0x0000000000800000ULL 
                                                       + 
                                                       (- vlSelfRef.edge_cnn_top__DOT__conv1__DOT__product_reg))
                                                       : 
                                                      (0x0000000000800000ULL 
                                                       + vlSelfRef.edge_cnn_top__DOT__conv1__DOT__product_reg))), 0x00000018U));
    edge_cnn_top__DOT__conv2__DOT__quant__DOT__gen_round_shift__DOT__shifted_magnitude 
        = (0x000007ffffffffffULL & VL_SHIFTRS_QQI(43,43,32, 
                                                  (0x000007ffffffffffULL 
                                                   & ((1U 
                                                       & (IData)(
                                                                 (vlSelfRef.edge_cnn_top__DOT__conv2__DOT__product_reg 
                                                                  >> 0x0000002aU)))
                                                       ? 
                                                      (0x0000000000800000ULL 
                                                       + 
                                                       (- vlSelfRef.edge_cnn_top__DOT__conv2__DOT__product_reg))
                                                       : 
                                                      (0x0000000000800000ULL 
                                                       + vlSelfRef.edge_cnn_top__DOT__conv2__DOT__product_reg))), 0x00000018U));
    edge_cnn_top__DOT__conv3__DOT__quant__DOT__gen_round_shift__DOT__shifted_magnitude 
        = (0x000007ffffffffffULL & VL_SHIFTRS_QQI(43,43,32, 
                                                  (0x000007ffffffffffULL 
                                                   & ((1U 
                                                       & (IData)(
                                                                 (vlSelfRef.edge_cnn_top__DOT__conv3__DOT__product_reg 
                                                                  >> 0x0000002aU)))
                                                       ? 
                                                      (0x0000000000800000ULL 
                                                       + 
                                                       (- vlSelfRef.edge_cnn_top__DOT__conv3__DOT__product_reg))
                                                       : 
                                                      (0x0000000000800000ULL 
                                                       + vlSelfRef.edge_cnn_top__DOT__conv3__DOT__product_reg))), 0x00000018U));
    edge_cnn_top__DOT__conv4__DOT__quant__DOT__gen_round_shift__DOT__shifted_magnitude 
        = (0x000007ffffffffffULL & VL_SHIFTRS_QQI(43,43,32, 
                                                  (0x000007ffffffffffULL 
                                                   & ((1U 
                                                       & (IData)(
                                                                 (vlSelfRef.edge_cnn_top__DOT__conv4__DOT__product_reg 
                                                                  >> 0x0000002aU)))
                                                       ? 
                                                      (0x0000000000800000ULL 
                                                       + 
                                                       (- vlSelfRef.edge_cnn_top__DOT__conv4__DOT__product_reg))
                                                       : 
                                                      (0x0000000000800000ULL 
                                                       + vlSelfRef.edge_cnn_top__DOT__conv4__DOT__product_reg))), 0x00000018U));
    vlSelfRef.class_id = 0U;
    vlSelfRef.max_logit = vlSelfRef.edge_cnn_top__DOT__logit0;
    if (VL_GTS_III(32, vlSelfRef.edge_cnn_top__DOT__logit1, vlSelfRef.max_logit)) {
        vlSelfRef.max_logit = vlSelfRef.edge_cnn_top__DOT__logit1;
        vlSelfRef.class_id = 1U;
    }
    if (VL_GTS_III(32, vlSelfRef.edge_cnn_top__DOT__logit2, vlSelfRef.max_logit)) {
        vlSelfRef.max_logit = vlSelfRef.edge_cnn_top__DOT__logit2;
        vlSelfRef.class_id = 2U;
    }
    if (VL_GTS_III(32, vlSelfRef.edge_cnn_top__DOT__logit3, vlSelfRef.max_logit)) {
        vlSelfRef.max_logit = vlSelfRef.edge_cnn_top__DOT__logit3;
        vlSelfRef.class_id = 3U;
    }
    if (VL_GTS_III(32, vlSelfRef.edge_cnn_top__DOT__logit4, vlSelfRef.max_logit)) {
        vlSelfRef.max_logit = vlSelfRef.edge_cnn_top__DOT__logit4;
        vlSelfRef.class_id = 4U;
    }
    if (VL_GTS_III(32, vlSelfRef.edge_cnn_top__DOT__logit5, vlSelfRef.max_logit)) {
        vlSelfRef.max_logit = vlSelfRef.edge_cnn_top__DOT__logit5;
        vlSelfRef.class_id = 5U;
    }
    if (VL_GTS_III(32, vlSelfRef.edge_cnn_top__DOT__logit6, vlSelfRef.max_logit)) {
        vlSelfRef.max_logit = vlSelfRef.edge_cnn_top__DOT__logit6;
        vlSelfRef.class_id = 6U;
    }
    if (VL_GTS_III(32, vlSelfRef.edge_cnn_top__DOT__logit7, vlSelfRef.max_logit)) {
        vlSelfRef.max_logit = vlSelfRef.edge_cnn_top__DOT__logit7;
        vlSelfRef.class_id = 7U;
    }
    if (VL_GTS_III(32, vlSelfRef.edge_cnn_top__DOT__logit8, vlSelfRef.max_logit)) {
        vlSelfRef.max_logit = vlSelfRef.edge_cnn_top__DOT__logit8;
        vlSelfRef.class_id = 8U;
    }
    if (VL_GTS_III(32, vlSelfRef.edge_cnn_top__DOT__logit9, vlSelfRef.max_logit)) {
        vlSelfRef.max_logit = vlSelfRef.edge_cnn_top__DOT__logit9;
        vlSelfRef.class_id = 9U;
    }
    vlSelfRef.__VdfgRegularize_h6e95ff9d_0_0 = ((~ (IData)(vlSelfRef.busy)) 
                                                & ((~ 
                                                    ((IData)(vlSelfRef.edge_cnn_top__DOT__conv1__DOT__postprocess_pending) 
                                                     | ((IData)(vlSelfRef.edge_cnn_top__DOT__conv1__DOT__product_valid) 
                                                        | ((IData)(vlSelfRef.edge_cnn_top__DOT__conv1__DOT__postprocess_valid) 
                                                           | ((IData)(vlSelfRef.edge_cnn_top__DOT__conv2__DOT__postprocess_pending) 
                                                              | ((IData)(vlSelfRef.edge_cnn_top__DOT__conv2__DOT__product_valid) 
                                                                 | ((IData)(vlSelfRef.edge_cnn_top__DOT__conv2__DOT__postprocess_valid) 
                                                                    | ((IData)(vlSelfRef.edge_cnn_top__DOT__conv3__DOT__postprocess_pending) 
                                                                       | ((IData)(vlSelfRef.edge_cnn_top__DOT__conv3__DOT__product_valid) 
                                                                          | ((IData)(vlSelfRef.edge_cnn_top__DOT__conv3__DOT__postprocess_valid) 
                                                                             | ((IData)(vlSelfRef.edge_cnn_top__DOT__conv4__DOT__postprocess_pending) 
                                                                                | ((IData)(vlSelfRef.edge_cnn_top__DOT__conv4__DOT__product_valid) 
                                                                                | ((IData)(vlSelfRef.edge_cnn_top__DOT__conv4__DOT__postprocess_valid) 
                                                                                | ((IData)(vlSelfRef.edge_cnn_top__DOT__conv1__DOT__controller_busy) 
                                                                                | ((IData)(vlSelfRef.edge_cnn_top__DOT__conv2__DOT__controller_busy) 
                                                                                | ((IData)(vlSelfRef.edge_cnn_top__DOT__conv3__DOT__controller_busy) 
                                                                                | ((IData)(vlSelfRef.edge_cnn_top__DOT__conv4__DOT__controller_busy) 
                                                                                | ((IData)(vlSelfRef.edge_cnn_top__DOT__pool1_busy) 
                                                                                | ((IData)(vlSelfRef.edge_cnn_top__DOT__pool2_busy) 
                                                                                | ((IData)(vlSelfRef.edge_cnn_top__DOT__gap_busy) 
                                                                                | (IData)(vlSelfRef.edge_cnn_top__DOT__linear_busy))))))))))))))))))))) 
                                                   & (IData)(vlSelfRef.input_we)));
    vlSelfRef.edge_cnn_top__DOT__conv1__DOT__engine__DOT__valid_pixel 
        = (VL_LTES_III(32, 0U, vlSelfRef.__VdfgRegularize_h6e95ff9d_0_2) 
           & (VL_GTS_III(32, 0x00000020U, vlSelfRef.__VdfgRegularize_h6e95ff9d_0_2) 
              & (VL_LTES_III(32, 0U, vlSelfRef.__VdfgRegularize_h6e95ff9d_0_3) 
                 & VL_GTS_III(32, 0x00000020U, vlSelfRef.__VdfgRegularize_h6e95ff9d_0_3))));
    vlSelfRef.edge_cnn_top__DOT__conv2__DOT__engine__DOT__valid_pixel 
        = (VL_LTES_III(32, 0U, vlSelfRef.__VdfgRegularize_h6e95ff9d_0_4) 
           & (VL_GTS_III(32, 0x00000010U, vlSelfRef.__VdfgRegularize_h6e95ff9d_0_4) 
              & (VL_LTES_III(32, 0U, vlSelfRef.__VdfgRegularize_h6e95ff9d_0_5) 
                 & VL_GTS_III(32, 0x00000010U, vlSelfRef.__VdfgRegularize_h6e95ff9d_0_5))));
    vlSelfRef.edge_cnn_top__DOT__conv3__DOT__engine__DOT__valid_pixel 
        = (VL_LTES_III(32, 0U, vlSelfRef.__VdfgRegularize_h6e95ff9d_0_6) 
           & (VL_GTS_III(32, 8U, vlSelfRef.__VdfgRegularize_h6e95ff9d_0_6) 
              & (VL_LTES_III(32, 0U, vlSelfRef.__VdfgRegularize_h6e95ff9d_0_7) 
                 & VL_GTS_III(32, 8U, vlSelfRef.__VdfgRegularize_h6e95ff9d_0_7))));
    vlSelfRef.edge_cnn_top__DOT__conv4__DOT__engine__DOT__valid_pixel 
        = (VL_LTES_III(32, 0U, vlSelfRef.__VdfgRegularize_h6e95ff9d_0_8) 
           & (VL_GTS_III(32, 8U, vlSelfRef.__VdfgRegularize_h6e95ff9d_0_8) 
              & (VL_LTES_III(32, 0U, vlSelfRef.__VdfgRegularize_h6e95ff9d_0_9) 
                 & VL_GTS_III(32, 8U, vlSelfRef.__VdfgRegularize_h6e95ff9d_0_9))));
    edge_cnn_top__DOT__conv1__DOT__quant__DOT__result43 
        = (0x000007ffffffffffULL & ((1U & (IData)((vlSelfRef.edge_cnn_top__DOT__conv1__DOT__product_reg 
                                                   >> 0x0000002aU)))
                                     ? (- edge_cnn_top__DOT__conv1__DOT__quant__DOT__gen_round_shift__DOT__shifted_magnitude)
                                     : edge_cnn_top__DOT__conv1__DOT__quant__DOT__gen_round_shift__DOT__shifted_magnitude));
    edge_cnn_top__DOT__conv2__DOT__quant__DOT__result43 
        = (0x000007ffffffffffULL & ((1U & (IData)((vlSelfRef.edge_cnn_top__DOT__conv2__DOT__product_reg 
                                                   >> 0x0000002aU)))
                                     ? (- edge_cnn_top__DOT__conv2__DOT__quant__DOT__gen_round_shift__DOT__shifted_magnitude)
                                     : edge_cnn_top__DOT__conv2__DOT__quant__DOT__gen_round_shift__DOT__shifted_magnitude));
    edge_cnn_top__DOT__conv3__DOT__quant__DOT__result43 
        = (0x000007ffffffffffULL & ((1U & (IData)((vlSelfRef.edge_cnn_top__DOT__conv3__DOT__product_reg 
                                                   >> 0x0000002aU)))
                                     ? (- edge_cnn_top__DOT__conv3__DOT__quant__DOT__gen_round_shift__DOT__shifted_magnitude)
                                     : edge_cnn_top__DOT__conv3__DOT__quant__DOT__gen_round_shift__DOT__shifted_magnitude));
    edge_cnn_top__DOT__conv4__DOT__quant__DOT__result43 
        = (0x000007ffffffffffULL & ((1U & (IData)((vlSelfRef.edge_cnn_top__DOT__conv4__DOT__product_reg 
                                                   >> 0x0000002aU)))
                                     ? (- edge_cnn_top__DOT__conv4__DOT__quant__DOT__gen_round_shift__DOT__shifted_magnitude)
                                     : edge_cnn_top__DOT__conv4__DOT__quant__DOT__gen_round_shift__DOT__shifted_magnitude));
    vlSelfRef.edge_cnn_top__DOT__conv1__DOT__activated 
        = ((IData)(edge_cnn_top__DOT__conv1__DOT__quant__DOT__result43) 
           & (- (IData)((1U & (~ (IData)((edge_cnn_top__DOT__conv1__DOT__quant__DOT__result43 
                                          >> 0x0000001fU)))))));
    vlSelfRef.edge_cnn_top__DOT__conv2__DOT__activated 
        = ((IData)(edge_cnn_top__DOT__conv2__DOT__quant__DOT__result43) 
           & (- (IData)((1U & (~ (IData)((edge_cnn_top__DOT__conv2__DOT__quant__DOT__result43 
                                          >> 0x0000001fU)))))));
    vlSelfRef.edge_cnn_top__DOT__conv3__DOT__activated 
        = ((IData)(edge_cnn_top__DOT__conv3__DOT__quant__DOT__result43) 
           & (- (IData)((1U & (~ (IData)((edge_cnn_top__DOT__conv3__DOT__quant__DOT__result43 
                                          >> 0x0000001fU)))))));
    vlSelfRef.edge_cnn_top__DOT__conv4__DOT__activated 
        = ((IData)(edge_cnn_top__DOT__conv4__DOT__quant__DOT__result43) 
           & (- (IData)((1U & (~ (IData)((edge_cnn_top__DOT__conv4__DOT__quant__DOT__result43 
                                          >> 0x0000001fU)))))));
}

VL_ATTR_COLD void Vedge_cnn_top___024root___eval_stl(Vedge_cnn_top___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vedge_cnn_top___024root___eval_stl\n"); );
    Vedge_cnn_top__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    if ((1ULL & vlSelfRef.__VstlTriggered[0U])) {
        Vedge_cnn_top___024root___stl_sequent__TOP__0(vlSelf);
    }
}

VL_ATTR_COLD bool Vedge_cnn_top___024root___eval_phase__stl(Vedge_cnn_top___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vedge_cnn_top___024root___eval_phase__stl\n"); );
    Vedge_cnn_top__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Locals
    CData/*0:0*/ __VstlExecute;
    // Body
    Vedge_cnn_top___024root___eval_triggers_vec__stl(vlSelf);
#ifdef VL_DEBUG
    if (VL_UNLIKELY(vlSymsp->_vm_contextp__->debug())) {
        Vedge_cnn_top___024root___dump_triggers__stl(vlSelfRef.__VstlTriggered, "stl"s);
    }
#endif
    __VstlExecute = Vedge_cnn_top___024root___trigger_anySet__stl(vlSelfRef.__VstlTriggered);
    if (__VstlExecute) {
        Vedge_cnn_top___024root___eval_stl(vlSelf);
    }
    return (__VstlExecute);
}

bool Vedge_cnn_top___024root___trigger_anySet__ico(const VlUnpacked<QData/*63:0*/, 1> &in);

#ifdef VL_DEBUG
VL_ATTR_COLD void Vedge_cnn_top___024root___dump_triggers__ico(const VlUnpacked<QData/*63:0*/, 1> &triggers, const std::string &tag) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vedge_cnn_top___024root___dump_triggers__ico\n"); );
    // Body
    if ((1U & (~ (IData)(Vedge_cnn_top___024root___trigger_anySet__ico(triggers))))) {
        VL_DBG_MSGS("         No '" + tag + "' region triggers active\n");
    }
    if ((1U & (IData)(triggers[0U]))) {
        VL_DBG_MSGS("         '" + tag + "' region trigger index 0 is active: Internal 'ico' trigger - first iteration\n");
    }
}
#endif  // VL_DEBUG

bool Vedge_cnn_top___024root___trigger_anySet__act(const VlUnpacked<QData/*63:0*/, 1> &in);

#ifdef VL_DEBUG
VL_ATTR_COLD void Vedge_cnn_top___024root___dump_triggers__act(const VlUnpacked<QData/*63:0*/, 1> &triggers, const std::string &tag) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vedge_cnn_top___024root___dump_triggers__act\n"); );
    // Body
    if ((1U & (~ (IData)(Vedge_cnn_top___024root___trigger_anySet__act(triggers))))) {
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

VL_ATTR_COLD void Vedge_cnn_top___024root___ctor_var_reset(Vedge_cnn_top___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vedge_cnn_top___024root___ctor_var_reset\n"); );
    Vedge_cnn_top__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    const uint64_t __VscopeHash = VL_MURMUR64_HASH(vlSelf->vlNamep);
    vlSelf->clk = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 16707436170211756652ull);
    vlSelf->rst_n = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 1638864771569018232ull);
    vlSelf->start = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 9867861323841650631ull);
    vlSelf->input_we = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 7025069179568517235ull);
    vlSelf->input_addr = VL_SCOPED_RAND_RESET_I(19, __VscopeHash, 13892080179392794878ull);
    vlSelf->input_data = VL_SCOPED_RAND_RESET_I(8, __VscopeHash, 1923588759227995539ull);
    vlSelf->conv1_start = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 11235579007043326849ull);
    vlSelf->pool1_start = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 15317023131219154678ull);
    vlSelf->conv2_start = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 8056422041737375597ull);
    vlSelf->pool2_start = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 16555163847064179918ull);
    vlSelf->conv3_start = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 12229945295359645444ull);
    vlSelf->conv4_start = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 15188192825613325956ull);
    vlSelf->gap_start = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 5165500772251323459ull);
    vlSelf->linear_start = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 4258263890256102508ull);
    vlSelf->argmax_valid = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 17584079252057304459ull);
    vlSelf->busy = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 6386567572483775230ull);
    vlSelf->done = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 10296494685231209730ull);
    vlSelf->class_id = VL_SCOPED_RAND_RESET_I(4, __VscopeHash, 137756330502096589ull);
    vlSelf->max_logit = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 12814312630002674517ull);
    vlSelf->state_dbg = VL_SCOPED_RAND_RESET_I(4, __VscopeHash, 6042213990076480022ull);
    vlSelf->edge_cnn_top__DOT__pool1_done = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 5269317907690175772ull);
    vlSelf->edge_cnn_top__DOT__pool2_done = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 10709369787424084031ull);
    vlSelf->edge_cnn_top__DOT__gap_done = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 15056305777495457246ull);
    vlSelf->edge_cnn_top__DOT__linear_done = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 3982118975258997713ull);
    vlSelf->edge_cnn_top__DOT__pool1_busy = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 1744818054292312868ull);
    vlSelf->edge_cnn_top__DOT__pool2_busy = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 6500343081614107945ull);
    vlSelf->edge_cnn_top__DOT__gap_busy = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 12585800009956627587ull);
    vlSelf->edge_cnn_top__DOT__linear_busy = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 13049133866890954865ull);
    vlSelf->edge_cnn_top__DOT__buf_a_rdata = VL_SCOPED_RAND_RESET_I(8, __VscopeHash, 6531310415785960929ull);
    vlSelf->edge_cnn_top__DOT__buf_b_rdata = VL_SCOPED_RAND_RESET_I(8, __VscopeHash, 2444912163986676864ull);
    vlSelf->edge_cnn_top__DOT__conv4_weight_addr = VL_SCOPED_RAND_RESET_I(19, __VscopeHash, 6189855806271578381ull);
    vlSelf->edge_cnn_top__DOT__pool1_out_we = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 6945398593697818302ull);
    vlSelf->edge_cnn_top__DOT__pool2_out_we = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 1461328127548674612ull);
    vlSelf->edge_cnn_top__DOT__pool1_out_addr = VL_SCOPED_RAND_RESET_I(19, __VscopeHash, 17000488647240753176ull);
    vlSelf->edge_cnn_top__DOT__pool2_out_addr = VL_SCOPED_RAND_RESET_I(19, __VscopeHash, 2555390778101180796ull);
    vlSelf->edge_cnn_top__DOT__pool1_out_data = VL_SCOPED_RAND_RESET_I(8, __VscopeHash, 8746836995175648232ull);
    vlSelf->edge_cnn_top__DOT__pool2_out_data = VL_SCOPED_RAND_RESET_I(8, __VscopeHash, 15337302887903954602ull);
    vlSelf->edge_cnn_top__DOT__gap_out_we = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 6070091091534170049ull);
    vlSelf->edge_cnn_top__DOT__gap_out_addr = VL_SCOPED_RAND_RESET_I(19, __VscopeHash, 18174614249698203139ull);
    vlSelf->edge_cnn_top__DOT__gap_out_data = VL_SCOPED_RAND_RESET_I(8, __VscopeHash, 7839139366438733750ull);
    vlSelf->edge_cnn_top__DOT__linear_logit_index = VL_SCOPED_RAND_RESET_I(4, __VscopeHash, 3895502673897514945ull);
    vlSelf->edge_cnn_top__DOT__linear_logit_data = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 13942691493989434316ull);
    vlSelf->edge_cnn_top__DOT__linear_logit_valid = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 793675831704641731ull);
    vlSelf->edge_cnn_top__DOT__conv1_weight_data = VL_SCOPED_RAND_RESET_I(8, __VscopeHash, 4303117105454750540ull);
    vlSelf->edge_cnn_top__DOT__conv2_weight_data = VL_SCOPED_RAND_RESET_I(8, __VscopeHash, 12925480441825947220ull);
    vlSelf->edge_cnn_top__DOT__conv3_weight_data = VL_SCOPED_RAND_RESET_I(8, __VscopeHash, 17572370834361900978ull);
    vlSelf->edge_cnn_top__DOT__conv4_weight_data = VL_SCOPED_RAND_RESET_I(8, __VscopeHash, 5779140003210588204ull);
    vlSelf->edge_cnn_top__DOT__linear_weight_data = VL_SCOPED_RAND_RESET_I(8, __VscopeHash, 17479110579802913781ull);
    vlSelf->edge_cnn_top__DOT__logit0 = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 4474567513140544660ull);
    vlSelf->edge_cnn_top__DOT__logit1 = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 8466050561511916345ull);
    vlSelf->edge_cnn_top__DOT__logit2 = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 10355796813547033297ull);
    vlSelf->edge_cnn_top__DOT__logit3 = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 13730145474413548430ull);
    vlSelf->edge_cnn_top__DOT__logit4 = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 11209941462706728167ull);
    vlSelf->edge_cnn_top__DOT__logit5 = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 7212391854789675102ull);
    vlSelf->edge_cnn_top__DOT__logit6 = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 4347874581765277811ull);
    vlSelf->edge_cnn_top__DOT__logit7 = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 3704683206682315608ull);
    vlSelf->edge_cnn_top__DOT__logit8 = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 11526736341669356521ull);
    vlSelf->edge_cnn_top__DOT__logit9 = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 6059839425755469809ull);
    vlSelf->edge_cnn_top__DOT__fsm__DOT__state = VL_SCOPED_RAND_RESET_I(4, __VscopeHash, 2320010277808978129ull);
    for (int __Vi0 = 0; __Vi0 < 32768; ++__Vi0) {
        vlSelf->edge_cnn_top__DOT__buffer_a__DOT__mem[__Vi0] = VL_SCOPED_RAND_RESET_I(8, __VscopeHash, 15438809282233796321ull);
    }
    for (int __Vi0 = 0; __Vi0 < 32768; ++__Vi0) {
        vlSelf->edge_cnn_top__DOT__buffer_b__DOT__mem[__Vi0] = VL_SCOPED_RAND_RESET_I(8, __VscopeHash, 4858220899174400324ull);
    }
    for (int __Vi0 = 0; __Vi0 < 864; ++__Vi0) {
        vlSelf->edge_cnn_top__DOT__conv1_weight_rom__DOT__rom[__Vi0] = VL_SCOPED_RAND_RESET_I(8, __VscopeHash, 14227345512339078035ull);
    }
    vlSelf->edge_cnn_top__DOT__conv1_weight_rom__DOT__rom_addr = VL_SCOPED_RAND_RESET_I(10, __VscopeHash, 12056351679071777643ull);
    for (int __Vi0 = 0; __Vi0 < 18432; ++__Vi0) {
        vlSelf->edge_cnn_top__DOT__conv2_weight_rom__DOT__rom[__Vi0] = VL_SCOPED_RAND_RESET_I(8, __VscopeHash, 3334817742817538752ull);
    }
    vlSelf->edge_cnn_top__DOT__conv2_weight_rom__DOT__rom_addr = VL_SCOPED_RAND_RESET_I(15, __VscopeHash, 810006402730529827ull);
    for (int __Vi0 = 0; __Vi0 < 73728; ++__Vi0) {
        vlSelf->edge_cnn_top__DOT__conv3_weight_rom__DOT__rom[__Vi0] = VL_SCOPED_RAND_RESET_I(8, __VscopeHash, 8197506339063256546ull);
    }
    vlSelf->edge_cnn_top__DOT__conv3_weight_rom__DOT__rom_addr = VL_SCOPED_RAND_RESET_I(17, __VscopeHash, 2170858813359955122ull);
    for (int __Vi0 = 0; __Vi0 < 294912; ++__Vi0) {
        vlSelf->edge_cnn_top__DOT__conv4_weight_rom__DOT__rom[__Vi0] = VL_SCOPED_RAND_RESET_I(8, __VscopeHash, 9083313997616422115ull);
    }
    for (int __Vi0 = 0; __Vi0 < 2560; ++__Vi0) {
        vlSelf->edge_cnn_top__DOT__linear_weight_rom__DOT__rom[__Vi0] = VL_SCOPED_RAND_RESET_I(8, __VscopeHash, 16901761193936492388ull);
    }
    vlSelf->edge_cnn_top__DOT__linear_weight_rom__DOT__rom_addr = VL_SCOPED_RAND_RESET_I(12, __VscopeHash, 17835954074443387325ull);
    for (int __Vi0 = 0; __Vi0 < 32; ++__Vi0) {
        vlSelf->edge_cnn_top__DOT__conv1_bias_rom__DOT__rom[__Vi0] = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 1977165987477776154ull);
    }
    for (int __Vi0 = 0; __Vi0 < 64; ++__Vi0) {
        vlSelf->edge_cnn_top__DOT__conv2_bias_rom__DOT__rom[__Vi0] = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 2968891740587776846ull);
    }
    for (int __Vi0 = 0; __Vi0 < 128; ++__Vi0) {
        vlSelf->edge_cnn_top__DOT__conv3_bias_rom__DOT__rom[__Vi0] = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 10774168881239987134ull);
    }
    for (int __Vi0 = 0; __Vi0 < 256; ++__Vi0) {
        vlSelf->edge_cnn_top__DOT__conv4_bias_rom__DOT__rom[__Vi0] = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 12331670325567883481ull);
    }
    for (int __Vi0 = 0; __Vi0 < 10; ++__Vi0) {
        vlSelf->edge_cnn_top__DOT__linear_bias_rom__DOT__rom[__Vi0] = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 16187607202950808985ull);
    }
    vlSelf->edge_cnn_top__DOT__conv1__DOT__engine_start = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 12937750768467925875ull);
    vlSelf->edge_cnn_top__DOT__conv1__DOT__engine_done = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 962879527646751111ull);
    vlSelf->edge_cnn_top__DOT__conv1__DOT__controller_busy = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 7447069840868852753ull);
    vlSelf->edge_cnn_top__DOT__conv1__DOT__controller_done = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 18371674235900369853ull);
    vlSelf->edge_cnn_top__DOT__conv1__DOT__raw_sum = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 12578028554179756917ull);
    vlSelf->edge_cnn_top__DOT__conv1__DOT__biased_sum_reg = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 7833206173667585118ull);
    vlSelf->edge_cnn_top__DOT__conv1__DOT__product_reg = VL_SCOPED_RAND_RESET_Q(43, __VscopeHash, 630630004279838441ull);
    vlSelf->edge_cnn_top__DOT__conv1__DOT__activated = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 8632252634580121853ull);
    vlSelf->edge_cnn_top__DOT__conv1__DOT__out_row = VL_SCOPED_RAND_RESET_I(16, __VscopeHash, 9832081338747344640ull);
    vlSelf->edge_cnn_top__DOT__conv1__DOT__out_col = VL_SCOPED_RAND_RESET_I(16, __VscopeHash, 9078797352260534425ull);
    vlSelf->edge_cnn_top__DOT__conv1__DOT__out_channel = VL_SCOPED_RAND_RESET_I(16, __VscopeHash, 8492175810222433810ull);
    vlSelf->edge_cnn_top__DOT__conv1__DOT__controller_out_addr = VL_SCOPED_RAND_RESET_I(19, __VscopeHash, 4185646082150392897ull);
    vlSelf->edge_cnn_top__DOT__conv1__DOT__postprocess_pending = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 5706011548181806545ull);
    vlSelf->edge_cnn_top__DOT__conv1__DOT__product_valid = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 18030592361691428000ull);
    vlSelf->edge_cnn_top__DOT__conv1__DOT__postprocess_valid = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 3081609339051630312ull);
    vlSelf->edge_cnn_top__DOT__conv1__DOT__product_addr_reg = VL_SCOPED_RAND_RESET_I(19, __VscopeHash, 14224245277873951990ull);
    vlSelf->edge_cnn_top__DOT__conv1__DOT__out_data_reg = VL_SCOPED_RAND_RESET_I(8, __VscopeHash, 11782467351746492884ull);
    vlSelf->edge_cnn_top__DOT__conv1__DOT__out_addr_reg = VL_SCOPED_RAND_RESET_I(19, __VscopeHash, 13877050395572756585ull);
    vlSelf->edge_cnn_top__DOT__conv1__DOT__controller_done_d1 = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 10511926741426739326ull);
    vlSelf->edge_cnn_top__DOT__conv1__DOT__controller_done_d2 = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 13660428888701387798ull);
    vlSelf->edge_cnn_top__DOT__conv1__DOT__controller__DOT__state = VL_SCOPED_RAND_RESET_I(2, __VscopeHash, 16088280323084468718ull);
    vlSelf->edge_cnn_top__DOT__conv1__DOT__engine__DOT__state = VL_SCOPED_RAND_RESET_I(2, __VscopeHash, 7884122946927556907ull);
    vlSelf->edge_cnn_top__DOT__conv1__DOT__engine__DOT__ic = VL_SCOPED_RAND_RESET_I(16, __VscopeHash, 16257747179218931028ull);
    vlSelf->edge_cnn_top__DOT__conv1__DOT__engine__DOT__kr = VL_SCOPED_RAND_RESET_I(2, __VscopeHash, 11819326235211257363ull);
    vlSelf->edge_cnn_top__DOT__conv1__DOT__engine__DOT__kc = VL_SCOPED_RAND_RESET_I(2, __VscopeHash, 15591549599378375536ull);
    vlSelf->edge_cnn_top__DOT__conv1__DOT__engine__DOT__valid_pixel = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 11045858158570656206ull);
    vlSelf->edge_cnn_top__DOT__conv1__DOT__engine__DOT__product_reg = VL_SCOPED_RAND_RESET_I(16, __VscopeHash, 16790812833022757350ull);
    vlSelf->edge_cnn_top__DOT__conv1__DOT__engine__DOT__product_valid = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 1986152983863269264ull);
    vlSelf->edge_cnn_top__DOT__conv1__DOT__engine__DOT__product_last = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 3718493521956677410ull);
    vlSelf->edge_cnn_top__DOT__conv1__DOT__engine__DOT__last_item = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 4630540679460796279ull);
    vlSelf->edge_cnn_top__DOT__pool1__DOT__state = VL_SCOPED_RAND_RESET_I(3, __VscopeHash, 16381441559514703729ull);
    vlSelf->edge_cnn_top__DOT__pool1__DOT__ch = VL_SCOPED_RAND_RESET_I(16, __VscopeHash, 10329444951394140550ull);
    vlSelf->edge_cnn_top__DOT__pool1__DOT__row = VL_SCOPED_RAND_RESET_I(16, __VscopeHash, 15787912419826558215ull);
    vlSelf->edge_cnn_top__DOT__pool1__DOT__col = VL_SCOPED_RAND_RESET_I(16, __VscopeHash, 13795808924742771909ull);
    vlSelf->edge_cnn_top__DOT__pool1__DOT__tap = VL_SCOPED_RAND_RESET_I(2, __VscopeHash, 8464509465528900242ull);
    vlSelf->edge_cnn_top__DOT__pool1__DOT__current_max = VL_SCOPED_RAND_RESET_I(8, __VscopeHash, 8692764737278716283ull);
    vlSelf->edge_cnn_top__DOT__conv2__DOT__engine_start = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 11521205665068839723ull);
    vlSelf->edge_cnn_top__DOT__conv2__DOT__engine_done = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 12290511285534022810ull);
    vlSelf->edge_cnn_top__DOT__conv2__DOT__controller_busy = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 15362538097227495500ull);
    vlSelf->edge_cnn_top__DOT__conv2__DOT__controller_done = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 16309164648788332311ull);
    vlSelf->edge_cnn_top__DOT__conv2__DOT__raw_sum = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 13323181785175571898ull);
    vlSelf->edge_cnn_top__DOT__conv2__DOT__biased_sum_reg = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 13515727128538677580ull);
    vlSelf->edge_cnn_top__DOT__conv2__DOT__product_reg = VL_SCOPED_RAND_RESET_Q(43, __VscopeHash, 12115919578483807341ull);
    vlSelf->edge_cnn_top__DOT__conv2__DOT__activated = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 11446004664223579897ull);
    vlSelf->edge_cnn_top__DOT__conv2__DOT__out_row = VL_SCOPED_RAND_RESET_I(16, __VscopeHash, 5044101625657354855ull);
    vlSelf->edge_cnn_top__DOT__conv2__DOT__out_col = VL_SCOPED_RAND_RESET_I(16, __VscopeHash, 10775393739193401461ull);
    vlSelf->edge_cnn_top__DOT__conv2__DOT__out_channel = VL_SCOPED_RAND_RESET_I(16, __VscopeHash, 14955021401976893751ull);
    vlSelf->edge_cnn_top__DOT__conv2__DOT__controller_out_addr = VL_SCOPED_RAND_RESET_I(19, __VscopeHash, 14325919173166436336ull);
    vlSelf->edge_cnn_top__DOT__conv2__DOT__postprocess_pending = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 7085686952193713293ull);
    vlSelf->edge_cnn_top__DOT__conv2__DOT__product_valid = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 3745200797105918478ull);
    vlSelf->edge_cnn_top__DOT__conv2__DOT__postprocess_valid = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 2384159104866786696ull);
    vlSelf->edge_cnn_top__DOT__conv2__DOT__product_addr_reg = VL_SCOPED_RAND_RESET_I(19, __VscopeHash, 16665218553797742262ull);
    vlSelf->edge_cnn_top__DOT__conv2__DOT__out_data_reg = VL_SCOPED_RAND_RESET_I(8, __VscopeHash, 13556652996451749610ull);
    vlSelf->edge_cnn_top__DOT__conv2__DOT__out_addr_reg = VL_SCOPED_RAND_RESET_I(19, __VscopeHash, 4154943513231298049ull);
    vlSelf->edge_cnn_top__DOT__conv2__DOT__controller_done_d1 = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 7842712467480602717ull);
    vlSelf->edge_cnn_top__DOT__conv2__DOT__controller_done_d2 = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 8020939516437341364ull);
    vlSelf->edge_cnn_top__DOT__conv2__DOT__controller__DOT__state = VL_SCOPED_RAND_RESET_I(2, __VscopeHash, 3848272929658276549ull);
    vlSelf->edge_cnn_top__DOT__conv2__DOT__engine__DOT__state = VL_SCOPED_RAND_RESET_I(2, __VscopeHash, 17011480276034811085ull);
    vlSelf->edge_cnn_top__DOT__conv2__DOT__engine__DOT__ic = VL_SCOPED_RAND_RESET_I(16, __VscopeHash, 14929544132281157091ull);
    vlSelf->edge_cnn_top__DOT__conv2__DOT__engine__DOT__kr = VL_SCOPED_RAND_RESET_I(2, __VscopeHash, 9751240895012242325ull);
    vlSelf->edge_cnn_top__DOT__conv2__DOT__engine__DOT__kc = VL_SCOPED_RAND_RESET_I(2, __VscopeHash, 12453820814003869276ull);
    vlSelf->edge_cnn_top__DOT__conv2__DOT__engine__DOT__valid_pixel = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 3112563216728129917ull);
    vlSelf->edge_cnn_top__DOT__conv2__DOT__engine__DOT__product_reg = VL_SCOPED_RAND_RESET_I(16, __VscopeHash, 17147063843335621474ull);
    vlSelf->edge_cnn_top__DOT__conv2__DOT__engine__DOT__product_valid = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 16341853160394244074ull);
    vlSelf->edge_cnn_top__DOT__conv2__DOT__engine__DOT__product_last = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 16374172710970821734ull);
    vlSelf->edge_cnn_top__DOT__conv2__DOT__engine__DOT__last_item = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 228808267130729041ull);
    vlSelf->edge_cnn_top__DOT__pool2__DOT__state = VL_SCOPED_RAND_RESET_I(3, __VscopeHash, 16211064265754582105ull);
    vlSelf->edge_cnn_top__DOT__pool2__DOT__ch = VL_SCOPED_RAND_RESET_I(16, __VscopeHash, 24891680813482251ull);
    vlSelf->edge_cnn_top__DOT__pool2__DOT__row = VL_SCOPED_RAND_RESET_I(16, __VscopeHash, 6257966442668815346ull);
    vlSelf->edge_cnn_top__DOT__pool2__DOT__col = VL_SCOPED_RAND_RESET_I(16, __VscopeHash, 3111179029199249121ull);
    vlSelf->edge_cnn_top__DOT__pool2__DOT__tap = VL_SCOPED_RAND_RESET_I(2, __VscopeHash, 3401657894204166045ull);
    vlSelf->edge_cnn_top__DOT__pool2__DOT__current_max = VL_SCOPED_RAND_RESET_I(8, __VscopeHash, 12733592961672261788ull);
    vlSelf->edge_cnn_top__DOT__conv3__DOT__engine_start = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 14645260199327341736ull);
    vlSelf->edge_cnn_top__DOT__conv3__DOT__engine_done = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 16353299492006374217ull);
    vlSelf->edge_cnn_top__DOT__conv3__DOT__controller_busy = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 16231455051702767330ull);
    vlSelf->edge_cnn_top__DOT__conv3__DOT__controller_done = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 10515195653289131742ull);
    vlSelf->edge_cnn_top__DOT__conv3__DOT__raw_sum = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 5244961082929499276ull);
    vlSelf->edge_cnn_top__DOT__conv3__DOT__biased_sum_reg = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 8218543510567191882ull);
    vlSelf->edge_cnn_top__DOT__conv3__DOT__product_reg = VL_SCOPED_RAND_RESET_Q(43, __VscopeHash, 3342905087379214027ull);
    vlSelf->edge_cnn_top__DOT__conv3__DOT__activated = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 3140373662115121825ull);
    vlSelf->edge_cnn_top__DOT__conv3__DOT__out_row = VL_SCOPED_RAND_RESET_I(16, __VscopeHash, 2910539284550185637ull);
    vlSelf->edge_cnn_top__DOT__conv3__DOT__out_col = VL_SCOPED_RAND_RESET_I(16, __VscopeHash, 14707763314864515442ull);
    vlSelf->edge_cnn_top__DOT__conv3__DOT__out_channel = VL_SCOPED_RAND_RESET_I(16, __VscopeHash, 5977854382669740382ull);
    vlSelf->edge_cnn_top__DOT__conv3__DOT__controller_out_addr = VL_SCOPED_RAND_RESET_I(19, __VscopeHash, 8809602622505752926ull);
    vlSelf->edge_cnn_top__DOT__conv3__DOT__postprocess_pending = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 13076118066069720262ull);
    vlSelf->edge_cnn_top__DOT__conv3__DOT__product_valid = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 8382631645774157026ull);
    vlSelf->edge_cnn_top__DOT__conv3__DOT__postprocess_valid = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 3252214235563264911ull);
    vlSelf->edge_cnn_top__DOT__conv3__DOT__product_addr_reg = VL_SCOPED_RAND_RESET_I(19, __VscopeHash, 2126274761298738187ull);
    vlSelf->edge_cnn_top__DOT__conv3__DOT__out_data_reg = VL_SCOPED_RAND_RESET_I(8, __VscopeHash, 17633390214796643151ull);
    vlSelf->edge_cnn_top__DOT__conv3__DOT__out_addr_reg = VL_SCOPED_RAND_RESET_I(19, __VscopeHash, 5978096508624072884ull);
    vlSelf->edge_cnn_top__DOT__conv3__DOT__controller_done_d1 = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 9530767009621649555ull);
    vlSelf->edge_cnn_top__DOT__conv3__DOT__controller_done_d2 = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 4436511661725480487ull);
    vlSelf->edge_cnn_top__DOT__conv3__DOT__controller__DOT__state = VL_SCOPED_RAND_RESET_I(2, __VscopeHash, 3683598917807255579ull);
    vlSelf->edge_cnn_top__DOT__conv3__DOT__engine__DOT__state = VL_SCOPED_RAND_RESET_I(2, __VscopeHash, 7880756888303291396ull);
    vlSelf->edge_cnn_top__DOT__conv3__DOT__engine__DOT__ic = VL_SCOPED_RAND_RESET_I(16, __VscopeHash, 11141331060833705217ull);
    vlSelf->edge_cnn_top__DOT__conv3__DOT__engine__DOT__kr = VL_SCOPED_RAND_RESET_I(2, __VscopeHash, 7667597415028446196ull);
    vlSelf->edge_cnn_top__DOT__conv3__DOT__engine__DOT__kc = VL_SCOPED_RAND_RESET_I(2, __VscopeHash, 1734028679094034458ull);
    vlSelf->edge_cnn_top__DOT__conv3__DOT__engine__DOT__valid_pixel = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 11797888370383317774ull);
    vlSelf->edge_cnn_top__DOT__conv3__DOT__engine__DOT__product_reg = VL_SCOPED_RAND_RESET_I(16, __VscopeHash, 1824671137794102711ull);
    vlSelf->edge_cnn_top__DOT__conv3__DOT__engine__DOT__product_valid = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 530024335621770084ull);
    vlSelf->edge_cnn_top__DOT__conv3__DOT__engine__DOT__product_last = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 8775383016209475255ull);
    vlSelf->edge_cnn_top__DOT__conv3__DOT__engine__DOT__last_item = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 14330460610518775125ull);
    vlSelf->edge_cnn_top__DOT__conv4__DOT__engine_start = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 2239329822878681614ull);
    vlSelf->edge_cnn_top__DOT__conv4__DOT__engine_done = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 5196932750743359325ull);
    vlSelf->edge_cnn_top__DOT__conv4__DOT__controller_busy = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 12756913280943030081ull);
    vlSelf->edge_cnn_top__DOT__conv4__DOT__controller_done = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 14731534522893061290ull);
    vlSelf->edge_cnn_top__DOT__conv4__DOT__raw_sum = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 6379308420626684295ull);
    vlSelf->edge_cnn_top__DOT__conv4__DOT__biased_sum_reg = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 16670370628871069213ull);
    vlSelf->edge_cnn_top__DOT__conv4__DOT__product_reg = VL_SCOPED_RAND_RESET_Q(43, __VscopeHash, 5765438357394903731ull);
    vlSelf->edge_cnn_top__DOT__conv4__DOT__activated = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 5005781510834364433ull);
    vlSelf->edge_cnn_top__DOT__conv4__DOT__out_row = VL_SCOPED_RAND_RESET_I(16, __VscopeHash, 9929727686062842498ull);
    vlSelf->edge_cnn_top__DOT__conv4__DOT__out_col = VL_SCOPED_RAND_RESET_I(16, __VscopeHash, 16777043910355246698ull);
    vlSelf->edge_cnn_top__DOT__conv4__DOT__out_channel = VL_SCOPED_RAND_RESET_I(16, __VscopeHash, 2094237632941702383ull);
    vlSelf->edge_cnn_top__DOT__conv4__DOT__controller_out_addr = VL_SCOPED_RAND_RESET_I(19, __VscopeHash, 3869225767351267907ull);
    vlSelf->edge_cnn_top__DOT__conv4__DOT__postprocess_pending = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 9129891457913391876ull);
    vlSelf->edge_cnn_top__DOT__conv4__DOT__product_valid = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 13874921777461704894ull);
    vlSelf->edge_cnn_top__DOT__conv4__DOT__postprocess_valid = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 12800036418035522880ull);
    vlSelf->edge_cnn_top__DOT__conv4__DOT__product_addr_reg = VL_SCOPED_RAND_RESET_I(19, __VscopeHash, 17779178463997431425ull);
    vlSelf->edge_cnn_top__DOT__conv4__DOT__out_data_reg = VL_SCOPED_RAND_RESET_I(8, __VscopeHash, 2773435631554538393ull);
    vlSelf->edge_cnn_top__DOT__conv4__DOT__out_addr_reg = VL_SCOPED_RAND_RESET_I(19, __VscopeHash, 11936170693072046740ull);
    vlSelf->edge_cnn_top__DOT__conv4__DOT__controller_done_d1 = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 16280502484848489310ull);
    vlSelf->edge_cnn_top__DOT__conv4__DOT__controller_done_d2 = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 2264816129948325353ull);
    vlSelf->edge_cnn_top__DOT__conv4__DOT__controller__DOT__state = VL_SCOPED_RAND_RESET_I(2, __VscopeHash, 14007085334179892035ull);
    vlSelf->edge_cnn_top__DOT__conv4__DOT__engine__DOT__state = VL_SCOPED_RAND_RESET_I(2, __VscopeHash, 14621478333659066108ull);
    vlSelf->edge_cnn_top__DOT__conv4__DOT__engine__DOT__ic = VL_SCOPED_RAND_RESET_I(16, __VscopeHash, 17286761737897908480ull);
    vlSelf->edge_cnn_top__DOT__conv4__DOT__engine__DOT__kr = VL_SCOPED_RAND_RESET_I(2, __VscopeHash, 16427564397549966808ull);
    vlSelf->edge_cnn_top__DOT__conv4__DOT__engine__DOT__kc = VL_SCOPED_RAND_RESET_I(2, __VscopeHash, 12608643090904756531ull);
    vlSelf->edge_cnn_top__DOT__conv4__DOT__engine__DOT__valid_pixel = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 12882659618348864531ull);
    vlSelf->edge_cnn_top__DOT__conv4__DOT__engine__DOT__product_reg = VL_SCOPED_RAND_RESET_I(16, __VscopeHash, 9165826072461027020ull);
    vlSelf->edge_cnn_top__DOT__conv4__DOT__engine__DOT__product_valid = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 3524428007037286828ull);
    vlSelf->edge_cnn_top__DOT__conv4__DOT__engine__DOT__product_last = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 5854611857099301538ull);
    vlSelf->edge_cnn_top__DOT__conv4__DOT__engine__DOT__last_item = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 7086082298678851504ull);
    vlSelf->edge_cnn_top__DOT__gap__DOT__state = VL_SCOPED_RAND_RESET_I(3, __VscopeHash, 11848075192513517033ull);
    vlSelf->edge_cnn_top__DOT__gap__DOT__ch = VL_SCOPED_RAND_RESET_I(16, __VscopeHash, 7343865572415631388ull);
    vlSelf->edge_cnn_top__DOT__gap__DOT__idx = VL_SCOPED_RAND_RESET_I(16, __VscopeHash, 7450659159286722553ull);
    vlSelf->edge_cnn_top__DOT__gap__DOT__sum = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 9235633743860122670ull);
    vlSelf->edge_cnn_top__DOT__linear__DOT__state = VL_SCOPED_RAND_RESET_I(3, __VscopeHash, 5950476183485114914ull);
    vlSelf->edge_cnn_top__DOT__linear__DOT__class_idx = VL_SCOPED_RAND_RESET_I(4, __VscopeHash, 1972693651203733358ull);
    vlSelf->edge_cnn_top__DOT__linear__DOT__feature_idx = VL_SCOPED_RAND_RESET_I(8, __VscopeHash, 2572852535955626771ull);
    vlSelf->edge_cnn_top__DOT__linear__DOT__acc = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 14191408427040738979ull);
    vlSelf->__VdfgRegularize_h6e95ff9d_0_0 = 0;
    vlSelf->__VdfgRegularize_h6e95ff9d_0_2 = 0;
    vlSelf->__VdfgRegularize_h6e95ff9d_0_3 = 0;
    vlSelf->__VdfgRegularize_h6e95ff9d_0_4 = 0;
    vlSelf->__VdfgRegularize_h6e95ff9d_0_5 = 0;
    vlSelf->__VdfgRegularize_h6e95ff9d_0_6 = 0;
    vlSelf->__VdfgRegularize_h6e95ff9d_0_7 = 0;
    vlSelf->__VdfgRegularize_h6e95ff9d_0_8 = 0;
    vlSelf->__VdfgRegularize_h6e95ff9d_0_9 = 0;
    vlSelf->__VdfgRegularize_h6e95ff9d_0_10 = 0;
    vlSelf->__Vdly__conv1_start = 0;
    vlSelf->__Vdly__pool1_start = 0;
    vlSelf->__Vdly__conv2_start = 0;
    vlSelf->__Vdly__pool2_start = 0;
    vlSelf->__Vdly__conv3_start = 0;
    vlSelf->__Vdly__conv4_start = 0;
    vlSelf->__Vdly__gap_start = 0;
    vlSelf->__Vdly__state_dbg = 0;
    vlSelf->__Vdly__edge_cnn_top__DOT__conv1__DOT__engine_start = 0;
    vlSelf->__Vdly__edge_cnn_top__DOT__conv1__DOT__controller__DOT__state = 0;
    vlSelf->__Vdly__edge_cnn_top__DOT__conv1__DOT__engine__DOT__state = 0;
    vlSelf->__Vdly__edge_cnn_top__DOT__conv1__DOT__engine__DOT__ic = 0;
    vlSelf->__Vdly__edge_cnn_top__DOT__conv1__DOT__engine__DOT__product_reg = 0;
    vlSelf->__Vdly__edge_cnn_top__DOT__conv1__DOT__engine__DOT__product_valid = 0;
    vlSelf->__Vdly__edge_cnn_top__DOT__conv1__DOT__engine__DOT__product_last = 0;
    vlSelf->__Vdly__edge_cnn_top__DOT__pool1__DOT__state = 0;
    vlSelf->__Vdly__edge_cnn_top__DOT__pool1__DOT__tap = 0;
    vlSelf->__Vdly__edge_cnn_top__DOT__pool1__DOT__col = 0;
    vlSelf->__Vdly__edge_cnn_top__DOT__pool1__DOT__row = 0;
    vlSelf->__Vdly__edge_cnn_top__DOT__pool1__DOT__ch = 0;
    vlSelf->__Vdly__edge_cnn_top__DOT__pool1__DOT__current_max = 0;
    vlSelf->__Vdly__edge_cnn_top__DOT__conv2__DOT__engine_start = 0;
    vlSelf->__Vdly__edge_cnn_top__DOT__conv2__DOT__controller__DOT__state = 0;
    vlSelf->__Vdly__edge_cnn_top__DOT__conv2__DOT__engine__DOT__state = 0;
    vlSelf->__Vdly__edge_cnn_top__DOT__conv2__DOT__engine__DOT__ic = 0;
    vlSelf->__Vdly__edge_cnn_top__DOT__conv2__DOT__engine__DOT__product_reg = 0;
    vlSelf->__Vdly__edge_cnn_top__DOT__conv2__DOT__engine__DOT__product_valid = 0;
    vlSelf->__Vdly__edge_cnn_top__DOT__conv2__DOT__engine__DOT__product_last = 0;
    vlSelf->__Vdly__edge_cnn_top__DOT__pool2__DOT__state = 0;
    vlSelf->__Vdly__edge_cnn_top__DOT__pool2__DOT__tap = 0;
    vlSelf->__Vdly__edge_cnn_top__DOT__pool2__DOT__col = 0;
    vlSelf->__Vdly__edge_cnn_top__DOT__pool2__DOT__row = 0;
    vlSelf->__Vdly__edge_cnn_top__DOT__pool2__DOT__ch = 0;
    vlSelf->__Vdly__edge_cnn_top__DOT__pool2__DOT__current_max = 0;
    vlSelf->__Vdly__edge_cnn_top__DOT__conv3__DOT__engine_start = 0;
    vlSelf->__Vdly__edge_cnn_top__DOT__conv3__DOT__controller__DOT__state = 0;
    vlSelf->__Vdly__edge_cnn_top__DOT__conv3__DOT__engine__DOT__state = 0;
    vlSelf->__Vdly__edge_cnn_top__DOT__conv3__DOT__engine__DOT__ic = 0;
    vlSelf->__Vdly__edge_cnn_top__DOT__conv3__DOT__engine__DOT__product_reg = 0;
    vlSelf->__Vdly__edge_cnn_top__DOT__conv3__DOT__engine__DOT__product_valid = 0;
    vlSelf->__Vdly__edge_cnn_top__DOT__conv3__DOT__engine__DOT__product_last = 0;
    vlSelf->__Vdly__edge_cnn_top__DOT__conv4__DOT__engine_start = 0;
    vlSelf->__Vdly__edge_cnn_top__DOT__conv4__DOT__controller__DOT__state = 0;
    vlSelf->__Vdly__edge_cnn_top__DOT__conv4__DOT__engine__DOT__state = 0;
    vlSelf->__Vdly__edge_cnn_top__DOT__conv4__DOT__engine__DOT__ic = 0;
    vlSelf->__Vdly__edge_cnn_top__DOT__conv4__DOT__engine__DOT__product_reg = 0;
    vlSelf->__Vdly__edge_cnn_top__DOT__conv4__DOT__engine__DOT__product_valid = 0;
    vlSelf->__Vdly__edge_cnn_top__DOT__conv4__DOT__engine__DOT__product_last = 0;
    vlSelf->__Vdly__edge_cnn_top__DOT__gap__DOT__state = 0;
    vlSelf->__Vdly__edge_cnn_top__DOT__gap__DOT__idx = 0;
    vlSelf->__Vdly__edge_cnn_top__DOT__gap__DOT__ch = 0;
    vlSelf->__Vdly__edge_cnn_top__DOT__gap__DOT__sum = 0;
    vlSelf->__Vdly__edge_cnn_top__DOT__linear__DOT__feature_idx = 0;
    vlSelf->__VdlyVal__edge_cnn_top__DOT__buffer_a__DOT__mem__v0 = 0;
    vlSelf->__VdlyDim0__edge_cnn_top__DOT__buffer_a__DOT__mem__v0 = 0;
    vlSelf->__VdlySet__edge_cnn_top__DOT__buffer_a__DOT__mem__v0 = 0;
    vlSelf->__VdlyVal__edge_cnn_top__DOT__buffer_b__DOT__mem__v0 = 0;
    vlSelf->__VdlyDim0__edge_cnn_top__DOT__buffer_b__DOT__mem__v0 = 0;
    vlSelf->__VdlySet__edge_cnn_top__DOT__buffer_b__DOT__mem__v0 = 0;
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
