// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design implementation internals
// See Vd2_small_top.h for the primary calling header

#include "Vd2_small_top__pch.h"

VL_ATTR_COLD void Vd2_small_top___024root___eval_static(Vd2_small_top___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vd2_small_top___024root___eval_static\n"); );
    Vd2_small_top__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    vlSelfRef.__Vtrigprevexpr___TOP__clk__0 = vlSelfRef.clk;
    vlSelfRef.__Vtrigprevexpr___TOP__rst_n__0 = vlSelfRef.rst_n;
}

VL_ATTR_COLD void Vd2_small_top___024root___eval_initial(Vd2_small_top___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vd2_small_top___024root___eval_initial\n"); );
    Vd2_small_top__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    VL_READMEM_N(true, 8, 432, 0, "artifacts/int8_params/conv1_weight.mem"s
                 ,  &(vlSelfRef.d2_small_top__DOT__conv1_weight_rom__DOT__rom)
                 , 0, ~0ULL);
    VL_READMEM_N(true, 8, 4608, 0, "artifacts/int8_params/conv2_weight.mem"s
                 ,  &(vlSelfRef.d2_small_top__DOT__conv2_weight_rom__DOT__rom)
                 , 0, ~0ULL);
    VL_READMEM_N(true, 8, 18432, 0, "artifacts/int8_params/conv3_weight.mem"s
                 ,  &(vlSelfRef.d2_small_top__DOT__conv3_weight_rom__DOT__rom)
                 , 0, ~0ULL);
    VL_READMEM_N(true, 8, 640, 0, "artifacts/int8_params/linear_weight.mem"s
                 ,  &(vlSelfRef.d2_small_top__DOT__linear_weight_rom__DOT__rom)
                 , 0, ~0ULL);
    VL_READMEM_N(true, 32, 16, 0, "artifacts/int8_params/conv1_bias.mem"s
                 ,  &(vlSelfRef.d2_small_top__DOT__conv1_bias_rom__DOT__rom)
                 , 0, ~0ULL);
    VL_READMEM_N(true, 32, 32, 0, "artifacts/int8_params/conv2_bias.mem"s
                 ,  &(vlSelfRef.d2_small_top__DOT__conv2_bias_rom__DOT__rom)
                 , 0, ~0ULL);
    VL_READMEM_N(true, 32, 64, 0, "artifacts/int8_params/conv3_bias.mem"s
                 ,  &(vlSelfRef.d2_small_top__DOT__conv3_bias_rom__DOT__rom)
                 , 0, ~0ULL);
    VL_READMEM_N(true, 32, 10, 0, "artifacts/int8_params/linear_bias.mem"s
                 ,  &(vlSelfRef.d2_small_top__DOT__linear_bias_rom__DOT__rom)
                 , 0, ~0ULL);
}

VL_ATTR_COLD void Vd2_small_top___024root___eval_initial__TOP(Vd2_small_top___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vd2_small_top___024root___eval_initial__TOP\n"); );
    Vd2_small_top__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    VL_READMEM_N(true, 8, 432, 0, "artifacts/int8_params/conv1_weight.mem"s
                 ,  &(vlSelfRef.d2_small_top__DOT__conv1_weight_rom__DOT__rom)
                 , 0, ~0ULL);
    VL_READMEM_N(true, 8, 4608, 0, "artifacts/int8_params/conv2_weight.mem"s
                 ,  &(vlSelfRef.d2_small_top__DOT__conv2_weight_rom__DOT__rom)
                 , 0, ~0ULL);
    VL_READMEM_N(true, 8, 18432, 0, "artifacts/int8_params/conv3_weight.mem"s
                 ,  &(vlSelfRef.d2_small_top__DOT__conv3_weight_rom__DOT__rom)
                 , 0, ~0ULL);
    VL_READMEM_N(true, 8, 640, 0, "artifacts/int8_params/linear_weight.mem"s
                 ,  &(vlSelfRef.d2_small_top__DOT__linear_weight_rom__DOT__rom)
                 , 0, ~0ULL);
    VL_READMEM_N(true, 32, 16, 0, "artifacts/int8_params/conv1_bias.mem"s
                 ,  &(vlSelfRef.d2_small_top__DOT__conv1_bias_rom__DOT__rom)
                 , 0, ~0ULL);
    VL_READMEM_N(true, 32, 32, 0, "artifacts/int8_params/conv2_bias.mem"s
                 ,  &(vlSelfRef.d2_small_top__DOT__conv2_bias_rom__DOT__rom)
                 , 0, ~0ULL);
    VL_READMEM_N(true, 32, 64, 0, "artifacts/int8_params/conv3_bias.mem"s
                 ,  &(vlSelfRef.d2_small_top__DOT__conv3_bias_rom__DOT__rom)
                 , 0, ~0ULL);
    VL_READMEM_N(true, 32, 10, 0, "artifacts/int8_params/linear_bias.mem"s
                 ,  &(vlSelfRef.d2_small_top__DOT__linear_bias_rom__DOT__rom)
                 , 0, ~0ULL);
}

VL_ATTR_COLD void Vd2_small_top___024root___eval_final(Vd2_small_top___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vd2_small_top___024root___eval_final\n"); );
    Vd2_small_top__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
}

#ifdef VL_DEBUG
VL_ATTR_COLD void Vd2_small_top___024root___dump_triggers__stl(const VlUnpacked<QData/*63:0*/, 1> &triggers, const std::string &tag);
#endif  // VL_DEBUG
VL_ATTR_COLD bool Vd2_small_top___024root___eval_phase__stl(Vd2_small_top___024root* vlSelf);

VL_ATTR_COLD void Vd2_small_top___024root___eval_settle(Vd2_small_top___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vd2_small_top___024root___eval_settle\n"); );
    Vd2_small_top__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Locals
    IData/*31:0*/ __VstlIterCount;
    // Body
    __VstlIterCount = 0U;
    vlSelfRef.__VstlFirstIteration = 1U;
    do {
        if (VL_UNLIKELY(((0x00002710U < __VstlIterCount)))) {
#ifdef VL_DEBUG
            Vd2_small_top___024root___dump_triggers__stl(vlSelfRef.__VstlTriggered, "stl"s);
#endif
            VL_FATAL_MT("rtl/src/d2_small_top.v", 21, "", "DIDNOTCONVERGE: Settle region did not converge after '--converge-limit' of 10000 tries");
        }
        __VstlIterCount = ((IData)(1U) + __VstlIterCount);
        vlSelfRef.__VstlPhaseResult = Vd2_small_top___024root___eval_phase__stl(vlSelf);
        vlSelfRef.__VstlFirstIteration = 0U;
    } while (vlSelfRef.__VstlPhaseResult);
}

VL_ATTR_COLD void Vd2_small_top___024root___eval_triggers_vec__stl(Vd2_small_top___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vd2_small_top___024root___eval_triggers_vec__stl\n"); );
    Vd2_small_top__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    vlSelfRef.__VstlTriggered[0U] = ((0xfffffffffffffffeULL 
                                      & vlSelfRef.__VstlTriggered[0U]) 
                                     | (IData)((IData)(vlSelfRef.__VstlFirstIteration)));
}

VL_ATTR_COLD bool Vd2_small_top___024root___trigger_anySet__stl(const VlUnpacked<QData/*63:0*/, 1> &in);

#ifdef VL_DEBUG
VL_ATTR_COLD void Vd2_small_top___024root___dump_triggers__stl(const VlUnpacked<QData/*63:0*/, 1> &triggers, const std::string &tag) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vd2_small_top___024root___dump_triggers__stl\n"); );
    // Body
    if ((1U & (~ (IData)(Vd2_small_top___024root___trigger_anySet__stl(triggers))))) {
        VL_DBG_MSGS("         No '" + tag + "' region triggers active\n");
    }
    if ((1U & (IData)(triggers[0U]))) {
        VL_DBG_MSGS("         '" + tag + "' region trigger index 0 is active: Internal 'stl' trigger - first iteration\n");
    }
}
#endif  // VL_DEBUG

VL_ATTR_COLD bool Vd2_small_top___024root___trigger_anySet__stl(const VlUnpacked<QData/*63:0*/, 1> &in) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vd2_small_top___024root___trigger_anySet__stl\n"); );
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

VL_ATTR_COLD void Vd2_small_top___024root___stl_sequent__TOP__0(Vd2_small_top___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vd2_small_top___024root___stl_sequent__TOP__0\n"); );
    Vd2_small_top__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Locals
    QData/*42:0*/ d2_small_top__DOT__conv1__DOT__quant__DOT__result43;
    d2_small_top__DOT__conv1__DOT__quant__DOT__result43 = 0;
    QData/*42:0*/ d2_small_top__DOT__conv1__DOT__quant__DOT__gen_round_shift__DOT__shifted_magnitude;
    d2_small_top__DOT__conv1__DOT__quant__DOT__gen_round_shift__DOT__shifted_magnitude = 0;
    QData/*42:0*/ d2_small_top__DOT__conv2__DOT__quant__DOT__result43;
    d2_small_top__DOT__conv2__DOT__quant__DOT__result43 = 0;
    QData/*42:0*/ d2_small_top__DOT__conv2__DOT__quant__DOT__gen_round_shift__DOT__shifted_magnitude;
    d2_small_top__DOT__conv2__DOT__quant__DOT__gen_round_shift__DOT__shifted_magnitude = 0;
    QData/*42:0*/ d2_small_top__DOT__conv3__DOT__quant__DOT__result43;
    d2_small_top__DOT__conv3__DOT__quant__DOT__result43 = 0;
    QData/*42:0*/ d2_small_top__DOT__conv3__DOT__quant__DOT__gen_round_shift__DOT__shifted_magnitude;
    d2_small_top__DOT__conv3__DOT__quant__DOT__gen_round_shift__DOT__shifted_magnitude = 0;
    // Body
    vlSelfRef.d2_small_top__DOT__linear_weight_rom__DOT__rom_addr 
        = (0x000003ffU & ((IData)(vlSelfRef.d2_small_top__DOT__linear__DOT__feature_idx) 
                          + ((IData)(vlSelfRef.d2_small_top__DOT__linear__DOT__class_idx) 
                             << 6U)));
    vlSelfRef.d2_small_top__DOT__conv1__DOT__engine__DOT__last_item 
        = ((2U == (IData)(vlSelfRef.d2_small_top__DOT__conv1__DOT__engine__DOT__ic)) 
           & ((2U == (IData)(vlSelfRef.d2_small_top__DOT__conv1__DOT__engine__DOT__kr)) 
              & (2U == (IData)(vlSelfRef.d2_small_top__DOT__conv1__DOT__engine__DOT__kc))));
    vlSelfRef.d2_small_top__DOT__conv2__DOT__engine__DOT__last_item 
        = ((0x000fU == (IData)(vlSelfRef.d2_small_top__DOT__conv2__DOT__engine__DOT__ic)) 
           & ((2U == (IData)(vlSelfRef.d2_small_top__DOT__conv2__DOT__engine__DOT__kr)) 
              & (2U == (IData)(vlSelfRef.d2_small_top__DOT__conv2__DOT__engine__DOT__kc))));
    vlSelfRef.d2_small_top__DOT__conv3__DOT__engine__DOT__last_item 
        = ((0x001fU == (IData)(vlSelfRef.d2_small_top__DOT__conv3__DOT__engine__DOT__ic)) 
           & ((2U == (IData)(vlSelfRef.d2_small_top__DOT__conv3__DOT__engine__DOT__kr)) 
              & (2U == (IData)(vlSelfRef.d2_small_top__DOT__conv3__DOT__engine__DOT__kc))));
    vlSelfRef.d2_small_top__DOT__conv1_weight_rom__DOT__rom_addr 
        = (0x000001ffU & ((IData)(vlSelfRef.d2_small_top__DOT__conv1__DOT__engine__DOT__kc) 
                          + (((IData)(0x0000001bU) 
                              * (IData)(vlSelfRef.d2_small_top__DOT__conv1__DOT__out_channel)) 
                             + (((IData)(9U) * (IData)(vlSelfRef.d2_small_top__DOT__conv1__DOT__engine__DOT__ic)) 
                                + ((IData)(3U) * (IData)(vlSelfRef.d2_small_top__DOT__conv1__DOT__engine__DOT__kr))))));
    vlSelfRef.d2_small_top__DOT__conv2_weight_rom__DOT__rom_addr 
        = (0x00001fffU & ((IData)(vlSelfRef.d2_small_top__DOT__conv2__DOT__engine__DOT__kc) 
                          + (((IData)(9U) * VL_SHIFTL_III(32,32,32, (IData)(vlSelfRef.d2_small_top__DOT__conv2__DOT__out_channel), 4U)) 
                             + (((IData)(9U) * (IData)(vlSelfRef.d2_small_top__DOT__conv2__DOT__engine__DOT__ic)) 
                                + ((IData)(3U) * (IData)(vlSelfRef.d2_small_top__DOT__conv2__DOT__engine__DOT__kr))))));
    vlSelfRef.d2_small_top__DOT__conv3_weight_rom__DOT__rom_addr 
        = (0x00007fffU & ((IData)(vlSelfRef.d2_small_top__DOT__conv3__DOT__engine__DOT__kc) 
                          + (((IData)(9U) * VL_SHIFTL_III(32,32,32, (IData)(vlSelfRef.d2_small_top__DOT__conv3__DOT__out_channel), 5U)) 
                             + (((IData)(9U) * (IData)(vlSelfRef.d2_small_top__DOT__conv3__DOT__engine__DOT__ic)) 
                                + ((IData)(3U) * (IData)(vlSelfRef.d2_small_top__DOT__conv3__DOT__engine__DOT__kr))))));
    vlSelfRef.__VdfgRegularize_h6e95ff9d_0_2 = (((IData)(vlSelfRef.d2_small_top__DOT__conv1__DOT__engine__DOT__kr) 
                                                 + (IData)(vlSelfRef.d2_small_top__DOT__conv1__DOT__out_row)) 
                                                - (IData)(1U));
    vlSelfRef.__VdfgRegularize_h6e95ff9d_0_3 = (((IData)(vlSelfRef.d2_small_top__DOT__conv1__DOT__engine__DOT__kc) 
                                                 + (IData)(vlSelfRef.d2_small_top__DOT__conv1__DOT__out_col)) 
                                                - (IData)(1U));
    vlSelfRef.__VdfgRegularize_h6e95ff9d_0_4 = (((IData)(vlSelfRef.d2_small_top__DOT__conv2__DOT__engine__DOT__kr) 
                                                 + (IData)(vlSelfRef.d2_small_top__DOT__conv2__DOT__out_row)) 
                                                - (IData)(1U));
    vlSelfRef.__VdfgRegularize_h6e95ff9d_0_5 = (((IData)(vlSelfRef.d2_small_top__DOT__conv2__DOT__engine__DOT__kc) 
                                                 + (IData)(vlSelfRef.d2_small_top__DOT__conv2__DOT__out_col)) 
                                                - (IData)(1U));
    vlSelfRef.__VdfgRegularize_h6e95ff9d_0_6 = (((IData)(vlSelfRef.d2_small_top__DOT__conv3__DOT__engine__DOT__kr) 
                                                 + (IData)(vlSelfRef.d2_small_top__DOT__conv3__DOT__out_row)) 
                                                - (IData)(1U));
    vlSelfRef.__VdfgRegularize_h6e95ff9d_0_7 = (((IData)(vlSelfRef.d2_small_top__DOT__conv3__DOT__engine__DOT__kc) 
                                                 + (IData)(vlSelfRef.d2_small_top__DOT__conv3__DOT__out_col)) 
                                                - (IData)(1U));
    d2_small_top__DOT__conv1__DOT__quant__DOT__gen_round_shift__DOT__shifted_magnitude 
        = (0x000007ffffffffffULL & VL_SHIFTRS_QQI(43,43,32, 
                                                  (0x000007ffffffffffULL 
                                                   & ((1U 
                                                       & (IData)(
                                                                 (vlSelfRef.d2_small_top__DOT__conv1__DOT__product_reg 
                                                                  >> 0x0000002aU)))
                                                       ? 
                                                      (0x0000000000800000ULL 
                                                       + 
                                                       (- vlSelfRef.d2_small_top__DOT__conv1__DOT__product_reg))
                                                       : 
                                                      (0x0000000000800000ULL 
                                                       + vlSelfRef.d2_small_top__DOT__conv1__DOT__product_reg))), 0x00000018U));
    d2_small_top__DOT__conv2__DOT__quant__DOT__gen_round_shift__DOT__shifted_magnitude 
        = (0x000007ffffffffffULL & VL_SHIFTRS_QQI(43,43,32, 
                                                  (0x000007ffffffffffULL 
                                                   & ((1U 
                                                       & (IData)(
                                                                 (vlSelfRef.d2_small_top__DOT__conv2__DOT__product_reg 
                                                                  >> 0x0000002aU)))
                                                       ? 
                                                      (0x0000000000800000ULL 
                                                       + 
                                                       (- vlSelfRef.d2_small_top__DOT__conv2__DOT__product_reg))
                                                       : 
                                                      (0x0000000000800000ULL 
                                                       + vlSelfRef.d2_small_top__DOT__conv2__DOT__product_reg))), 0x00000018U));
    d2_small_top__DOT__conv3__DOT__quant__DOT__gen_round_shift__DOT__shifted_magnitude 
        = (0x000007ffffffffffULL & VL_SHIFTRS_QQI(43,43,32, 
                                                  (0x000007ffffffffffULL 
                                                   & ((1U 
                                                       & (IData)(
                                                                 (vlSelfRef.d2_small_top__DOT__conv3__DOT__product_reg 
                                                                  >> 0x0000002aU)))
                                                       ? 
                                                      (0x0000000000800000ULL 
                                                       + 
                                                       (- vlSelfRef.d2_small_top__DOT__conv3__DOT__product_reg))
                                                       : 
                                                      (0x0000000000800000ULL 
                                                       + vlSelfRef.d2_small_top__DOT__conv3__DOT__product_reg))), 0x00000018U));
    vlSelfRef.class_id = 0U;
    vlSelfRef.max_logit = vlSelfRef.d2_small_top__DOT__logit0;
    if (VL_GTS_III(32, vlSelfRef.d2_small_top__DOT__logit1, vlSelfRef.max_logit)) {
        vlSelfRef.max_logit = vlSelfRef.d2_small_top__DOT__logit1;
        vlSelfRef.class_id = 1U;
    }
    if (VL_GTS_III(32, vlSelfRef.d2_small_top__DOT__logit2, vlSelfRef.max_logit)) {
        vlSelfRef.max_logit = vlSelfRef.d2_small_top__DOT__logit2;
        vlSelfRef.class_id = 2U;
    }
    if (VL_GTS_III(32, vlSelfRef.d2_small_top__DOT__logit3, vlSelfRef.max_logit)) {
        vlSelfRef.max_logit = vlSelfRef.d2_small_top__DOT__logit3;
        vlSelfRef.class_id = 3U;
    }
    if (VL_GTS_III(32, vlSelfRef.d2_small_top__DOT__logit4, vlSelfRef.max_logit)) {
        vlSelfRef.max_logit = vlSelfRef.d2_small_top__DOT__logit4;
        vlSelfRef.class_id = 4U;
    }
    if (VL_GTS_III(32, vlSelfRef.d2_small_top__DOT__logit5, vlSelfRef.max_logit)) {
        vlSelfRef.max_logit = vlSelfRef.d2_small_top__DOT__logit5;
        vlSelfRef.class_id = 5U;
    }
    if (VL_GTS_III(32, vlSelfRef.d2_small_top__DOT__logit6, vlSelfRef.max_logit)) {
        vlSelfRef.max_logit = vlSelfRef.d2_small_top__DOT__logit6;
        vlSelfRef.class_id = 6U;
    }
    if (VL_GTS_III(32, vlSelfRef.d2_small_top__DOT__logit7, vlSelfRef.max_logit)) {
        vlSelfRef.max_logit = vlSelfRef.d2_small_top__DOT__logit7;
        vlSelfRef.class_id = 7U;
    }
    if (VL_GTS_III(32, vlSelfRef.d2_small_top__DOT__logit8, vlSelfRef.max_logit)) {
        vlSelfRef.max_logit = vlSelfRef.d2_small_top__DOT__logit8;
        vlSelfRef.class_id = 8U;
    }
    if (VL_GTS_III(32, vlSelfRef.d2_small_top__DOT__logit9, vlSelfRef.max_logit)) {
        vlSelfRef.max_logit = vlSelfRef.d2_small_top__DOT__logit9;
        vlSelfRef.class_id = 9U;
    }
    vlSelfRef.__VdfgRegularize_h6e95ff9d_0_0 = ((~ (IData)(vlSelfRef.busy)) 
                                                & ((~ 
                                                    ((IData)(vlSelfRef.d2_small_top__DOT__conv1__DOT__postprocess_pending) 
                                                     | ((IData)(vlSelfRef.d2_small_top__DOT__conv1__DOT__product_valid) 
                                                        | ((IData)(vlSelfRef.d2_small_top__DOT__conv1__DOT__postprocess_valid) 
                                                           | ((IData)(vlSelfRef.d2_small_top__DOT__conv2__DOT__postprocess_pending) 
                                                              | ((IData)(vlSelfRef.d2_small_top__DOT__conv2__DOT__product_valid) 
                                                                 | ((IData)(vlSelfRef.d2_small_top__DOT__conv2__DOT__postprocess_valid) 
                                                                    | ((IData)(vlSelfRef.d2_small_top__DOT__conv3__DOT__postprocess_pending) 
                                                                       | ((IData)(vlSelfRef.d2_small_top__DOT__conv3__DOT__product_valid) 
                                                                          | ((IData)(vlSelfRef.d2_small_top__DOT__conv3__DOT__postprocess_valid) 
                                                                             | ((IData)(vlSelfRef.d2_small_top__DOT__conv1__DOT__controller_busy) 
                                                                                | ((IData)(vlSelfRef.d2_small_top__DOT__conv2__DOT__controller_busy) 
                                                                                | ((IData)(vlSelfRef.d2_small_top__DOT__conv3__DOT__controller_busy) 
                                                                                | ((IData)(vlSelfRef.d2_small_top__DOT__pool1_busy) 
                                                                                | ((IData)(vlSelfRef.d2_small_top__DOT__pool2_busy) 
                                                                                | ((IData)(vlSelfRef.d2_small_top__DOT__gap_busy) 
                                                                                | (IData)(vlSelfRef.d2_small_top__DOT__linear_busy))))))))))))))))) 
                                                   & (IData)(vlSelfRef.input_we)));
    vlSelfRef.d2_small_top__DOT__conv1__DOT__engine__DOT__valid_pixel 
        = (VL_LTES_III(32, 0U, vlSelfRef.__VdfgRegularize_h6e95ff9d_0_2) 
           & (VL_GTS_III(32, 0x00000020U, vlSelfRef.__VdfgRegularize_h6e95ff9d_0_2) 
              & (VL_LTES_III(32, 0U, vlSelfRef.__VdfgRegularize_h6e95ff9d_0_3) 
                 & VL_GTS_III(32, 0x00000020U, vlSelfRef.__VdfgRegularize_h6e95ff9d_0_3))));
    vlSelfRef.d2_small_top__DOT__conv2__DOT__engine__DOT__valid_pixel 
        = (VL_LTES_III(32, 0U, vlSelfRef.__VdfgRegularize_h6e95ff9d_0_4) 
           & (VL_GTS_III(32, 0x00000010U, vlSelfRef.__VdfgRegularize_h6e95ff9d_0_4) 
              & (VL_LTES_III(32, 0U, vlSelfRef.__VdfgRegularize_h6e95ff9d_0_5) 
                 & VL_GTS_III(32, 0x00000010U, vlSelfRef.__VdfgRegularize_h6e95ff9d_0_5))));
    vlSelfRef.d2_small_top__DOT__conv3__DOT__engine__DOT__valid_pixel 
        = (VL_LTES_III(32, 0U, vlSelfRef.__VdfgRegularize_h6e95ff9d_0_6) 
           & (VL_GTS_III(32, 8U, vlSelfRef.__VdfgRegularize_h6e95ff9d_0_6) 
              & (VL_LTES_III(32, 0U, vlSelfRef.__VdfgRegularize_h6e95ff9d_0_7) 
                 & VL_GTS_III(32, 8U, vlSelfRef.__VdfgRegularize_h6e95ff9d_0_7))));
    d2_small_top__DOT__conv1__DOT__quant__DOT__result43 
        = (0x000007ffffffffffULL & ((1U & (IData)((vlSelfRef.d2_small_top__DOT__conv1__DOT__product_reg 
                                                   >> 0x0000002aU)))
                                     ? (- d2_small_top__DOT__conv1__DOT__quant__DOT__gen_round_shift__DOT__shifted_magnitude)
                                     : d2_small_top__DOT__conv1__DOT__quant__DOT__gen_round_shift__DOT__shifted_magnitude));
    d2_small_top__DOT__conv2__DOT__quant__DOT__result43 
        = (0x000007ffffffffffULL & ((1U & (IData)((vlSelfRef.d2_small_top__DOT__conv2__DOT__product_reg 
                                                   >> 0x0000002aU)))
                                     ? (- d2_small_top__DOT__conv2__DOT__quant__DOT__gen_round_shift__DOT__shifted_magnitude)
                                     : d2_small_top__DOT__conv2__DOT__quant__DOT__gen_round_shift__DOT__shifted_magnitude));
    d2_small_top__DOT__conv3__DOT__quant__DOT__result43 
        = (0x000007ffffffffffULL & ((1U & (IData)((vlSelfRef.d2_small_top__DOT__conv3__DOT__product_reg 
                                                   >> 0x0000002aU)))
                                     ? (- d2_small_top__DOT__conv3__DOT__quant__DOT__gen_round_shift__DOT__shifted_magnitude)
                                     : d2_small_top__DOT__conv3__DOT__quant__DOT__gen_round_shift__DOT__shifted_magnitude));
    vlSelfRef.d2_small_top__DOT__conv1__DOT__activated 
        = ((IData)(d2_small_top__DOT__conv1__DOT__quant__DOT__result43) 
           & (- (IData)((1U & (~ (IData)((d2_small_top__DOT__conv1__DOT__quant__DOT__result43 
                                          >> 0x0000001fU)))))));
    vlSelfRef.d2_small_top__DOT__conv2__DOT__activated 
        = ((IData)(d2_small_top__DOT__conv2__DOT__quant__DOT__result43) 
           & (- (IData)((1U & (~ (IData)((d2_small_top__DOT__conv2__DOT__quant__DOT__result43 
                                          >> 0x0000001fU)))))));
    vlSelfRef.d2_small_top__DOT__conv3__DOT__activated 
        = ((IData)(d2_small_top__DOT__conv3__DOT__quant__DOT__result43) 
           & (- (IData)((1U & (~ (IData)((d2_small_top__DOT__conv3__DOT__quant__DOT__result43 
                                          >> 0x0000001fU)))))));
}

VL_ATTR_COLD void Vd2_small_top___024root___eval_stl(Vd2_small_top___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vd2_small_top___024root___eval_stl\n"); );
    Vd2_small_top__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    if ((1ULL & vlSelfRef.__VstlTriggered[0U])) {
        Vd2_small_top___024root___stl_sequent__TOP__0(vlSelf);
    }
}

VL_ATTR_COLD bool Vd2_small_top___024root___eval_phase__stl(Vd2_small_top___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vd2_small_top___024root___eval_phase__stl\n"); );
    Vd2_small_top__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Locals
    CData/*0:0*/ __VstlExecute;
    // Body
    Vd2_small_top___024root___eval_triggers_vec__stl(vlSelf);
#ifdef VL_DEBUG
    if (VL_UNLIKELY(vlSymsp->_vm_contextp__->debug())) {
        Vd2_small_top___024root___dump_triggers__stl(vlSelfRef.__VstlTriggered, "stl"s);
    }
#endif
    __VstlExecute = Vd2_small_top___024root___trigger_anySet__stl(vlSelfRef.__VstlTriggered);
    if (__VstlExecute) {
        Vd2_small_top___024root___eval_stl(vlSelf);
    }
    return (__VstlExecute);
}

bool Vd2_small_top___024root___trigger_anySet__ico(const VlUnpacked<QData/*63:0*/, 1> &in);

#ifdef VL_DEBUG
VL_ATTR_COLD void Vd2_small_top___024root___dump_triggers__ico(const VlUnpacked<QData/*63:0*/, 1> &triggers, const std::string &tag) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vd2_small_top___024root___dump_triggers__ico\n"); );
    // Body
    if ((1U & (~ (IData)(Vd2_small_top___024root___trigger_anySet__ico(triggers))))) {
        VL_DBG_MSGS("         No '" + tag + "' region triggers active\n");
    }
    if ((1U & (IData)(triggers[0U]))) {
        VL_DBG_MSGS("         '" + tag + "' region trigger index 0 is active: Internal 'ico' trigger - first iteration\n");
    }
}
#endif  // VL_DEBUG

bool Vd2_small_top___024root___trigger_anySet__act(const VlUnpacked<QData/*63:0*/, 1> &in);

#ifdef VL_DEBUG
VL_ATTR_COLD void Vd2_small_top___024root___dump_triggers__act(const VlUnpacked<QData/*63:0*/, 1> &triggers, const std::string &tag) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vd2_small_top___024root___dump_triggers__act\n"); );
    // Body
    if ((1U & (~ (IData)(Vd2_small_top___024root___trigger_anySet__act(triggers))))) {
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

VL_ATTR_COLD void Vd2_small_top___024root___ctor_var_reset(Vd2_small_top___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vd2_small_top___024root___ctor_var_reset\n"); );
    Vd2_small_top__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    const uint64_t __VscopeHash = VL_MURMUR64_HASH(vlSelf->vlNamep);
    vlSelf->clk = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 16707436170211756652ull);
    vlSelf->rst_n = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 1638864771569018232ull);
    vlSelf->start = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 9867861323841650631ull);
    vlSelf->input_we = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 7025069179568517235ull);
    vlSelf->input_addr = VL_SCOPED_RAND_RESET_I(17, __VscopeHash, 13892080179392794878ull);
    vlSelf->input_data = VL_SCOPED_RAND_RESET_I(8, __VscopeHash, 1923588759227995539ull);
    vlSelf->conv1_start = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 11235579007043326849ull);
    vlSelf->pool1_start = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 15317023131219154678ull);
    vlSelf->conv2_start = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 8056422041737375597ull);
    vlSelf->pool2_start = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 16555163847064179918ull);
    vlSelf->conv3_start = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 12229945295359645444ull);
    vlSelf->gap_start = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 5165500772251323459ull);
    vlSelf->linear_start = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 4258263890256102508ull);
    vlSelf->argmax_valid = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 17584079252057304459ull);
    vlSelf->busy = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 6386567572483775230ull);
    vlSelf->done = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 10296494685231209730ull);
    vlSelf->class_id = VL_SCOPED_RAND_RESET_I(4, __VscopeHash, 137756330502096589ull);
    vlSelf->max_logit = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 12814312630002674517ull);
    vlSelf->state_dbg = VL_SCOPED_RAND_RESET_I(4, __VscopeHash, 6042213990076480022ull);
    vlSelf->d2_small_top__DOT__pool1_done = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 12199984471277978958ull);
    vlSelf->d2_small_top__DOT__pool2_done = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 15926217569184981404ull);
    vlSelf->d2_small_top__DOT__gap_done = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 2842389905616498807ull);
    vlSelf->d2_small_top__DOT__linear_done = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 4204898241449153475ull);
    vlSelf->d2_small_top__DOT__pool1_busy = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 8441522476203464884ull);
    vlSelf->d2_small_top__DOT__pool2_busy = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 13479559086385471836ull);
    vlSelf->d2_small_top__DOT__gap_busy = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 3622130809746380070ull);
    vlSelf->d2_small_top__DOT__linear_busy = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 4793377036620573348ull);
    vlSelf->d2_small_top__DOT__buf_a_rdata = VL_SCOPED_RAND_RESET_I(8, __VscopeHash, 12227773437831020867ull);
    vlSelf->d2_small_top__DOT__buf_b_rdata = VL_SCOPED_RAND_RESET_I(8, __VscopeHash, 17234357846218179829ull);
    vlSelf->d2_small_top__DOT__pool1_out_we = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 9045092881117273570ull);
    vlSelf->d2_small_top__DOT__pool2_out_we = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 15508159971508594086ull);
    vlSelf->d2_small_top__DOT__pool1_out_addr = VL_SCOPED_RAND_RESET_I(17, __VscopeHash, 9426998326107401897ull);
    vlSelf->d2_small_top__DOT__pool2_out_addr = VL_SCOPED_RAND_RESET_I(17, __VscopeHash, 5244202016948666021ull);
    vlSelf->d2_small_top__DOT__pool1_out_data = VL_SCOPED_RAND_RESET_I(8, __VscopeHash, 9521178656302449325ull);
    vlSelf->d2_small_top__DOT__pool2_out_data = VL_SCOPED_RAND_RESET_I(8, __VscopeHash, 4138430925474900469ull);
    vlSelf->d2_small_top__DOT__gap_out_we = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 1362928884291932529ull);
    vlSelf->d2_small_top__DOT__gap_out_addr = VL_SCOPED_RAND_RESET_I(17, __VscopeHash, 18239609371206889056ull);
    vlSelf->d2_small_top__DOT__gap_out_data = VL_SCOPED_RAND_RESET_I(8, __VscopeHash, 6037608595235344745ull);
    vlSelf->d2_small_top__DOT__linear_logit_index = VL_SCOPED_RAND_RESET_I(4, __VscopeHash, 7484552950109034627ull);
    vlSelf->d2_small_top__DOT__linear_logit_data = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 1911119593354136604ull);
    vlSelf->d2_small_top__DOT__linear_logit_valid = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 9894356042275440975ull);
    vlSelf->d2_small_top__DOT__conv1_weight_data = VL_SCOPED_RAND_RESET_I(8, __VscopeHash, 13978419774328643891ull);
    vlSelf->d2_small_top__DOT__conv2_weight_data = VL_SCOPED_RAND_RESET_I(8, __VscopeHash, 5168165040136417181ull);
    vlSelf->d2_small_top__DOT__conv3_weight_data = VL_SCOPED_RAND_RESET_I(8, __VscopeHash, 6113630678286167222ull);
    vlSelf->d2_small_top__DOT__linear_weight_data = VL_SCOPED_RAND_RESET_I(8, __VscopeHash, 10549724837386213654ull);
    vlSelf->d2_small_top__DOT__logit0 = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 2897973552575142021ull);
    vlSelf->d2_small_top__DOT__logit1 = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 17584178280022279000ull);
    vlSelf->d2_small_top__DOT__logit2 = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 11283503766646517630ull);
    vlSelf->d2_small_top__DOT__logit3 = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 11561955556088652686ull);
    vlSelf->d2_small_top__DOT__logit4 = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 15043525278130741479ull);
    vlSelf->d2_small_top__DOT__logit5 = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 18275832351121228367ull);
    vlSelf->d2_small_top__DOT__logit6 = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 9365615815514568137ull);
    vlSelf->d2_small_top__DOT__logit7 = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 13250891529458946688ull);
    vlSelf->d2_small_top__DOT__logit8 = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 10031731456867883339ull);
    vlSelf->d2_small_top__DOT__logit9 = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 12528410705984395242ull);
    vlSelf->d2_small_top__DOT__fsm__DOT__state = VL_SCOPED_RAND_RESET_I(4, __VscopeHash, 6101791709816612088ull);
    for (int __Vi0 = 0; __Vi0 < 8192; ++__Vi0) {
        vlSelf->d2_small_top__DOT__buffer_a__DOT__mem[__Vi0] = VL_SCOPED_RAND_RESET_I(8, __VscopeHash, 13734766364088855549ull);
    }
    for (int __Vi0 = 0; __Vi0 < 16384; ++__Vi0) {
        vlSelf->d2_small_top__DOT__buffer_b__DOT__mem[__Vi0] = VL_SCOPED_RAND_RESET_I(8, __VscopeHash, 878507987495489448ull);
    }
    for (int __Vi0 = 0; __Vi0 < 432; ++__Vi0) {
        vlSelf->d2_small_top__DOT__conv1_weight_rom__DOT__rom[__Vi0] = VL_SCOPED_RAND_RESET_I(8, __VscopeHash, 3160936557364407452ull);
    }
    vlSelf->d2_small_top__DOT__conv1_weight_rom__DOT__rom_addr = VL_SCOPED_RAND_RESET_I(9, __VscopeHash, 15640386223156264414ull);
    for (int __Vi0 = 0; __Vi0 < 4608; ++__Vi0) {
        vlSelf->d2_small_top__DOT__conv2_weight_rom__DOT__rom[__Vi0] = VL_SCOPED_RAND_RESET_I(8, __VscopeHash, 7461402268413274640ull);
    }
    vlSelf->d2_small_top__DOT__conv2_weight_rom__DOT__rom_addr = VL_SCOPED_RAND_RESET_I(13, __VscopeHash, 12616246656071259186ull);
    for (int __Vi0 = 0; __Vi0 < 18432; ++__Vi0) {
        vlSelf->d2_small_top__DOT__conv3_weight_rom__DOT__rom[__Vi0] = VL_SCOPED_RAND_RESET_I(8, __VscopeHash, 5771771090502102383ull);
    }
    vlSelf->d2_small_top__DOT__conv3_weight_rom__DOT__rom_addr = VL_SCOPED_RAND_RESET_I(15, __VscopeHash, 11759036170343729779ull);
    for (int __Vi0 = 0; __Vi0 < 640; ++__Vi0) {
        vlSelf->d2_small_top__DOT__linear_weight_rom__DOT__rom[__Vi0] = VL_SCOPED_RAND_RESET_I(8, __VscopeHash, 11225062116892732508ull);
    }
    vlSelf->d2_small_top__DOT__linear_weight_rom__DOT__rom_addr = VL_SCOPED_RAND_RESET_I(10, __VscopeHash, 3730662896829060374ull);
    for (int __Vi0 = 0; __Vi0 < 16; ++__Vi0) {
        vlSelf->d2_small_top__DOT__conv1_bias_rom__DOT__rom[__Vi0] = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 5898092097430568237ull);
    }
    for (int __Vi0 = 0; __Vi0 < 32; ++__Vi0) {
        vlSelf->d2_small_top__DOT__conv2_bias_rom__DOT__rom[__Vi0] = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 240060493811855373ull);
    }
    for (int __Vi0 = 0; __Vi0 < 64; ++__Vi0) {
        vlSelf->d2_small_top__DOT__conv3_bias_rom__DOT__rom[__Vi0] = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 11457125040215041060ull);
    }
    for (int __Vi0 = 0; __Vi0 < 10; ++__Vi0) {
        vlSelf->d2_small_top__DOT__linear_bias_rom__DOT__rom[__Vi0] = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 12881258348570516432ull);
    }
    vlSelf->d2_small_top__DOT__conv1__DOT__engine_start = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 9628236738779282574ull);
    vlSelf->d2_small_top__DOT__conv1__DOT__engine_done = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 7430156159447743872ull);
    vlSelf->d2_small_top__DOT__conv1__DOT__controller_busy = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 4903944776717538970ull);
    vlSelf->d2_small_top__DOT__conv1__DOT__controller_done = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 9910694590037949368ull);
    vlSelf->d2_small_top__DOT__conv1__DOT__raw_sum = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 5248976865231051859ull);
    vlSelf->d2_small_top__DOT__conv1__DOT__biased_sum_reg = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 15956983490932731596ull);
    vlSelf->d2_small_top__DOT__conv1__DOT__product_reg = VL_SCOPED_RAND_RESET_Q(43, __VscopeHash, 5942765192238206454ull);
    vlSelf->d2_small_top__DOT__conv1__DOT__activated = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 6465898970605737077ull);
    vlSelf->d2_small_top__DOT__conv1__DOT__out_row = VL_SCOPED_RAND_RESET_I(16, __VscopeHash, 8189561980707146663ull);
    vlSelf->d2_small_top__DOT__conv1__DOT__out_col = VL_SCOPED_RAND_RESET_I(16, __VscopeHash, 14623637823386831837ull);
    vlSelf->d2_small_top__DOT__conv1__DOT__out_channel = VL_SCOPED_RAND_RESET_I(16, __VscopeHash, 6365546864417171918ull);
    vlSelf->d2_small_top__DOT__conv1__DOT__controller_out_addr = VL_SCOPED_RAND_RESET_I(17, __VscopeHash, 5007090424336035701ull);
    vlSelf->d2_small_top__DOT__conv1__DOT__postprocess_pending = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 6774284472926305146ull);
    vlSelf->d2_small_top__DOT__conv1__DOT__product_valid = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 7639537034341413797ull);
    vlSelf->d2_small_top__DOT__conv1__DOT__postprocess_valid = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 15935516533741830535ull);
    vlSelf->d2_small_top__DOT__conv1__DOT__product_addr_reg = VL_SCOPED_RAND_RESET_I(17, __VscopeHash, 10835055657187901819ull);
    vlSelf->d2_small_top__DOT__conv1__DOT__out_data_reg = VL_SCOPED_RAND_RESET_I(8, __VscopeHash, 1533313424791991934ull);
    vlSelf->d2_small_top__DOT__conv1__DOT__out_addr_reg = VL_SCOPED_RAND_RESET_I(17, __VscopeHash, 2488502594336977750ull);
    vlSelf->d2_small_top__DOT__conv1__DOT__controller_done_d1 = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 11340274594731490901ull);
    vlSelf->d2_small_top__DOT__conv1__DOT__controller_done_d2 = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 4635378031013391453ull);
    vlSelf->d2_small_top__DOT__conv1__DOT__controller__DOT__state = VL_SCOPED_RAND_RESET_I(2, __VscopeHash, 4256332670129681961ull);
    vlSelf->d2_small_top__DOT__conv1__DOT__engine__DOT__state = VL_SCOPED_RAND_RESET_I(2, __VscopeHash, 4039857429999289814ull);
    vlSelf->d2_small_top__DOT__conv1__DOT__engine__DOT__ic = VL_SCOPED_RAND_RESET_I(16, __VscopeHash, 5333561589246594952ull);
    vlSelf->d2_small_top__DOT__conv1__DOT__engine__DOT__kr = VL_SCOPED_RAND_RESET_I(2, __VscopeHash, 3881791016594159429ull);
    vlSelf->d2_small_top__DOT__conv1__DOT__engine__DOT__kc = VL_SCOPED_RAND_RESET_I(2, __VscopeHash, 5852122001331090411ull);
    vlSelf->d2_small_top__DOT__conv1__DOT__engine__DOT__valid_pixel = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 557329481925442986ull);
    vlSelf->d2_small_top__DOT__conv1__DOT__engine__DOT__product_reg = VL_SCOPED_RAND_RESET_I(16, __VscopeHash, 7216494438655908964ull);
    vlSelf->d2_small_top__DOT__conv1__DOT__engine__DOT__product_valid = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 16089215351552432034ull);
    vlSelf->d2_small_top__DOT__conv1__DOT__engine__DOT__product_last = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 15204484623439251470ull);
    vlSelf->d2_small_top__DOT__conv1__DOT__engine__DOT__last_item = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 3242258922138881208ull);
    vlSelf->d2_small_top__DOT__pool1__DOT__state = VL_SCOPED_RAND_RESET_I(3, __VscopeHash, 3058472455026268759ull);
    vlSelf->d2_small_top__DOT__pool1__DOT__ch = VL_SCOPED_RAND_RESET_I(16, __VscopeHash, 15178899382521049959ull);
    vlSelf->d2_small_top__DOT__pool1__DOT__row = VL_SCOPED_RAND_RESET_I(16, __VscopeHash, 9414392324991873203ull);
    vlSelf->d2_small_top__DOT__pool1__DOT__col = VL_SCOPED_RAND_RESET_I(16, __VscopeHash, 6149319795826115338ull);
    vlSelf->d2_small_top__DOT__pool1__DOT__tap = VL_SCOPED_RAND_RESET_I(2, __VscopeHash, 2602080925292862597ull);
    vlSelf->d2_small_top__DOT__pool1__DOT__current_max = VL_SCOPED_RAND_RESET_I(8, __VscopeHash, 7273018591166886542ull);
    vlSelf->d2_small_top__DOT__conv2__DOT__engine_start = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 7141181730879869662ull);
    vlSelf->d2_small_top__DOT__conv2__DOT__engine_done = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 2002157180843319329ull);
    vlSelf->d2_small_top__DOT__conv2__DOT__controller_busy = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 4030652246920363555ull);
    vlSelf->d2_small_top__DOT__conv2__DOT__controller_done = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 6451207222070014528ull);
    vlSelf->d2_small_top__DOT__conv2__DOT__raw_sum = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 11453039544214534330ull);
    vlSelf->d2_small_top__DOT__conv2__DOT__biased_sum_reg = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 3376491345586006777ull);
    vlSelf->d2_small_top__DOT__conv2__DOT__product_reg = VL_SCOPED_RAND_RESET_Q(43, __VscopeHash, 15520676810451074145ull);
    vlSelf->d2_small_top__DOT__conv2__DOT__activated = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 12350010938087076166ull);
    vlSelf->d2_small_top__DOT__conv2__DOT__out_row = VL_SCOPED_RAND_RESET_I(16, __VscopeHash, 15170569606622235089ull);
    vlSelf->d2_small_top__DOT__conv2__DOT__out_col = VL_SCOPED_RAND_RESET_I(16, __VscopeHash, 3089178148464392495ull);
    vlSelf->d2_small_top__DOT__conv2__DOT__out_channel = VL_SCOPED_RAND_RESET_I(16, __VscopeHash, 4150624344971497954ull);
    vlSelf->d2_small_top__DOT__conv2__DOT__controller_out_addr = VL_SCOPED_RAND_RESET_I(17, __VscopeHash, 13324114147021870186ull);
    vlSelf->d2_small_top__DOT__conv2__DOT__postprocess_pending = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 6237638990305777679ull);
    vlSelf->d2_small_top__DOT__conv2__DOT__product_valid = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 12612368882764039578ull);
    vlSelf->d2_small_top__DOT__conv2__DOT__postprocess_valid = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 2694553671735484652ull);
    vlSelf->d2_small_top__DOT__conv2__DOT__product_addr_reg = VL_SCOPED_RAND_RESET_I(17, __VscopeHash, 9360926779160102164ull);
    vlSelf->d2_small_top__DOT__conv2__DOT__out_data_reg = VL_SCOPED_RAND_RESET_I(8, __VscopeHash, 3407452164340906965ull);
    vlSelf->d2_small_top__DOT__conv2__DOT__out_addr_reg = VL_SCOPED_RAND_RESET_I(17, __VscopeHash, 15814887376898276953ull);
    vlSelf->d2_small_top__DOT__conv2__DOT__controller_done_d1 = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 10682321544974909844ull);
    vlSelf->d2_small_top__DOT__conv2__DOT__controller_done_d2 = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 4618060019381109542ull);
    vlSelf->d2_small_top__DOT__conv2__DOT__controller__DOT__state = VL_SCOPED_RAND_RESET_I(2, __VscopeHash, 11204886692442445275ull);
    vlSelf->d2_small_top__DOT__conv2__DOT__engine__DOT__state = VL_SCOPED_RAND_RESET_I(2, __VscopeHash, 14784319305571935810ull);
    vlSelf->d2_small_top__DOT__conv2__DOT__engine__DOT__ic = VL_SCOPED_RAND_RESET_I(16, __VscopeHash, 17487995883076641773ull);
    vlSelf->d2_small_top__DOT__conv2__DOT__engine__DOT__kr = VL_SCOPED_RAND_RESET_I(2, __VscopeHash, 12309692645807713687ull);
    vlSelf->d2_small_top__DOT__conv2__DOT__engine__DOT__kc = VL_SCOPED_RAND_RESET_I(2, __VscopeHash, 14117263495255307137ull);
    vlSelf->d2_small_top__DOT__conv2__DOT__engine__DOT__valid_pixel = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 10841412100517742206ull);
    vlSelf->d2_small_top__DOT__conv2__DOT__engine__DOT__product_reg = VL_SCOPED_RAND_RESET_I(16, __VscopeHash, 6316073712115585906ull);
    vlSelf->d2_small_top__DOT__conv2__DOT__engine__DOT__product_valid = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 971095955497175738ull);
    vlSelf->d2_small_top__DOT__conv2__DOT__engine__DOT__product_last = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 12514243686226735044ull);
    vlSelf->d2_small_top__DOT__conv2__DOT__engine__DOT__last_item = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 8616544732444494696ull);
    vlSelf->d2_small_top__DOT__pool2__DOT__state = VL_SCOPED_RAND_RESET_I(3, __VscopeHash, 11924353233822890808ull);
    vlSelf->d2_small_top__DOT__pool2__DOT__ch = VL_SCOPED_RAND_RESET_I(16, __VscopeHash, 9254936039066493788ull);
    vlSelf->d2_small_top__DOT__pool2__DOT__row = VL_SCOPED_RAND_RESET_I(16, __VscopeHash, 4521304554957888350ull);
    vlSelf->d2_small_top__DOT__pool2__DOT__col = VL_SCOPED_RAND_RESET_I(16, __VscopeHash, 9516283518974489837ull);
    vlSelf->d2_small_top__DOT__pool2__DOT__tap = VL_SCOPED_RAND_RESET_I(2, __VscopeHash, 14084875563508851106ull);
    vlSelf->d2_small_top__DOT__pool2__DOT__current_max = VL_SCOPED_RAND_RESET_I(8, __VscopeHash, 3354495335309657716ull);
    vlSelf->d2_small_top__DOT__conv3__DOT__engine_start = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 3348933935158667721ull);
    vlSelf->d2_small_top__DOT__conv3__DOT__engine_done = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 6954291635405080899ull);
    vlSelf->d2_small_top__DOT__conv3__DOT__controller_busy = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 8029425186331842832ull);
    vlSelf->d2_small_top__DOT__conv3__DOT__controller_done = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 15612209200415350506ull);
    vlSelf->d2_small_top__DOT__conv3__DOT__raw_sum = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 17588318022328981714ull);
    vlSelf->d2_small_top__DOT__conv3__DOT__biased_sum_reg = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 5193727426836682007ull);
    vlSelf->d2_small_top__DOT__conv3__DOT__product_reg = VL_SCOPED_RAND_RESET_Q(43, __VscopeHash, 12146795278257923956ull);
    vlSelf->d2_small_top__DOT__conv3__DOT__activated = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 10238340606963407801ull);
    vlSelf->d2_small_top__DOT__conv3__DOT__out_row = VL_SCOPED_RAND_RESET_I(16, __VscopeHash, 698249955614312964ull);
    vlSelf->d2_small_top__DOT__conv3__DOT__out_col = VL_SCOPED_RAND_RESET_I(16, __VscopeHash, 13725022362427942624ull);
    vlSelf->d2_small_top__DOT__conv3__DOT__out_channel = VL_SCOPED_RAND_RESET_I(16, __VscopeHash, 16351700748470079754ull);
    vlSelf->d2_small_top__DOT__conv3__DOT__controller_out_addr = VL_SCOPED_RAND_RESET_I(17, __VscopeHash, 7686382431749917153ull);
    vlSelf->d2_small_top__DOT__conv3__DOT__postprocess_pending = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 2966719802431018521ull);
    vlSelf->d2_small_top__DOT__conv3__DOT__product_valid = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 15215580059944877992ull);
    vlSelf->d2_small_top__DOT__conv3__DOT__postprocess_valid = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 4034620184831198030ull);
    vlSelf->d2_small_top__DOT__conv3__DOT__product_addr_reg = VL_SCOPED_RAND_RESET_I(17, __VscopeHash, 11578701843501909783ull);
    vlSelf->d2_small_top__DOT__conv3__DOT__out_data_reg = VL_SCOPED_RAND_RESET_I(8, __VscopeHash, 15451955901558131105ull);
    vlSelf->d2_small_top__DOT__conv3__DOT__out_addr_reg = VL_SCOPED_RAND_RESET_I(17, __VscopeHash, 16895263237438794887ull);
    vlSelf->d2_small_top__DOT__conv3__DOT__controller_done_d1 = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 16273646209726779710ull);
    vlSelf->d2_small_top__DOT__conv3__DOT__controller_done_d2 = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 11097650403884080752ull);
    vlSelf->d2_small_top__DOT__conv3__DOT__controller__DOT__state = VL_SCOPED_RAND_RESET_I(2, __VscopeHash, 15149025948064522538ull);
    vlSelf->d2_small_top__DOT__conv3__DOT__engine__DOT__state = VL_SCOPED_RAND_RESET_I(2, __VscopeHash, 2475540553764625894ull);
    vlSelf->d2_small_top__DOT__conv3__DOT__engine__DOT__ic = VL_SCOPED_RAND_RESET_I(16, __VscopeHash, 13484648391762821896ull);
    vlSelf->d2_small_top__DOT__conv3__DOT__engine__DOT__kr = VL_SCOPED_RAND_RESET_I(2, __VscopeHash, 7860925615487982513ull);
    vlSelf->d2_small_top__DOT__conv3__DOT__engine__DOT__kc = VL_SCOPED_RAND_RESET_I(2, __VscopeHash, 69076299290422110ull);
    vlSelf->d2_small_top__DOT__conv3__DOT__engine__DOT__valid_pixel = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 8250510072586513702ull);
    vlSelf->d2_small_top__DOT__conv3__DOT__engine__DOT__product_reg = VL_SCOPED_RAND_RESET_I(16, __VscopeHash, 858301705371713243ull);
    vlSelf->d2_small_top__DOT__conv3__DOT__engine__DOT__product_valid = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 3350063678241149218ull);
    vlSelf->d2_small_top__DOT__conv3__DOT__engine__DOT__product_last = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 14344354891499678597ull);
    vlSelf->d2_small_top__DOT__conv3__DOT__engine__DOT__last_item = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 5263293498108691345ull);
    vlSelf->d2_small_top__DOT__gap__DOT__state = VL_SCOPED_RAND_RESET_I(3, __VscopeHash, 14999129975407488185ull);
    vlSelf->d2_small_top__DOT__gap__DOT__ch = VL_SCOPED_RAND_RESET_I(16, __VscopeHash, 4796167085544250669ull);
    vlSelf->d2_small_top__DOT__gap__DOT__idx = VL_SCOPED_RAND_RESET_I(16, __VscopeHash, 5986884002109078167ull);
    vlSelf->d2_small_top__DOT__gap__DOT__sum = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 17031704850223607152ull);
    vlSelf->d2_small_top__DOT__linear__DOT__state = VL_SCOPED_RAND_RESET_I(3, __VscopeHash, 7188393510450475430ull);
    vlSelf->d2_small_top__DOT__linear__DOT__class_idx = VL_SCOPED_RAND_RESET_I(4, __VscopeHash, 4807917417213259218ull);
    vlSelf->d2_small_top__DOT__linear__DOT__feature_idx = VL_SCOPED_RAND_RESET_I(6, __VscopeHash, 11213923354244418302ull);
    vlSelf->d2_small_top__DOT__linear__DOT__acc = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 9775376143808495768ull);
    vlSelf->__VdfgRegularize_h6e95ff9d_0_0 = 0;
    vlSelf->__VdfgRegularize_h6e95ff9d_0_2 = 0;
    vlSelf->__VdfgRegularize_h6e95ff9d_0_3 = 0;
    vlSelf->__VdfgRegularize_h6e95ff9d_0_4 = 0;
    vlSelf->__VdfgRegularize_h6e95ff9d_0_5 = 0;
    vlSelf->__VdfgRegularize_h6e95ff9d_0_6 = 0;
    vlSelf->__VdfgRegularize_h6e95ff9d_0_7 = 0;
    vlSelf->__Vdly__conv1_start = 0;
    vlSelf->__Vdly__pool1_start = 0;
    vlSelf->__Vdly__conv2_start = 0;
    vlSelf->__Vdly__pool2_start = 0;
    vlSelf->__Vdly__conv3_start = 0;
    vlSelf->__Vdly__gap_start = 0;
    vlSelf->__Vdly__state_dbg = 0;
    vlSelf->__Vdly__d2_small_top__DOT__conv1__DOT__engine_start = 0;
    vlSelf->__Vdly__d2_small_top__DOT__conv1__DOT__controller__DOT__state = 0;
    vlSelf->__Vdly__d2_small_top__DOT__conv1__DOT__engine__DOT__state = 0;
    vlSelf->__Vdly__d2_small_top__DOT__conv1__DOT__engine__DOT__ic = 0;
    vlSelf->__Vdly__d2_small_top__DOT__conv1__DOT__engine__DOT__product_reg = 0;
    vlSelf->__Vdly__d2_small_top__DOT__conv1__DOT__engine__DOT__product_valid = 0;
    vlSelf->__Vdly__d2_small_top__DOT__conv1__DOT__engine__DOT__product_last = 0;
    vlSelf->__Vdly__d2_small_top__DOT__pool1__DOT__state = 0;
    vlSelf->__Vdly__d2_small_top__DOT__pool1__DOT__tap = 0;
    vlSelf->__Vdly__d2_small_top__DOT__pool1__DOT__col = 0;
    vlSelf->__Vdly__d2_small_top__DOT__pool1__DOT__row = 0;
    vlSelf->__Vdly__d2_small_top__DOT__pool1__DOT__ch = 0;
    vlSelf->__Vdly__d2_small_top__DOT__pool1__DOT__current_max = 0;
    vlSelf->__Vdly__d2_small_top__DOT__conv2__DOT__engine_start = 0;
    vlSelf->__Vdly__d2_small_top__DOT__conv2__DOT__controller__DOT__state = 0;
    vlSelf->__Vdly__d2_small_top__DOT__conv2__DOT__engine__DOT__state = 0;
    vlSelf->__Vdly__d2_small_top__DOT__conv2__DOT__engine__DOT__ic = 0;
    vlSelf->__Vdly__d2_small_top__DOT__conv2__DOT__engine__DOT__product_reg = 0;
    vlSelf->__Vdly__d2_small_top__DOT__conv2__DOT__engine__DOT__product_valid = 0;
    vlSelf->__Vdly__d2_small_top__DOT__conv2__DOT__engine__DOT__product_last = 0;
    vlSelf->__Vdly__d2_small_top__DOT__pool2__DOT__state = 0;
    vlSelf->__Vdly__d2_small_top__DOT__pool2__DOT__tap = 0;
    vlSelf->__Vdly__d2_small_top__DOT__pool2__DOT__col = 0;
    vlSelf->__Vdly__d2_small_top__DOT__pool2__DOT__row = 0;
    vlSelf->__Vdly__d2_small_top__DOT__pool2__DOT__ch = 0;
    vlSelf->__Vdly__d2_small_top__DOT__pool2__DOT__current_max = 0;
    vlSelf->__Vdly__d2_small_top__DOT__conv3__DOT__engine_start = 0;
    vlSelf->__Vdly__d2_small_top__DOT__conv3__DOT__controller__DOT__state = 0;
    vlSelf->__Vdly__d2_small_top__DOT__conv3__DOT__engine__DOT__state = 0;
    vlSelf->__Vdly__d2_small_top__DOT__conv3__DOT__engine__DOT__ic = 0;
    vlSelf->__Vdly__d2_small_top__DOT__conv3__DOT__engine__DOT__product_reg = 0;
    vlSelf->__Vdly__d2_small_top__DOT__conv3__DOT__engine__DOT__product_valid = 0;
    vlSelf->__Vdly__d2_small_top__DOT__conv3__DOT__engine__DOT__product_last = 0;
    vlSelf->__Vdly__d2_small_top__DOT__gap__DOT__state = 0;
    vlSelf->__Vdly__d2_small_top__DOT__gap__DOT__idx = 0;
    vlSelf->__Vdly__d2_small_top__DOT__gap__DOT__ch = 0;
    vlSelf->__Vdly__d2_small_top__DOT__gap__DOT__sum = 0;
    vlSelf->__Vdly__d2_small_top__DOT__linear__DOT__feature_idx = 0;
    vlSelf->__VdlyVal__d2_small_top__DOT__buffer_a__DOT__mem__v0 = 0;
    vlSelf->__VdlyDim0__d2_small_top__DOT__buffer_a__DOT__mem__v0 = 0;
    vlSelf->__VdlySet__d2_small_top__DOT__buffer_a__DOT__mem__v0 = 0;
    vlSelf->__VdlyVal__d2_small_top__DOT__buffer_b__DOT__mem__v0 = 0;
    vlSelf->__VdlyDim0__d2_small_top__DOT__buffer_b__DOT__mem__v0 = 0;
    vlSelf->__VdlySet__d2_small_top__DOT__buffer_b__DOT__mem__v0 = 0;
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
