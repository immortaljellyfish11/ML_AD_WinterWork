// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design implementation internals
// See Vd2_small_top.h for the primary calling header

#include "Vd2_small_top__pch.h"

void Vd2_small_top___024root___eval_triggers_vec__ico(Vd2_small_top___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vd2_small_top___024root___eval_triggers_vec__ico\n"); );
    Vd2_small_top__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    vlSelfRef.__VicoTriggered[0U] = ((0xfffffffffffffffeULL 
                                      & vlSelfRef.__VicoTriggered[0U]) 
                                     | (IData)((IData)(vlSelfRef.__VicoFirstIteration)));
}

bool Vd2_small_top___024root___trigger_anySet__ico(const VlUnpacked<QData/*63:0*/, 1> &in) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vd2_small_top___024root___trigger_anySet__ico\n"); );
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

void Vd2_small_top___024root___ico_sequent__TOP__0(Vd2_small_top___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vd2_small_top___024root___ico_sequent__TOP__0\n"); );
    Vd2_small_top__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
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
}

void Vd2_small_top___024root___eval_ico(Vd2_small_top___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vd2_small_top___024root___eval_ico\n"); );
    Vd2_small_top__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    if ((1ULL & vlSelfRef.__VicoTriggered[0U])) {
        vlSelfRef.__VdfgRegularize_h6e95ff9d_0_0 = 
            ((~ (IData)(vlSelfRef.busy)) & ((~ ((IData)(vlSelfRef.d2_small_top__DOT__conv1__DOT__postprocess_pending) 
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
    }
}

#ifdef VL_DEBUG
VL_ATTR_COLD void Vd2_small_top___024root___dump_triggers__ico(const VlUnpacked<QData/*63:0*/, 1> &triggers, const std::string &tag);
#endif  // VL_DEBUG

bool Vd2_small_top___024root___eval_phase__ico(Vd2_small_top___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vd2_small_top___024root___eval_phase__ico\n"); );
    Vd2_small_top__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Locals
    CData/*0:0*/ __VicoExecute;
    // Body
    Vd2_small_top___024root___eval_triggers_vec__ico(vlSelf);
#ifdef VL_DEBUG
    if (VL_UNLIKELY(vlSymsp->_vm_contextp__->debug())) {
        Vd2_small_top___024root___dump_triggers__ico(vlSelfRef.__VicoTriggered, "ico"s);
    }
#endif
    __VicoExecute = Vd2_small_top___024root___trigger_anySet__ico(vlSelfRef.__VicoTriggered);
    if (__VicoExecute) {
        Vd2_small_top___024root___eval_ico(vlSelf);
    }
    return (__VicoExecute);
}

void Vd2_small_top___024root___eval_triggers_vec__act(Vd2_small_top___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vd2_small_top___024root___eval_triggers_vec__act\n"); );
    Vd2_small_top__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    vlSelfRef.__VactTriggered[0U] = (QData)((IData)(
                                                    ((((~ (IData)(vlSelfRef.rst_n)) 
                                                       & (IData)(vlSelfRef.__Vtrigprevexpr___TOP__rst_n__0)) 
                                                      << 1U) 
                                                     | ((IData)(vlSelfRef.clk) 
                                                        & (~ (IData)(vlSelfRef.__Vtrigprevexpr___TOP__clk__0))))));
    vlSelfRef.__Vtrigprevexpr___TOP__clk__0 = vlSelfRef.clk;
    vlSelfRef.__Vtrigprevexpr___TOP__rst_n__0 = vlSelfRef.rst_n;
}

bool Vd2_small_top___024root___trigger_anySet__act(const VlUnpacked<QData/*63:0*/, 1> &in) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vd2_small_top___024root___trigger_anySet__act\n"); );
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

void Vd2_small_top___024root___nba_sequent__TOP__0(Vd2_small_top___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vd2_small_top___024root___nba_sequent__TOP__0\n"); );
    Vd2_small_top__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Locals
    IData/*31:0*/ __Vdly__d2_small_top__DOT__logit0;
    __Vdly__d2_small_top__DOT__logit0 = 0;
    CData/*0:0*/ __Vdly__linear_start;
    __Vdly__linear_start = 0;
    CData/*3:0*/ __Vdly__d2_small_top__DOT__fsm__DOT__state;
    __Vdly__d2_small_top__DOT__fsm__DOT__state = 0;
    CData/*2:0*/ __Vdly__d2_small_top__DOT__linear__DOT__state;
    __Vdly__d2_small_top__DOT__linear__DOT__state = 0;
    IData/*31:0*/ __Vdly__d2_small_top__DOT__linear__DOT__acc;
    __Vdly__d2_small_top__DOT__linear__DOT__acc = 0;
    // Body
    vlSelfRef.__Vdly__d2_small_top__DOT__gap__DOT__state 
        = vlSelfRef.d2_small_top__DOT__gap__DOT__state;
    vlSelfRef.__Vdly__d2_small_top__DOT__gap__DOT__idx 
        = vlSelfRef.d2_small_top__DOT__gap__DOT__idx;
    vlSelfRef.__Vdly__d2_small_top__DOT__gap__DOT__ch 
        = vlSelfRef.d2_small_top__DOT__gap__DOT__ch;
    vlSelfRef.__Vdly__d2_small_top__DOT__gap__DOT__sum 
        = vlSelfRef.d2_small_top__DOT__gap__DOT__sum;
    vlSelfRef.__Vdly__d2_small_top__DOT__pool1__DOT__state 
        = vlSelfRef.d2_small_top__DOT__pool1__DOT__state;
    vlSelfRef.__Vdly__d2_small_top__DOT__pool1__DOT__tap 
        = vlSelfRef.d2_small_top__DOT__pool1__DOT__tap;
    vlSelfRef.__Vdly__d2_small_top__DOT__pool1__DOT__col 
        = vlSelfRef.d2_small_top__DOT__pool1__DOT__col;
    vlSelfRef.__Vdly__d2_small_top__DOT__pool1__DOT__row 
        = vlSelfRef.d2_small_top__DOT__pool1__DOT__row;
    vlSelfRef.__Vdly__d2_small_top__DOT__pool1__DOT__ch 
        = vlSelfRef.d2_small_top__DOT__pool1__DOT__ch;
    vlSelfRef.__Vdly__d2_small_top__DOT__pool1__DOT__current_max 
        = vlSelfRef.d2_small_top__DOT__pool1__DOT__current_max;
    vlSelfRef.__Vdly__d2_small_top__DOT__pool2__DOT__state 
        = vlSelfRef.d2_small_top__DOT__pool2__DOT__state;
    vlSelfRef.__Vdly__d2_small_top__DOT__pool2__DOT__tap 
        = vlSelfRef.d2_small_top__DOT__pool2__DOT__tap;
    vlSelfRef.__Vdly__d2_small_top__DOT__pool2__DOT__col 
        = vlSelfRef.d2_small_top__DOT__pool2__DOT__col;
    vlSelfRef.__Vdly__d2_small_top__DOT__pool2__DOT__row 
        = vlSelfRef.d2_small_top__DOT__pool2__DOT__row;
    vlSelfRef.__Vdly__d2_small_top__DOT__pool2__DOT__ch 
        = vlSelfRef.d2_small_top__DOT__pool2__DOT__ch;
    vlSelfRef.__Vdly__d2_small_top__DOT__pool2__DOT__current_max 
        = vlSelfRef.d2_small_top__DOT__pool2__DOT__current_max;
    __Vdly__d2_small_top__DOT__linear__DOT__state = vlSelfRef.d2_small_top__DOT__linear__DOT__state;
    __Vdly__d2_small_top__DOT__linear__DOT__acc = vlSelfRef.d2_small_top__DOT__linear__DOT__acc;
    vlSelfRef.__Vdly__d2_small_top__DOT__linear__DOT__feature_idx 
        = vlSelfRef.d2_small_top__DOT__linear__DOT__feature_idx;
    vlSelfRef.__Vdly__d2_small_top__DOT__conv1__DOT__engine__DOT__state 
        = vlSelfRef.d2_small_top__DOT__conv1__DOT__engine__DOT__state;
    vlSelfRef.__Vdly__d2_small_top__DOT__conv1__DOT__engine__DOT__product_reg 
        = vlSelfRef.d2_small_top__DOT__conv1__DOT__engine__DOT__product_reg;
    vlSelfRef.__Vdly__d2_small_top__DOT__conv1__DOT__engine__DOT__product_valid 
        = vlSelfRef.d2_small_top__DOT__conv1__DOT__engine__DOT__product_valid;
    vlSelfRef.__Vdly__d2_small_top__DOT__conv1__DOT__engine__DOT__product_last 
        = vlSelfRef.d2_small_top__DOT__conv1__DOT__engine__DOT__product_last;
    vlSelfRef.__Vdly__d2_small_top__DOT__conv2__DOT__engine__DOT__state 
        = vlSelfRef.d2_small_top__DOT__conv2__DOT__engine__DOT__state;
    vlSelfRef.__Vdly__d2_small_top__DOT__conv2__DOT__engine__DOT__product_reg 
        = vlSelfRef.d2_small_top__DOT__conv2__DOT__engine__DOT__product_reg;
    vlSelfRef.__Vdly__d2_small_top__DOT__conv2__DOT__engine__DOT__product_valid 
        = vlSelfRef.d2_small_top__DOT__conv2__DOT__engine__DOT__product_valid;
    vlSelfRef.__Vdly__d2_small_top__DOT__conv2__DOT__engine__DOT__product_last 
        = vlSelfRef.d2_small_top__DOT__conv2__DOT__engine__DOT__product_last;
    vlSelfRef.__Vdly__d2_small_top__DOT__conv3__DOT__engine__DOT__state 
        = vlSelfRef.d2_small_top__DOT__conv3__DOT__engine__DOT__state;
    vlSelfRef.__Vdly__d2_small_top__DOT__conv3__DOT__engine__DOT__product_reg 
        = vlSelfRef.d2_small_top__DOT__conv3__DOT__engine__DOT__product_reg;
    vlSelfRef.__Vdly__d2_small_top__DOT__conv3__DOT__engine__DOT__product_valid 
        = vlSelfRef.d2_small_top__DOT__conv3__DOT__engine__DOT__product_valid;
    vlSelfRef.__Vdly__d2_small_top__DOT__conv3__DOT__engine__DOT__product_last 
        = vlSelfRef.d2_small_top__DOT__conv3__DOT__engine__DOT__product_last;
    vlSelfRef.__Vdly__d2_small_top__DOT__conv1__DOT__engine__DOT__ic 
        = vlSelfRef.d2_small_top__DOT__conv1__DOT__engine__DOT__ic;
    vlSelfRef.__Vdly__d2_small_top__DOT__conv2__DOT__engine__DOT__ic 
        = vlSelfRef.d2_small_top__DOT__conv2__DOT__engine__DOT__ic;
    vlSelfRef.__Vdly__d2_small_top__DOT__conv3__DOT__engine__DOT__ic 
        = vlSelfRef.d2_small_top__DOT__conv3__DOT__engine__DOT__ic;
    vlSelfRef.__Vdly__d2_small_top__DOT__conv1__DOT__engine_start 
        = vlSelfRef.d2_small_top__DOT__conv1__DOT__engine_start;
    vlSelfRef.__Vdly__d2_small_top__DOT__conv1__DOT__controller__DOT__state 
        = vlSelfRef.d2_small_top__DOT__conv1__DOT__controller__DOT__state;
    vlSelfRef.__Vdly__d2_small_top__DOT__conv2__DOT__engine_start 
        = vlSelfRef.d2_small_top__DOT__conv2__DOT__engine_start;
    vlSelfRef.__Vdly__d2_small_top__DOT__conv2__DOT__controller__DOT__state 
        = vlSelfRef.d2_small_top__DOT__conv2__DOT__controller__DOT__state;
    vlSelfRef.__Vdly__d2_small_top__DOT__conv3__DOT__engine_start 
        = vlSelfRef.d2_small_top__DOT__conv3__DOT__engine_start;
    vlSelfRef.__Vdly__d2_small_top__DOT__conv3__DOT__controller__DOT__state 
        = vlSelfRef.d2_small_top__DOT__conv3__DOT__controller__DOT__state;
    __Vdly__d2_small_top__DOT__logit0 = vlSelfRef.d2_small_top__DOT__logit0;
    vlSelfRef.__Vdly__conv1_start = vlSelfRef.conv1_start;
    vlSelfRef.__Vdly__pool1_start = vlSelfRef.pool1_start;
    vlSelfRef.__Vdly__conv2_start = vlSelfRef.conv2_start;
    vlSelfRef.__Vdly__pool2_start = vlSelfRef.pool2_start;
    vlSelfRef.__Vdly__conv3_start = vlSelfRef.conv3_start;
    vlSelfRef.__Vdly__gap_start = vlSelfRef.gap_start;
    __Vdly__linear_start = vlSelfRef.linear_start;
    vlSelfRef.__Vdly__state_dbg = vlSelfRef.state_dbg;
    __Vdly__d2_small_top__DOT__fsm__DOT__state = vlSelfRef.d2_small_top__DOT__fsm__DOT__state;
    if (vlSelfRef.rst_n) {
        if (vlSelfRef.d2_small_top__DOT__linear_logit_valid) {
            if ((8U & (IData)(vlSelfRef.d2_small_top__DOT__linear_logit_index))) {
                if ((4U & (IData)(vlSelfRef.d2_small_top__DOT__linear_logit_index))) {
                    __Vdly__d2_small_top__DOT__logit0 
                        = vlSelfRef.d2_small_top__DOT__logit0;
                } else if ((2U & (IData)(vlSelfRef.d2_small_top__DOT__linear_logit_index))) {
                    __Vdly__d2_small_top__DOT__logit0 
                        = vlSelfRef.d2_small_top__DOT__logit0;
                }
                if ((1U & (~ ((IData)(vlSelfRef.d2_small_top__DOT__linear_logit_index) 
                              >> 2U)))) {
                    if ((1U & (~ ((IData)(vlSelfRef.d2_small_top__DOT__linear_logit_index) 
                                  >> 1U)))) {
                        if ((1U & (~ (IData)(vlSelfRef.d2_small_top__DOT__linear_logit_index)))) {
                            vlSelfRef.d2_small_top__DOT__logit8 
                                = vlSelfRef.d2_small_top__DOT__linear_logit_data;
                        }
                        if ((1U & (IData)(vlSelfRef.d2_small_top__DOT__linear_logit_index))) {
                            vlSelfRef.d2_small_top__DOT__logit9 
                                = vlSelfRef.d2_small_top__DOT__linear_logit_data;
                        }
                    }
                }
            } else if ((1U & (~ ((IData)(vlSelfRef.d2_small_top__DOT__linear_logit_index) 
                                 >> 2U)))) {
                if ((1U & (~ ((IData)(vlSelfRef.d2_small_top__DOT__linear_logit_index) 
                              >> 1U)))) {
                    if ((1U & (~ (IData)(vlSelfRef.d2_small_top__DOT__linear_logit_index)))) {
                        __Vdly__d2_small_top__DOT__logit0 
                            = vlSelfRef.d2_small_top__DOT__linear_logit_data;
                    }
                }
            }
            if ((1U & (~ ((IData)(vlSelfRef.d2_small_top__DOT__linear_logit_index) 
                          >> 3U)))) {
                if ((1U & (~ ((IData)(vlSelfRef.d2_small_top__DOT__linear_logit_index) 
                              >> 2U)))) {
                    if ((2U & (IData)(vlSelfRef.d2_small_top__DOT__linear_logit_index))) {
                        if ((1U & (~ (IData)(vlSelfRef.d2_small_top__DOT__linear_logit_index)))) {
                            vlSelfRef.d2_small_top__DOT__logit2 
                                = vlSelfRef.d2_small_top__DOT__linear_logit_data;
                        }
                        if ((1U & (IData)(vlSelfRef.d2_small_top__DOT__linear_logit_index))) {
                            vlSelfRef.d2_small_top__DOT__logit3 
                                = vlSelfRef.d2_small_top__DOT__linear_logit_data;
                        }
                    }
                    if ((1U & (~ ((IData)(vlSelfRef.d2_small_top__DOT__linear_logit_index) 
                                  >> 1U)))) {
                        if ((1U & (IData)(vlSelfRef.d2_small_top__DOT__linear_logit_index))) {
                            vlSelfRef.d2_small_top__DOT__logit1 
                                = vlSelfRef.d2_small_top__DOT__linear_logit_data;
                        }
                    }
                }
                if ((4U & (IData)(vlSelfRef.d2_small_top__DOT__linear_logit_index))) {
                    if ((2U & (IData)(vlSelfRef.d2_small_top__DOT__linear_logit_index))) {
                        if ((1U & (IData)(vlSelfRef.d2_small_top__DOT__linear_logit_index))) {
                            vlSelfRef.d2_small_top__DOT__logit7 
                                = vlSelfRef.d2_small_top__DOT__linear_logit_data;
                        }
                        if ((1U & (~ (IData)(vlSelfRef.d2_small_top__DOT__linear_logit_index)))) {
                            vlSelfRef.d2_small_top__DOT__logit6 
                                = vlSelfRef.d2_small_top__DOT__linear_logit_data;
                        }
                    }
                    if ((1U & (~ ((IData)(vlSelfRef.d2_small_top__DOT__linear_logit_index) 
                                  >> 1U)))) {
                        if ((1U & (~ (IData)(vlSelfRef.d2_small_top__DOT__linear_logit_index)))) {
                            vlSelfRef.d2_small_top__DOT__logit4 
                                = vlSelfRef.d2_small_top__DOT__linear_logit_data;
                        }
                        if ((1U & (IData)(vlSelfRef.d2_small_top__DOT__linear_logit_index))) {
                            vlSelfRef.d2_small_top__DOT__logit5 
                                = vlSelfRef.d2_small_top__DOT__linear_logit_data;
                        }
                    }
                }
            }
        }
        vlSelfRef.__Vdly__conv1_start = 0U;
        vlSelfRef.__Vdly__pool1_start = 0U;
        vlSelfRef.__Vdly__conv2_start = 0U;
        vlSelfRef.__Vdly__pool2_start = 0U;
        vlSelfRef.__Vdly__conv3_start = 0U;
        vlSelfRef.__Vdly__gap_start = 0U;
        __Vdly__linear_start = 0U;
        vlSelfRef.argmax_valid = 0U;
        vlSelfRef.done = 0U;
        vlSelfRef.__Vdly__state_dbg = vlSelfRef.d2_small_top__DOT__fsm__DOT__state;
        if ((8U & (IData)(vlSelfRef.d2_small_top__DOT__fsm__DOT__state))) {
            if ((4U & (IData)(vlSelfRef.d2_small_top__DOT__fsm__DOT__state))) {
                __Vdly__d2_small_top__DOT__fsm__DOT__state = 0U;
            } else if ((2U & (IData)(vlSelfRef.d2_small_top__DOT__fsm__DOT__state))) {
                __Vdly__d2_small_top__DOT__fsm__DOT__state = 0U;
            } else if ((1U & (IData)(vlSelfRef.d2_small_top__DOT__fsm__DOT__state))) {
                vlSelfRef.busy = 0U;
                vlSelfRef.done = 1U;
                __Vdly__d2_small_top__DOT__fsm__DOT__state = 0U;
            } else {
                __Vdly__d2_small_top__DOT__fsm__DOT__state = 9U;
            }
        } else if ((4U & (IData)(vlSelfRef.d2_small_top__DOT__fsm__DOT__state))) {
            if ((2U & (IData)(vlSelfRef.d2_small_top__DOT__fsm__DOT__state))) {
                if ((1U & (IData)(vlSelfRef.d2_small_top__DOT__fsm__DOT__state))) {
                    if (vlSelfRef.d2_small_top__DOT__linear_done) {
                        vlSelfRef.argmax_valid = 1U;
                        __Vdly__d2_small_top__DOT__fsm__DOT__state = 8U;
                    }
                } else if (vlSelfRef.d2_small_top__DOT__gap_done) {
                    __Vdly__linear_start = 1U;
                    __Vdly__d2_small_top__DOT__fsm__DOT__state = 7U;
                }
            } else if ((1U & (IData)(vlSelfRef.d2_small_top__DOT__fsm__DOT__state))) {
                if (vlSelfRef.d2_small_top__DOT__conv3__DOT__controller_done_d2) {
                    vlSelfRef.__Vdly__gap_start = 1U;
                    __Vdly__d2_small_top__DOT__fsm__DOT__state = 6U;
                }
            } else if (vlSelfRef.d2_small_top__DOT__pool2_done) {
                vlSelfRef.__Vdly__conv3_start = 1U;
                __Vdly__d2_small_top__DOT__fsm__DOT__state = 5U;
            }
        } else if ((2U & (IData)(vlSelfRef.d2_small_top__DOT__fsm__DOT__state))) {
            if ((1U & (IData)(vlSelfRef.d2_small_top__DOT__fsm__DOT__state))) {
                if (vlSelfRef.d2_small_top__DOT__conv2__DOT__controller_done_d2) {
                    vlSelfRef.__Vdly__pool2_start = 1U;
                    __Vdly__d2_small_top__DOT__fsm__DOT__state = 4U;
                }
            } else if (vlSelfRef.d2_small_top__DOT__pool1_done) {
                vlSelfRef.__Vdly__conv2_start = 1U;
                __Vdly__d2_small_top__DOT__fsm__DOT__state = 3U;
            }
        } else if ((1U & (IData)(vlSelfRef.d2_small_top__DOT__fsm__DOT__state))) {
            if (vlSelfRef.d2_small_top__DOT__conv1__DOT__controller_done_d2) {
                vlSelfRef.__Vdly__pool1_start = 1U;
                __Vdly__d2_small_top__DOT__fsm__DOT__state = 2U;
            }
        } else {
            vlSelfRef.busy = 0U;
            if (vlSelfRef.start) {
                vlSelfRef.busy = 1U;
                vlSelfRef.__Vdly__conv1_start = 1U;
                __Vdly__d2_small_top__DOT__fsm__DOT__state = 1U;
            }
        }
        vlSelfRef.d2_small_top__DOT__linear_logit_valid = 0U;
        vlSelfRef.d2_small_top__DOT__linear_done = 0U;
        if ((4U & (IData)(vlSelfRef.d2_small_top__DOT__linear__DOT__state))) {
            if ((2U & (IData)(vlSelfRef.d2_small_top__DOT__linear__DOT__state))) {
                __Vdly__d2_small_top__DOT__linear__DOT__state = 0U;
            } else if ((1U & (IData)(vlSelfRef.d2_small_top__DOT__linear__DOT__state))) {
                __Vdly__d2_small_top__DOT__linear__DOT__state = 0U;
            } else {
                vlSelfRef.d2_small_top__DOT__linear_busy = 0U;
                vlSelfRef.d2_small_top__DOT__linear_done = 1U;
                __Vdly__d2_small_top__DOT__linear__DOT__state = 0U;
            }
        } else if ((2U & (IData)(vlSelfRef.d2_small_top__DOT__linear__DOT__state))) {
            if ((1U & (IData)(vlSelfRef.d2_small_top__DOT__linear__DOT__state))) {
                vlSelfRef.d2_small_top__DOT__linear_logit_index 
                    = vlSelfRef.d2_small_top__DOT__linear__DOT__class_idx;
                vlSelfRef.d2_small_top__DOT__linear_logit_valid = 1U;
                vlSelfRef.d2_small_top__DOT__linear_logit_data 
                    = (vlSelfRef.d2_small_top__DOT__linear__DOT__acc 
                       + (vlSelfRef.d2_small_top__DOT__linear_bias_rom__DOT__rom
                          [vlSelfRef.d2_small_top__DOT__linear__DOT__class_idx] 
                          & (- (IData)((9U >= (IData)(vlSelfRef.d2_small_top__DOT__linear__DOT__class_idx))))));
                if ((9U == (IData)(vlSelfRef.d2_small_top__DOT__linear__DOT__class_idx))) {
                    __Vdly__d2_small_top__DOT__linear__DOT__state = 4U;
                } else {
                    vlSelfRef.d2_small_top__DOT__linear__DOT__class_idx 
                        = (0x0000000fU & ((IData)(1U) 
                                          + (IData)(vlSelfRef.d2_small_top__DOT__linear__DOT__class_idx)));
                    vlSelfRef.__Vdly__d2_small_top__DOT__linear__DOT__feature_idx = 0U;
                    __Vdly__d2_small_top__DOT__linear__DOT__acc = 0U;
                    __Vdly__d2_small_top__DOT__linear__DOT__state = 1U;
                }
            } else {
                __Vdly__d2_small_top__DOT__linear__DOT__acc 
                    = (vlSelfRef.d2_small_top__DOT__linear__DOT__acc 
                       + VL_MULS_III(32, VL_EXTENDS_II(32,8, (IData)(vlSelfRef.d2_small_top__DOT__buf_a_rdata)), 
                                     VL_EXTENDS_II(32,8, (IData)(vlSelfRef.d2_small_top__DOT__linear_weight_data))));
                if ((0x0000003fU == (IData)(vlSelfRef.d2_small_top__DOT__linear__DOT__feature_idx))) {
                    __Vdly__d2_small_top__DOT__linear__DOT__state = 3U;
                } else {
                    vlSelfRef.__Vdly__d2_small_top__DOT__linear__DOT__feature_idx 
                        = (0x0000003fU & ((IData)(1U) 
                                          + (IData)(vlSelfRef.d2_small_top__DOT__linear__DOT__feature_idx)));
                    __Vdly__d2_small_top__DOT__linear__DOT__state = 1U;
                }
            }
        } else if ((1U & (IData)(vlSelfRef.d2_small_top__DOT__linear__DOT__state))) {
            __Vdly__d2_small_top__DOT__linear__DOT__state = 2U;
        } else {
            vlSelfRef.d2_small_top__DOT__linear_busy = 0U;
            if (vlSelfRef.linear_start) {
                vlSelfRef.d2_small_top__DOT__linear__DOT__class_idx = 0U;
                vlSelfRef.__Vdly__d2_small_top__DOT__linear__DOT__feature_idx = 0U;
                __Vdly__d2_small_top__DOT__linear__DOT__acc = 0U;
                vlSelfRef.d2_small_top__DOT__linear_busy = 1U;
                __Vdly__d2_small_top__DOT__linear__DOT__state = 1U;
            }
        }
    } else {
        __Vdly__d2_small_top__DOT__logit0 = 0U;
        vlSelfRef.d2_small_top__DOT__logit2 = 0U;
        vlSelfRef.d2_small_top__DOT__logit7 = 0U;
        vlSelfRef.d2_small_top__DOT__logit8 = 0U;
        vlSelfRef.d2_small_top__DOT__logit4 = 0U;
        vlSelfRef.d2_small_top__DOT__logit9 = 0U;
        vlSelfRef.d2_small_top__DOT__logit6 = 0U;
        vlSelfRef.d2_small_top__DOT__logit5 = 0U;
        vlSelfRef.d2_small_top__DOT__logit3 = 0U;
        vlSelfRef.d2_small_top__DOT__logit1 = 0U;
        __Vdly__d2_small_top__DOT__fsm__DOT__state = 0U;
        vlSelfRef.__Vdly__conv1_start = 0U;
        vlSelfRef.__Vdly__pool1_start = 0U;
        vlSelfRef.__Vdly__conv2_start = 0U;
        vlSelfRef.__Vdly__pool2_start = 0U;
        vlSelfRef.__Vdly__conv3_start = 0U;
        vlSelfRef.__Vdly__gap_start = 0U;
        __Vdly__linear_start = 0U;
        vlSelfRef.argmax_valid = 0U;
        vlSelfRef.busy = 0U;
        vlSelfRef.done = 0U;
        vlSelfRef.__Vdly__state_dbg = 0U;
        vlSelfRef.d2_small_top__DOT__linear__DOT__class_idx = 0U;
        __Vdly__d2_small_top__DOT__linear__DOT__state = 0U;
        vlSelfRef.__Vdly__d2_small_top__DOT__linear__DOT__feature_idx = 0U;
        __Vdly__d2_small_top__DOT__linear__DOT__acc = 0U;
        vlSelfRef.d2_small_top__DOT__linear_logit_index = 0U;
        vlSelfRef.d2_small_top__DOT__linear_logit_data = 0U;
        vlSelfRef.d2_small_top__DOT__linear_logit_valid = 0U;
        vlSelfRef.d2_small_top__DOT__linear_busy = 0U;
        vlSelfRef.d2_small_top__DOT__linear_done = 0U;
    }
    vlSelfRef.d2_small_top__DOT__logit0 = __Vdly__d2_small_top__DOT__logit0;
    vlSelfRef.d2_small_top__DOT__fsm__DOT__state = __Vdly__d2_small_top__DOT__fsm__DOT__state;
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
    vlSelfRef.d2_small_top__DOT__conv3__DOT__controller_done_d2 
        = ((IData)(vlSelfRef.rst_n) && (IData)(vlSelfRef.d2_small_top__DOT__conv3__DOT__controller_done_d1));
    vlSelfRef.d2_small_top__DOT__conv2__DOT__controller_done_d2 
        = ((IData)(vlSelfRef.rst_n) && (IData)(vlSelfRef.d2_small_top__DOT__conv2__DOT__controller_done_d1));
    vlSelfRef.d2_small_top__DOT__conv1__DOT__controller_done_d2 
        = ((IData)(vlSelfRef.rst_n) && (IData)(vlSelfRef.d2_small_top__DOT__conv1__DOT__controller_done_d1));
    vlSelfRef.linear_start = __Vdly__linear_start;
    vlSelfRef.d2_small_top__DOT__linear__DOT__state 
        = __Vdly__d2_small_top__DOT__linear__DOT__state;
    vlSelfRef.d2_small_top__DOT__linear__DOT__acc = __Vdly__d2_small_top__DOT__linear__DOT__acc;
    vlSelfRef.d2_small_top__DOT__conv3__DOT__controller_done_d1 
        = ((IData)(vlSelfRef.rst_n) && (IData)(vlSelfRef.d2_small_top__DOT__conv3__DOT__controller_done));
    vlSelfRef.d2_small_top__DOT__conv2__DOT__controller_done_d1 
        = ((IData)(vlSelfRef.rst_n) && (IData)(vlSelfRef.d2_small_top__DOT__conv2__DOT__controller_done));
    vlSelfRef.d2_small_top__DOT__conv1__DOT__controller_done_d1 
        = ((IData)(vlSelfRef.rst_n) && (IData)(vlSelfRef.d2_small_top__DOT__conv1__DOT__controller_done));
}

void Vd2_small_top___024root___nba_sequent__TOP__1(Vd2_small_top___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vd2_small_top___024root___nba_sequent__TOP__1\n"); );
    Vd2_small_top__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    vlSelfRef.__VdlySet__d2_small_top__DOT__buffer_a__DOT__mem__v0 = 0U;
    vlSelfRef.__VdlySet__d2_small_top__DOT__buffer_b__DOT__mem__v0 = 0U;
    if (((IData)(vlSelfRef.__VdfgRegularize_h6e95ff9d_0_0) 
         | ((IData)(vlSelfRef.d2_small_top__DOT__pool1_out_we) 
            | ((IData)(vlSelfRef.d2_small_top__DOT__pool2_out_we) 
               | (IData)(vlSelfRef.d2_small_top__DOT__gap_out_we))))) {
        if (vlSelfRef.__VdfgRegularize_h6e95ff9d_0_0) {
            vlSelfRef.__VdlyVal__d2_small_top__DOT__buffer_a__DOT__mem__v0 
                = vlSelfRef.input_data;
            vlSelfRef.__VdlyDim0__d2_small_top__DOT__buffer_a__DOT__mem__v0 
                = (0x00001fffU & vlSelfRef.input_addr);
        } else if (vlSelfRef.d2_small_top__DOT__pool1_out_we) {
            vlSelfRef.__VdlyVal__d2_small_top__DOT__buffer_a__DOT__mem__v0 
                = vlSelfRef.d2_small_top__DOT__pool1_out_data;
            vlSelfRef.__VdlyDim0__d2_small_top__DOT__buffer_a__DOT__mem__v0 
                = (0x00001fffU & vlSelfRef.d2_small_top__DOT__pool1_out_addr);
        } else if (vlSelfRef.d2_small_top__DOT__pool2_out_we) {
            vlSelfRef.__VdlyVal__d2_small_top__DOT__buffer_a__DOT__mem__v0 
                = vlSelfRef.d2_small_top__DOT__pool2_out_data;
            vlSelfRef.__VdlyDim0__d2_small_top__DOT__buffer_a__DOT__mem__v0 
                = (0x00001fffU & vlSelfRef.d2_small_top__DOT__pool2_out_addr);
        } else {
            vlSelfRef.__VdlyVal__d2_small_top__DOT__buffer_a__DOT__mem__v0 
                = vlSelfRef.d2_small_top__DOT__gap_out_data;
            vlSelfRef.__VdlyDim0__d2_small_top__DOT__buffer_a__DOT__mem__v0 
                = (0x00001fffU & vlSelfRef.d2_small_top__DOT__gap_out_addr);
        }
        vlSelfRef.__VdlySet__d2_small_top__DOT__buffer_a__DOT__mem__v0 = 1U;
    }
    if (((IData)(vlSelfRef.d2_small_top__DOT__conv1__DOT__postprocess_valid) 
         | ((IData)(vlSelfRef.d2_small_top__DOT__conv2__DOT__postprocess_valid) 
            | (IData)(vlSelfRef.d2_small_top__DOT__conv3__DOT__postprocess_valid)))) {
        if (vlSelfRef.d2_small_top__DOT__conv1__DOT__postprocess_valid) {
            vlSelfRef.__VdlyVal__d2_small_top__DOT__buffer_b__DOT__mem__v0 
                = vlSelfRef.d2_small_top__DOT__conv1__DOT__out_data_reg;
            vlSelfRef.__VdlyDim0__d2_small_top__DOT__buffer_b__DOT__mem__v0 
                = (0x00003fffU & vlSelfRef.d2_small_top__DOT__conv1__DOT__out_addr_reg);
        } else if (vlSelfRef.d2_small_top__DOT__conv2__DOT__postprocess_valid) {
            vlSelfRef.__VdlyVal__d2_small_top__DOT__buffer_b__DOT__mem__v0 
                = vlSelfRef.d2_small_top__DOT__conv2__DOT__out_data_reg;
            vlSelfRef.__VdlyDim0__d2_small_top__DOT__buffer_b__DOT__mem__v0 
                = (0x00003fffU & vlSelfRef.d2_small_top__DOT__conv2__DOT__out_addr_reg);
        } else {
            vlSelfRef.__VdlyVal__d2_small_top__DOT__buffer_b__DOT__mem__v0 
                = vlSelfRef.d2_small_top__DOT__conv3__DOT__out_data_reg;
            vlSelfRef.__VdlyDim0__d2_small_top__DOT__buffer_b__DOT__mem__v0 
                = (0x00003fffU & vlSelfRef.d2_small_top__DOT__conv3__DOT__out_addr_reg);
        }
        vlSelfRef.__VdlySet__d2_small_top__DOT__buffer_b__DOT__mem__v0 = 1U;
    }
    vlSelfRef.d2_small_top__DOT__linear_weight_data 
        = ((0x027fU >= (IData)(vlSelfRef.d2_small_top__DOT__linear_weight_rom__DOT__rom_addr))
            ? vlSelfRef.d2_small_top__DOT__linear_weight_rom__DOT__rom
           [vlSelfRef.d2_small_top__DOT__linear_weight_rom__DOT__rom_addr]
            : 0U);
}

void Vd2_small_top___024root___nba_sequent__TOP__2(Vd2_small_top___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vd2_small_top___024root___nba_sequent__TOP__2\n"); );
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
    if (vlSelfRef.rst_n) {
        vlSelfRef.d2_small_top__DOT__gap_out_we = 0U;
        vlSelfRef.d2_small_top__DOT__gap_done = 0U;
        if ((4U & (IData)(vlSelfRef.d2_small_top__DOT__gap__DOT__state))) {
            if ((2U & (IData)(vlSelfRef.d2_small_top__DOT__gap__DOT__state))) {
                vlSelfRef.__Vdly__d2_small_top__DOT__gap__DOT__state = 0U;
            } else if ((1U & (IData)(vlSelfRef.d2_small_top__DOT__gap__DOT__state))) {
                vlSelfRef.__Vdly__d2_small_top__DOT__gap__DOT__state = 0U;
            } else {
                vlSelfRef.d2_small_top__DOT__gap_busy = 0U;
                vlSelfRef.d2_small_top__DOT__gap_done = 1U;
                vlSelfRef.__Vdly__d2_small_top__DOT__gap__DOT__state = 0U;
            }
        } else if ((2U & (IData)(vlSelfRef.d2_small_top__DOT__gap__DOT__state))) {
            if ((1U & (IData)(vlSelfRef.d2_small_top__DOT__gap__DOT__state))) {
                vlSelfRef.d2_small_top__DOT__gap_out_addr 
                    = (0x0001ffffU & (IData)(vlSelfRef.d2_small_top__DOT__gap__DOT__ch));
                vlSelfRef.d2_small_top__DOT__gap_out_data 
                    = (0x000000ffU & (vlSelfRef.d2_small_top__DOT__gap__DOT__sum 
                                      >> 6U));
                vlSelfRef.d2_small_top__DOT__gap_out_we = 1U;
                vlSelfRef.__Vdly__d2_small_top__DOT__gap__DOT__idx = 0U;
                if ((0x003fU == (IData)(vlSelfRef.d2_small_top__DOT__gap__DOT__ch))) {
                    vlSelfRef.__Vdly__d2_small_top__DOT__gap__DOT__state = 4U;
                } else {
                    vlSelfRef.__Vdly__d2_small_top__DOT__gap__DOT__ch 
                        = (0x0000ffffU & ((IData)(1U) 
                                          + (IData)(vlSelfRef.d2_small_top__DOT__gap__DOT__ch)));
                    vlSelfRef.__Vdly__d2_small_top__DOT__gap__DOT__state = 1U;
                }
                vlSelfRef.__Vdly__d2_small_top__DOT__gap__DOT__sum = 0U;
            } else {
                vlSelfRef.__Vdly__d2_small_top__DOT__gap__DOT__sum 
                    = (vlSelfRef.d2_small_top__DOT__gap__DOT__sum 
                       + (((- (IData)((1U & ((IData)(vlSelfRef.d2_small_top__DOT__buf_b_rdata) 
                                             >> 7U)))) 
                           << 8U) | (IData)(vlSelfRef.d2_small_top__DOT__buf_b_rdata)));
                if ((0x003fU == (IData)(vlSelfRef.d2_small_top__DOT__gap__DOT__idx))) {
                    vlSelfRef.__Vdly__d2_small_top__DOT__gap__DOT__state = 3U;
                } else {
                    vlSelfRef.__Vdly__d2_small_top__DOT__gap__DOT__idx 
                        = (0x0000ffffU & ((IData)(1U) 
                                          + (IData)(vlSelfRef.d2_small_top__DOT__gap__DOT__idx)));
                    vlSelfRef.__Vdly__d2_small_top__DOT__gap__DOT__state = 1U;
                }
            }
        } else if ((1U & (IData)(vlSelfRef.d2_small_top__DOT__gap__DOT__state))) {
            vlSelfRef.__Vdly__d2_small_top__DOT__gap__DOT__state = 2U;
        } else {
            vlSelfRef.d2_small_top__DOT__gap_busy = 0U;
            if (vlSelfRef.gap_start) {
                vlSelfRef.__Vdly__d2_small_top__DOT__gap__DOT__ch = 0U;
                vlSelfRef.__Vdly__d2_small_top__DOT__gap__DOT__idx = 0U;
                vlSelfRef.__Vdly__d2_small_top__DOT__gap__DOT__sum = 0U;
                vlSelfRef.d2_small_top__DOT__gap_busy = 1U;
                vlSelfRef.__Vdly__d2_small_top__DOT__gap__DOT__state = 1U;
            }
        }
        vlSelfRef.gap_start = vlSelfRef.__Vdly__gap_start;
        vlSelfRef.d2_small_top__DOT__gap__DOT__state 
            = vlSelfRef.__Vdly__d2_small_top__DOT__gap__DOT__state;
        vlSelfRef.d2_small_top__DOT__gap__DOT__sum 
            = vlSelfRef.__Vdly__d2_small_top__DOT__gap__DOT__sum;
        vlSelfRef.d2_small_top__DOT__pool1_out_we = 0U;
        vlSelfRef.d2_small_top__DOT__pool1_done = 0U;
        if ((4U & (IData)(vlSelfRef.d2_small_top__DOT__pool1__DOT__state))) {
            if ((2U & (IData)(vlSelfRef.d2_small_top__DOT__pool1__DOT__state))) {
                vlSelfRef.__Vdly__d2_small_top__DOT__pool1__DOT__state = 0U;
            } else if ((1U & (IData)(vlSelfRef.d2_small_top__DOT__pool1__DOT__state))) {
                vlSelfRef.__Vdly__d2_small_top__DOT__pool1__DOT__state = 0U;
            } else {
                vlSelfRef.d2_small_top__DOT__pool1_busy = 0U;
                vlSelfRef.d2_small_top__DOT__pool1_done = 1U;
                vlSelfRef.__Vdly__d2_small_top__DOT__pool1__DOT__state = 0U;
            }
        } else if ((2U & (IData)(vlSelfRef.d2_small_top__DOT__pool1__DOT__state))) {
            if ((1U & (IData)(vlSelfRef.d2_small_top__DOT__pool1__DOT__state))) {
                vlSelfRef.d2_small_top__DOT__pool1_out_addr 
                    = (0x0001ffffU & (VL_SHIFTL_III(17,32,32, (IData)(vlSelfRef.d2_small_top__DOT__pool1__DOT__ch), 8U) 
                                      + ((IData)(vlSelfRef.d2_small_top__DOT__pool1__DOT__col) 
                                         + ((IData)(vlSelfRef.d2_small_top__DOT__pool1__DOT__row) 
                                            << 4U))));
                vlSelfRef.d2_small_top__DOT__pool1_out_data 
                    = vlSelfRef.d2_small_top__DOT__pool1__DOT__current_max;
                vlSelfRef.d2_small_top__DOT__pool1_out_we = 1U;
                vlSelfRef.__Vdly__d2_small_top__DOT__pool1__DOT__tap = 0U;
                if (((0x000fU == (IData)(vlSelfRef.d2_small_top__DOT__pool1__DOT__ch)) 
                     & ((0x000fU == (IData)(vlSelfRef.d2_small_top__DOT__pool1__DOT__row)) 
                        & (0x000fU == (IData)(vlSelfRef.d2_small_top__DOT__pool1__DOT__col))))) {
                    vlSelfRef.__Vdly__d2_small_top__DOT__pool1__DOT__state = 4U;
                } else {
                    if ((0x000fU != (IData)(vlSelfRef.d2_small_top__DOT__pool1__DOT__col))) {
                        vlSelfRef.__Vdly__d2_small_top__DOT__pool1__DOT__col 
                            = (0x0000ffffU & ((IData)(1U) 
                                              + (IData)(vlSelfRef.d2_small_top__DOT__pool1__DOT__col)));
                    } else {
                        if ((0x000fU != (IData)(vlSelfRef.d2_small_top__DOT__pool1__DOT__row))) {
                            vlSelfRef.__Vdly__d2_small_top__DOT__pool1__DOT__row 
                                = (0x0000ffffU & ((IData)(1U) 
                                                  + (IData)(vlSelfRef.d2_small_top__DOT__pool1__DOT__row)));
                        } else {
                            vlSelfRef.__Vdly__d2_small_top__DOT__pool1__DOT__ch 
                                = (0x0000ffffU & ((IData)(1U) 
                                                  + (IData)(vlSelfRef.d2_small_top__DOT__pool1__DOT__ch)));
                            vlSelfRef.__Vdly__d2_small_top__DOT__pool1__DOT__row = 0U;
                        }
                        vlSelfRef.__Vdly__d2_small_top__DOT__pool1__DOT__col = 0U;
                    }
                    vlSelfRef.__Vdly__d2_small_top__DOT__pool1__DOT__state = 1U;
                }
                vlSelfRef.__Vdly__d2_small_top__DOT__pool1__DOT__current_max = 0x80U;
            } else {
                if (((0U == (IData)(vlSelfRef.d2_small_top__DOT__pool1__DOT__tap)) 
                     | VL_GTS_III(8, (IData)(vlSelfRef.d2_small_top__DOT__buf_b_rdata), (IData)(vlSelfRef.d2_small_top__DOT__pool1__DOT__current_max)))) {
                    vlSelfRef.__Vdly__d2_small_top__DOT__pool1__DOT__current_max 
                        = vlSelfRef.d2_small_top__DOT__buf_b_rdata;
                }
                if ((3U == (IData)(vlSelfRef.d2_small_top__DOT__pool1__DOT__tap))) {
                    vlSelfRef.__Vdly__d2_small_top__DOT__pool1__DOT__state = 3U;
                } else {
                    vlSelfRef.__Vdly__d2_small_top__DOT__pool1__DOT__tap 
                        = (3U & ((IData)(1U) + (IData)(vlSelfRef.d2_small_top__DOT__pool1__DOT__tap)));
                    vlSelfRef.__Vdly__d2_small_top__DOT__pool1__DOT__state = 1U;
                }
            }
        } else if ((1U & (IData)(vlSelfRef.d2_small_top__DOT__pool1__DOT__state))) {
            vlSelfRef.__Vdly__d2_small_top__DOT__pool1__DOT__state = 2U;
        } else {
            vlSelfRef.d2_small_top__DOT__pool1_busy = 0U;
            if (vlSelfRef.pool1_start) {
                vlSelfRef.__Vdly__d2_small_top__DOT__pool1__DOT__ch = 0U;
                vlSelfRef.__Vdly__d2_small_top__DOT__pool1__DOT__row = 0U;
                vlSelfRef.__Vdly__d2_small_top__DOT__pool1__DOT__col = 0U;
                vlSelfRef.__Vdly__d2_small_top__DOT__pool1__DOT__tap = 0U;
                vlSelfRef.__Vdly__d2_small_top__DOT__pool1__DOT__current_max = 0x80U;
                vlSelfRef.d2_small_top__DOT__pool1_busy = 1U;
                vlSelfRef.__Vdly__d2_small_top__DOT__pool1__DOT__state = 1U;
            }
        }
        vlSelfRef.pool1_start = vlSelfRef.__Vdly__pool1_start;
        vlSelfRef.d2_small_top__DOT__pool1__DOT__state 
            = vlSelfRef.__Vdly__d2_small_top__DOT__pool1__DOT__state;
        vlSelfRef.d2_small_top__DOT__pool1__DOT__current_max 
            = vlSelfRef.__Vdly__d2_small_top__DOT__pool1__DOT__current_max;
        vlSelfRef.d2_small_top__DOT__pool2_out_we = 0U;
        vlSelfRef.d2_small_top__DOT__pool2_done = 0U;
        if ((4U & (IData)(vlSelfRef.d2_small_top__DOT__pool2__DOT__state))) {
            if ((2U & (IData)(vlSelfRef.d2_small_top__DOT__pool2__DOT__state))) {
                vlSelfRef.__Vdly__d2_small_top__DOT__pool2__DOT__state = 0U;
            } else if ((1U & (IData)(vlSelfRef.d2_small_top__DOT__pool2__DOT__state))) {
                vlSelfRef.__Vdly__d2_small_top__DOT__pool2__DOT__state = 0U;
            } else {
                vlSelfRef.d2_small_top__DOT__pool2_busy = 0U;
                vlSelfRef.d2_small_top__DOT__pool2_done = 1U;
                vlSelfRef.__Vdly__d2_small_top__DOT__pool2__DOT__state = 0U;
            }
        } else if ((2U & (IData)(vlSelfRef.d2_small_top__DOT__pool2__DOT__state))) {
            if ((1U & (IData)(vlSelfRef.d2_small_top__DOT__pool2__DOT__state))) {
                vlSelfRef.d2_small_top__DOT__pool2_out_addr 
                    = (0x0001ffffU & (VL_SHIFTL_III(17,32,32, (IData)(vlSelfRef.d2_small_top__DOT__pool2__DOT__ch), 6U) 
                                      + ((IData)(vlSelfRef.d2_small_top__DOT__pool2__DOT__col) 
                                         + ((IData)(vlSelfRef.d2_small_top__DOT__pool2__DOT__row) 
                                            << 3U))));
                vlSelfRef.d2_small_top__DOT__pool2_out_data 
                    = vlSelfRef.d2_small_top__DOT__pool2__DOT__current_max;
                vlSelfRef.d2_small_top__DOT__pool2_out_we = 1U;
                vlSelfRef.__Vdly__d2_small_top__DOT__pool2__DOT__tap = 0U;
                if (((0x001fU == (IData)(vlSelfRef.d2_small_top__DOT__pool2__DOT__ch)) 
                     & ((7U == (IData)(vlSelfRef.d2_small_top__DOT__pool2__DOT__row)) 
                        & (7U == (IData)(vlSelfRef.d2_small_top__DOT__pool2__DOT__col))))) {
                    vlSelfRef.__Vdly__d2_small_top__DOT__pool2__DOT__state = 4U;
                } else {
                    if ((7U != (IData)(vlSelfRef.d2_small_top__DOT__pool2__DOT__col))) {
                        vlSelfRef.__Vdly__d2_small_top__DOT__pool2__DOT__col 
                            = (0x0000ffffU & ((IData)(1U) 
                                              + (IData)(vlSelfRef.d2_small_top__DOT__pool2__DOT__col)));
                    } else {
                        if ((7U != (IData)(vlSelfRef.d2_small_top__DOT__pool2__DOT__row))) {
                            vlSelfRef.__Vdly__d2_small_top__DOT__pool2__DOT__row 
                                = (0x0000ffffU & ((IData)(1U) 
                                                  + (IData)(vlSelfRef.d2_small_top__DOT__pool2__DOT__row)));
                        } else {
                            vlSelfRef.__Vdly__d2_small_top__DOT__pool2__DOT__ch 
                                = (0x0000ffffU & ((IData)(1U) 
                                                  + (IData)(vlSelfRef.d2_small_top__DOT__pool2__DOT__ch)));
                            vlSelfRef.__Vdly__d2_small_top__DOT__pool2__DOT__row = 0U;
                        }
                        vlSelfRef.__Vdly__d2_small_top__DOT__pool2__DOT__col = 0U;
                    }
                    vlSelfRef.__Vdly__d2_small_top__DOT__pool2__DOT__state = 1U;
                }
                vlSelfRef.__Vdly__d2_small_top__DOT__pool2__DOT__current_max = 0x80U;
            } else {
                if (((0U == (IData)(vlSelfRef.d2_small_top__DOT__pool2__DOT__tap)) 
                     | VL_GTS_III(8, (IData)(vlSelfRef.d2_small_top__DOT__buf_b_rdata), (IData)(vlSelfRef.d2_small_top__DOT__pool2__DOT__current_max)))) {
                    vlSelfRef.__Vdly__d2_small_top__DOT__pool2__DOT__current_max 
                        = vlSelfRef.d2_small_top__DOT__buf_b_rdata;
                }
                if ((3U == (IData)(vlSelfRef.d2_small_top__DOT__pool2__DOT__tap))) {
                    vlSelfRef.__Vdly__d2_small_top__DOT__pool2__DOT__state = 3U;
                } else {
                    vlSelfRef.__Vdly__d2_small_top__DOT__pool2__DOT__tap 
                        = (3U & ((IData)(1U) + (IData)(vlSelfRef.d2_small_top__DOT__pool2__DOT__tap)));
                    vlSelfRef.__Vdly__d2_small_top__DOT__pool2__DOT__state = 1U;
                }
            }
        } else if ((1U & (IData)(vlSelfRef.d2_small_top__DOT__pool2__DOT__state))) {
            vlSelfRef.__Vdly__d2_small_top__DOT__pool2__DOT__state = 2U;
        } else {
            vlSelfRef.d2_small_top__DOT__pool2_busy = 0U;
            if (vlSelfRef.pool2_start) {
                vlSelfRef.__Vdly__d2_small_top__DOT__pool2__DOT__ch = 0U;
                vlSelfRef.__Vdly__d2_small_top__DOT__pool2__DOT__row = 0U;
                vlSelfRef.__Vdly__d2_small_top__DOT__pool2__DOT__col = 0U;
                vlSelfRef.__Vdly__d2_small_top__DOT__pool2__DOT__tap = 0U;
                vlSelfRef.__Vdly__d2_small_top__DOT__pool2__DOT__current_max = 0x80U;
                vlSelfRef.d2_small_top__DOT__pool2_busy = 1U;
                vlSelfRef.__Vdly__d2_small_top__DOT__pool2__DOT__state = 1U;
            }
        }
        vlSelfRef.pool2_start = vlSelfRef.__Vdly__pool2_start;
        vlSelfRef.d2_small_top__DOT__pool2__DOT__state 
            = vlSelfRef.__Vdly__d2_small_top__DOT__pool2__DOT__state;
        vlSelfRef.d2_small_top__DOT__pool2__DOT__current_max 
            = vlSelfRef.__Vdly__d2_small_top__DOT__pool2__DOT__current_max;
        if (vlSelfRef.d2_small_top__DOT__conv1__DOT__product_valid) {
            vlSelfRef.d2_small_top__DOT__conv1__DOT__postprocess_valid = 1U;
            vlSelfRef.d2_small_top__DOT__conv1__DOT__out_data_reg 
                = (VL_LTS_III(32, 0x0000007fU, vlSelfRef.d2_small_top__DOT__conv1__DOT__activated)
                    ? 0x0000007fU : (VL_GTS_III(32, 0xffffff80U, vlSelfRef.d2_small_top__DOT__conv1__DOT__activated)
                                      ? 0x00000080U
                                      : (0x000000ffU 
                                         & vlSelfRef.d2_small_top__DOT__conv1__DOT__activated)));
            vlSelfRef.d2_small_top__DOT__conv1__DOT__out_addr_reg 
                = vlSelfRef.d2_small_top__DOT__conv1__DOT__product_addr_reg;
        } else {
            vlSelfRef.d2_small_top__DOT__conv1__DOT__postprocess_valid = 0U;
        }
        if (vlSelfRef.d2_small_top__DOT__conv2__DOT__product_valid) {
            vlSelfRef.d2_small_top__DOT__conv2__DOT__postprocess_valid = 1U;
            vlSelfRef.d2_small_top__DOT__conv2__DOT__out_data_reg 
                = (VL_LTS_III(32, 0x0000007fU, vlSelfRef.d2_small_top__DOT__conv2__DOT__activated)
                    ? 0x0000007fU : (VL_GTS_III(32, 0xffffff80U, vlSelfRef.d2_small_top__DOT__conv2__DOT__activated)
                                      ? 0x00000080U
                                      : (0x000000ffU 
                                         & vlSelfRef.d2_small_top__DOT__conv2__DOT__activated)));
            vlSelfRef.d2_small_top__DOT__conv2__DOT__out_addr_reg 
                = vlSelfRef.d2_small_top__DOT__conv2__DOT__product_addr_reg;
        } else {
            vlSelfRef.d2_small_top__DOT__conv2__DOT__postprocess_valid = 0U;
        }
        if (vlSelfRef.d2_small_top__DOT__conv3__DOT__product_valid) {
            vlSelfRef.d2_small_top__DOT__conv3__DOT__postprocess_valid = 1U;
            vlSelfRef.d2_small_top__DOT__conv3__DOT__out_data_reg 
                = (VL_LTS_III(32, 0x0000007fU, vlSelfRef.d2_small_top__DOT__conv3__DOT__activated)
                    ? 0x0000007fU : (VL_GTS_III(32, 0xffffff80U, vlSelfRef.d2_small_top__DOT__conv3__DOT__activated)
                                      ? 0x00000080U
                                      : (0x000000ffU 
                                         & vlSelfRef.d2_small_top__DOT__conv3__DOT__activated)));
            vlSelfRef.d2_small_top__DOT__conv3__DOT__out_addr_reg 
                = vlSelfRef.d2_small_top__DOT__conv3__DOT__product_addr_reg;
        } else {
            vlSelfRef.d2_small_top__DOT__conv3__DOT__postprocess_valid = 0U;
        }
        if (vlSelfRef.d2_small_top__DOT__conv1__DOT__postprocess_pending) {
            vlSelfRef.d2_small_top__DOT__conv1__DOT__product_valid = 1U;
            vlSelfRef.d2_small_top__DOT__conv1__DOT__product_reg 
                = (0x000007ffffffffffULL & VL_MULS_QQQ(43, 0x000000000000a9e3ULL, 
                                                       (0x000007ffffffffffULL 
                                                        & VL_EXTENDS_QI(43,25, 
                                                                        (0x01ffffffU 
                                                                         & vlSelfRef.d2_small_top__DOT__conv1__DOT__biased_sum_reg)))));
            vlSelfRef.d2_small_top__DOT__conv1__DOT__product_addr_reg 
                = vlSelfRef.d2_small_top__DOT__conv1__DOT__controller_out_addr;
        } else {
            vlSelfRef.d2_small_top__DOT__conv1__DOT__product_valid = 0U;
        }
    } else {
        vlSelfRef.__Vdly__d2_small_top__DOT__gap__DOT__ch = 0U;
        vlSelfRef.__Vdly__d2_small_top__DOT__gap__DOT__state = 0U;
        vlSelfRef.__Vdly__d2_small_top__DOT__gap__DOT__idx = 0U;
        vlSelfRef.__Vdly__d2_small_top__DOT__gap__DOT__sum = 0U;
        vlSelfRef.d2_small_top__DOT__gap_out_we = 0U;
        vlSelfRef.d2_small_top__DOT__gap_out_addr = 0U;
        vlSelfRef.d2_small_top__DOT__gap_out_data = 0U;
        vlSelfRef.d2_small_top__DOT__gap_busy = 0U;
        vlSelfRef.d2_small_top__DOT__gap_done = 0U;
        vlSelfRef.gap_start = vlSelfRef.__Vdly__gap_start;
        vlSelfRef.d2_small_top__DOT__gap__DOT__state 
            = vlSelfRef.__Vdly__d2_small_top__DOT__gap__DOT__state;
        vlSelfRef.d2_small_top__DOT__gap__DOT__sum 
            = vlSelfRef.__Vdly__d2_small_top__DOT__gap__DOT__sum;
        vlSelfRef.__Vdly__d2_small_top__DOT__pool1__DOT__ch = 0U;
        vlSelfRef.__Vdly__d2_small_top__DOT__pool1__DOT__row = 0U;
        vlSelfRef.__Vdly__d2_small_top__DOT__pool1__DOT__col = 0U;
        vlSelfRef.__Vdly__d2_small_top__DOT__pool1__DOT__state = 0U;
        vlSelfRef.__Vdly__d2_small_top__DOT__pool1__DOT__tap = 0U;
        vlSelfRef.__Vdly__d2_small_top__DOT__pool1__DOT__current_max = 0x80U;
        vlSelfRef.d2_small_top__DOT__pool1_out_data = 0U;
        vlSelfRef.d2_small_top__DOT__pool1_out_addr = 0U;
        vlSelfRef.d2_small_top__DOT__pool1_out_we = 0U;
        vlSelfRef.d2_small_top__DOT__pool1_busy = 0U;
        vlSelfRef.d2_small_top__DOT__pool1_done = 0U;
        vlSelfRef.pool1_start = vlSelfRef.__Vdly__pool1_start;
        vlSelfRef.d2_small_top__DOT__pool1__DOT__state 
            = vlSelfRef.__Vdly__d2_small_top__DOT__pool1__DOT__state;
        vlSelfRef.d2_small_top__DOT__pool1__DOT__current_max 
            = vlSelfRef.__Vdly__d2_small_top__DOT__pool1__DOT__current_max;
        vlSelfRef.__Vdly__d2_small_top__DOT__pool2__DOT__ch = 0U;
        vlSelfRef.__Vdly__d2_small_top__DOT__pool2__DOT__row = 0U;
        vlSelfRef.__Vdly__d2_small_top__DOT__pool2__DOT__col = 0U;
        vlSelfRef.__Vdly__d2_small_top__DOT__pool2__DOT__state = 0U;
        vlSelfRef.__Vdly__d2_small_top__DOT__pool2__DOT__tap = 0U;
        vlSelfRef.__Vdly__d2_small_top__DOT__pool2__DOT__current_max = 0x80U;
        vlSelfRef.d2_small_top__DOT__pool2_out_data = 0U;
        vlSelfRef.d2_small_top__DOT__pool2_out_addr = 0U;
        vlSelfRef.d2_small_top__DOT__pool2_out_we = 0U;
        vlSelfRef.d2_small_top__DOT__pool2_busy = 0U;
        vlSelfRef.d2_small_top__DOT__pool2_done = 0U;
        vlSelfRef.pool2_start = vlSelfRef.__Vdly__pool2_start;
        vlSelfRef.d2_small_top__DOT__pool2__DOT__state 
            = vlSelfRef.__Vdly__d2_small_top__DOT__pool2__DOT__state;
        vlSelfRef.d2_small_top__DOT__pool2__DOT__current_max 
            = vlSelfRef.__Vdly__d2_small_top__DOT__pool2__DOT__current_max;
        vlSelfRef.d2_small_top__DOT__conv1__DOT__postprocess_valid = 0U;
        vlSelfRef.d2_small_top__DOT__conv1__DOT__out_data_reg = 0U;
        vlSelfRef.d2_small_top__DOT__conv1__DOT__out_addr_reg = 0U;
        vlSelfRef.d2_small_top__DOT__conv2__DOT__postprocess_valid = 0U;
        vlSelfRef.d2_small_top__DOT__conv2__DOT__out_data_reg = 0U;
        vlSelfRef.d2_small_top__DOT__conv2__DOT__out_addr_reg = 0U;
        vlSelfRef.d2_small_top__DOT__conv3__DOT__postprocess_valid = 0U;
        vlSelfRef.d2_small_top__DOT__conv3__DOT__out_data_reg = 0U;
        vlSelfRef.d2_small_top__DOT__conv3__DOT__out_addr_reg = 0U;
        vlSelfRef.d2_small_top__DOT__conv1__DOT__product_reg = 0ULL;
        vlSelfRef.d2_small_top__DOT__conv1__DOT__product_valid = 0U;
        vlSelfRef.d2_small_top__DOT__conv1__DOT__product_addr_reg = 0U;
    }
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
    vlSelfRef.d2_small_top__DOT__conv1__DOT__postprocess_pending 
        = ((IData)(vlSelfRef.rst_n) && (IData)(vlSelfRef.d2_small_top__DOT__conv1__DOT__engine_done));
    if (vlSelfRef.rst_n) {
        if (vlSelfRef.d2_small_top__DOT__conv2__DOT__postprocess_pending) {
            vlSelfRef.d2_small_top__DOT__conv2__DOT__product_valid = 1U;
            vlSelfRef.d2_small_top__DOT__conv2__DOT__product_reg 
                = (0x000007ffffffffffULL & VL_MULS_QQQ(43, 0x0000000000011d54ULL, 
                                                       (0x000007ffffffffffULL 
                                                        & VL_EXTENDS_QI(43,25, 
                                                                        (0x01ffffffU 
                                                                         & vlSelfRef.d2_small_top__DOT__conv2__DOT__biased_sum_reg)))));
            vlSelfRef.d2_small_top__DOT__conv2__DOT__product_addr_reg 
                = vlSelfRef.d2_small_top__DOT__conv2__DOT__controller_out_addr;
        } else {
            vlSelfRef.d2_small_top__DOT__conv2__DOT__product_valid = 0U;
        }
    } else {
        vlSelfRef.d2_small_top__DOT__conv2__DOT__product_reg = 0ULL;
        vlSelfRef.d2_small_top__DOT__conv2__DOT__product_valid = 0U;
        vlSelfRef.d2_small_top__DOT__conv2__DOT__product_addr_reg = 0U;
    }
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
    vlSelfRef.d2_small_top__DOT__conv2__DOT__postprocess_pending 
        = ((IData)(vlSelfRef.rst_n) && (IData)(vlSelfRef.d2_small_top__DOT__conv2__DOT__engine_done));
    if (vlSelfRef.rst_n) {
        if (vlSelfRef.d2_small_top__DOT__conv3__DOT__postprocess_pending) {
            vlSelfRef.d2_small_top__DOT__conv3__DOT__product_valid = 1U;
            vlSelfRef.d2_small_top__DOT__conv3__DOT__product_reg 
                = (0x000007ffffffffffULL & VL_MULS_QQQ(43, 0x0000000000012c92ULL, 
                                                       (0x000007ffffffffffULL 
                                                        & VL_EXTENDS_QI(43,25, 
                                                                        (0x01ffffffU 
                                                                         & vlSelfRef.d2_small_top__DOT__conv3__DOT__biased_sum_reg)))));
            vlSelfRef.d2_small_top__DOT__conv3__DOT__product_addr_reg 
                = vlSelfRef.d2_small_top__DOT__conv3__DOT__controller_out_addr;
        } else {
            vlSelfRef.d2_small_top__DOT__conv3__DOT__product_valid = 0U;
        }
    } else {
        vlSelfRef.d2_small_top__DOT__conv3__DOT__product_reg = 0ULL;
        vlSelfRef.d2_small_top__DOT__conv3__DOT__product_valid = 0U;
        vlSelfRef.d2_small_top__DOT__conv3__DOT__product_addr_reg = 0U;
    }
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
    vlSelfRef.d2_small_top__DOT__conv3__DOT__postprocess_pending 
        = ((IData)(vlSelfRef.rst_n) && (IData)(vlSelfRef.d2_small_top__DOT__conv3__DOT__engine_done));
    if (vlSelfRef.rst_n) {
        if (vlSelfRef.d2_small_top__DOT__conv1__DOT__engine_done) {
            vlSelfRef.d2_small_top__DOT__conv1__DOT__biased_sum_reg 
                = (vlSelfRef.d2_small_top__DOT__conv1__DOT__raw_sum 
                   + vlSelfRef.d2_small_top__DOT__conv1_bias_rom__DOT__rom
                   [(0x0000000fU & (IData)(vlSelfRef.d2_small_top__DOT__conv1__DOT__out_channel))]);
        }
        vlSelfRef.__Vdly__d2_small_top__DOT__conv1__DOT__engine_start = 0U;
        vlSelfRef.d2_small_top__DOT__conv1__DOT__controller_done = 0U;
        if ((2U & (IData)(vlSelfRef.d2_small_top__DOT__conv1__DOT__controller__DOT__state))) {
            if ((1U & (IData)(vlSelfRef.d2_small_top__DOT__conv1__DOT__controller__DOT__state))) {
                vlSelfRef.d2_small_top__DOT__conv1__DOT__controller_busy = 0U;
                vlSelfRef.d2_small_top__DOT__conv1__DOT__controller_done = 1U;
                vlSelfRef.__Vdly__d2_small_top__DOT__conv1__DOT__controller__DOT__state = 0U;
            } else if (vlSelfRef.d2_small_top__DOT__conv1__DOT__engine_done) {
                vlSelfRef.d2_small_top__DOT__conv1__DOT__controller_out_addr 
                    = (0x0001ffffU & (VL_SHIFTL_III(17,32,32, (IData)(vlSelfRef.d2_small_top__DOT__conv1__DOT__out_channel), 0x0000000aU) 
                                      + (VL_SHIFTL_III(17,32,32, (IData)(vlSelfRef.d2_small_top__DOT__conv1__DOT__out_row), 5U) 
                                         + (IData)(vlSelfRef.d2_small_top__DOT__conv1__DOT__out_col))));
                if (((0x000fU == (IData)(vlSelfRef.d2_small_top__DOT__conv1__DOT__out_channel)) 
                     & ((0x001fU == (IData)(vlSelfRef.d2_small_top__DOT__conv1__DOT__out_row)) 
                        & (0x001fU == (IData)(vlSelfRef.d2_small_top__DOT__conv1__DOT__out_col))))) {
                    vlSelfRef.__Vdly__d2_small_top__DOT__conv1__DOT__controller__DOT__state = 3U;
                } else {
                    if ((0x001fU != (IData)(vlSelfRef.d2_small_top__DOT__conv1__DOT__out_col))) {
                        vlSelfRef.d2_small_top__DOT__conv1__DOT__out_col 
                            = (0x0000ffffU & ((IData)(1U) 
                                              + (IData)(vlSelfRef.d2_small_top__DOT__conv1__DOT__out_col)));
                    } else {
                        if ((0x001fU != (IData)(vlSelfRef.d2_small_top__DOT__conv1__DOT__out_row))) {
                            vlSelfRef.d2_small_top__DOT__conv1__DOT__out_row 
                                = (0x0000ffffU & ((IData)(1U) 
                                                  + (IData)(vlSelfRef.d2_small_top__DOT__conv1__DOT__out_row)));
                        } else {
                            vlSelfRef.d2_small_top__DOT__conv1__DOT__out_channel 
                                = (0x0000ffffU & ((IData)(1U) 
                                                  + (IData)(vlSelfRef.d2_small_top__DOT__conv1__DOT__out_channel)));
                            vlSelfRef.d2_small_top__DOT__conv1__DOT__out_row = 0U;
                        }
                        vlSelfRef.d2_small_top__DOT__conv1__DOT__out_col = 0U;
                    }
                    vlSelfRef.__Vdly__d2_small_top__DOT__conv1__DOT__controller__DOT__state = 1U;
                }
            }
        } else if ((1U & (IData)(vlSelfRef.d2_small_top__DOT__conv1__DOT__controller__DOT__state))) {
            vlSelfRef.__Vdly__d2_small_top__DOT__conv1__DOT__engine_start = 1U;
            vlSelfRef.__Vdly__d2_small_top__DOT__conv1__DOT__controller__DOT__state = 2U;
        } else {
            vlSelfRef.d2_small_top__DOT__conv1__DOT__controller_busy = 0U;
            if (vlSelfRef.conv1_start) {
                vlSelfRef.d2_small_top__DOT__conv1__DOT__out_row = 0U;
                vlSelfRef.d2_small_top__DOT__conv1__DOT__out_col = 0U;
                vlSelfRef.d2_small_top__DOT__conv1__DOT__out_channel = 0U;
                vlSelfRef.d2_small_top__DOT__conv1__DOT__controller_busy = 1U;
                vlSelfRef.__Vdly__d2_small_top__DOT__conv1__DOT__controller__DOT__state = 1U;
            }
        }
        if (vlSelfRef.d2_small_top__DOT__conv2__DOT__engine_done) {
            vlSelfRef.d2_small_top__DOT__conv2__DOT__biased_sum_reg 
                = (vlSelfRef.d2_small_top__DOT__conv2__DOT__raw_sum 
                   + vlSelfRef.d2_small_top__DOT__conv2_bias_rom__DOT__rom
                   [(0x0000001fU & (IData)(vlSelfRef.d2_small_top__DOT__conv2__DOT__out_channel))]);
        }
        vlSelfRef.__Vdly__d2_small_top__DOT__conv2__DOT__engine_start = 0U;
        vlSelfRef.d2_small_top__DOT__conv2__DOT__controller_done = 0U;
        if ((2U & (IData)(vlSelfRef.d2_small_top__DOT__conv2__DOT__controller__DOT__state))) {
            if ((1U & (IData)(vlSelfRef.d2_small_top__DOT__conv2__DOT__controller__DOT__state))) {
                vlSelfRef.d2_small_top__DOT__conv2__DOT__controller_busy = 0U;
                vlSelfRef.d2_small_top__DOT__conv2__DOT__controller_done = 1U;
                vlSelfRef.__Vdly__d2_small_top__DOT__conv2__DOT__controller__DOT__state = 0U;
            } else if (vlSelfRef.d2_small_top__DOT__conv2__DOT__engine_done) {
                vlSelfRef.d2_small_top__DOT__conv2__DOT__controller_out_addr 
                    = (0x0001ffffU & (VL_SHIFTL_III(17,32,32, (IData)(vlSelfRef.d2_small_top__DOT__conv2__DOT__out_channel), 8U) 
                                      + (VL_SHIFTL_III(17,32,32, (IData)(vlSelfRef.d2_small_top__DOT__conv2__DOT__out_row), 4U) 
                                         + (IData)(vlSelfRef.d2_small_top__DOT__conv2__DOT__out_col))));
                if (((0x001fU == (IData)(vlSelfRef.d2_small_top__DOT__conv2__DOT__out_channel)) 
                     & ((0x000fU == (IData)(vlSelfRef.d2_small_top__DOT__conv2__DOT__out_row)) 
                        & (0x000fU == (IData)(vlSelfRef.d2_small_top__DOT__conv2__DOT__out_col))))) {
                    vlSelfRef.__Vdly__d2_small_top__DOT__conv2__DOT__controller__DOT__state = 3U;
                } else {
                    if ((0x000fU != (IData)(vlSelfRef.d2_small_top__DOT__conv2__DOT__out_col))) {
                        vlSelfRef.d2_small_top__DOT__conv2__DOT__out_col 
                            = (0x0000ffffU & ((IData)(1U) 
                                              + (IData)(vlSelfRef.d2_small_top__DOT__conv2__DOT__out_col)));
                    } else {
                        if ((0x000fU != (IData)(vlSelfRef.d2_small_top__DOT__conv2__DOT__out_row))) {
                            vlSelfRef.d2_small_top__DOT__conv2__DOT__out_row 
                                = (0x0000ffffU & ((IData)(1U) 
                                                  + (IData)(vlSelfRef.d2_small_top__DOT__conv2__DOT__out_row)));
                        } else {
                            vlSelfRef.d2_small_top__DOT__conv2__DOT__out_channel 
                                = (0x0000ffffU & ((IData)(1U) 
                                                  + (IData)(vlSelfRef.d2_small_top__DOT__conv2__DOT__out_channel)));
                            vlSelfRef.d2_small_top__DOT__conv2__DOT__out_row = 0U;
                        }
                        vlSelfRef.d2_small_top__DOT__conv2__DOT__out_col = 0U;
                    }
                    vlSelfRef.__Vdly__d2_small_top__DOT__conv2__DOT__controller__DOT__state = 1U;
                }
            }
        } else if ((1U & (IData)(vlSelfRef.d2_small_top__DOT__conv2__DOT__controller__DOT__state))) {
            vlSelfRef.__Vdly__d2_small_top__DOT__conv2__DOT__engine_start = 1U;
            vlSelfRef.__Vdly__d2_small_top__DOT__conv2__DOT__controller__DOT__state = 2U;
        } else {
            vlSelfRef.d2_small_top__DOT__conv2__DOT__controller_busy = 0U;
            if (vlSelfRef.conv2_start) {
                vlSelfRef.d2_small_top__DOT__conv2__DOT__out_row = 0U;
                vlSelfRef.d2_small_top__DOT__conv2__DOT__out_col = 0U;
                vlSelfRef.d2_small_top__DOT__conv2__DOT__out_channel = 0U;
                vlSelfRef.d2_small_top__DOT__conv2__DOT__controller_busy = 1U;
                vlSelfRef.__Vdly__d2_small_top__DOT__conv2__DOT__controller__DOT__state = 1U;
            }
        }
        if (vlSelfRef.d2_small_top__DOT__conv3__DOT__engine_done) {
            vlSelfRef.d2_small_top__DOT__conv3__DOT__biased_sum_reg 
                = (vlSelfRef.d2_small_top__DOT__conv3__DOT__raw_sum 
                   + vlSelfRef.d2_small_top__DOT__conv3_bias_rom__DOT__rom
                   [(0x0000003fU & (IData)(vlSelfRef.d2_small_top__DOT__conv3__DOT__out_channel))]);
        }
        vlSelfRef.__Vdly__d2_small_top__DOT__conv3__DOT__engine_start = 0U;
        vlSelfRef.d2_small_top__DOT__conv3__DOT__controller_done = 0U;
        if ((2U & (IData)(vlSelfRef.d2_small_top__DOT__conv3__DOT__controller__DOT__state))) {
            if ((1U & (IData)(vlSelfRef.d2_small_top__DOT__conv3__DOT__controller__DOT__state))) {
                vlSelfRef.d2_small_top__DOT__conv3__DOT__controller_busy = 0U;
                vlSelfRef.d2_small_top__DOT__conv3__DOT__controller_done = 1U;
                vlSelfRef.__Vdly__d2_small_top__DOT__conv3__DOT__controller__DOT__state = 0U;
            } else if (vlSelfRef.d2_small_top__DOT__conv3__DOT__engine_done) {
                vlSelfRef.d2_small_top__DOT__conv3__DOT__controller_out_addr 
                    = (0x0001ffffU & (VL_SHIFTL_III(17,32,32, (IData)(vlSelfRef.d2_small_top__DOT__conv3__DOT__out_channel), 6U) 
                                      + (VL_SHIFTL_III(17,32,32, (IData)(vlSelfRef.d2_small_top__DOT__conv3__DOT__out_row), 3U) 
                                         + (IData)(vlSelfRef.d2_small_top__DOT__conv3__DOT__out_col))));
                if (((0x003fU == (IData)(vlSelfRef.d2_small_top__DOT__conv3__DOT__out_channel)) 
                     & ((7U == (IData)(vlSelfRef.d2_small_top__DOT__conv3__DOT__out_row)) 
                        & (7U == (IData)(vlSelfRef.d2_small_top__DOT__conv3__DOT__out_col))))) {
                    vlSelfRef.__Vdly__d2_small_top__DOT__conv3__DOT__controller__DOT__state = 3U;
                } else {
                    if ((7U != (IData)(vlSelfRef.d2_small_top__DOT__conv3__DOT__out_col))) {
                        vlSelfRef.d2_small_top__DOT__conv3__DOT__out_col 
                            = (0x0000ffffU & ((IData)(1U) 
                                              + (IData)(vlSelfRef.d2_small_top__DOT__conv3__DOT__out_col)));
                    } else {
                        if ((7U != (IData)(vlSelfRef.d2_small_top__DOT__conv3__DOT__out_row))) {
                            vlSelfRef.d2_small_top__DOT__conv3__DOT__out_row 
                                = (0x0000ffffU & ((IData)(1U) 
                                                  + (IData)(vlSelfRef.d2_small_top__DOT__conv3__DOT__out_row)));
                        } else {
                            vlSelfRef.d2_small_top__DOT__conv3__DOT__out_channel 
                                = (0x0000ffffU & ((IData)(1U) 
                                                  + (IData)(vlSelfRef.d2_small_top__DOT__conv3__DOT__out_channel)));
                            vlSelfRef.d2_small_top__DOT__conv3__DOT__out_row = 0U;
                        }
                        vlSelfRef.d2_small_top__DOT__conv3__DOT__out_col = 0U;
                    }
                    vlSelfRef.__Vdly__d2_small_top__DOT__conv3__DOT__controller__DOT__state = 1U;
                }
            }
        } else if ((1U & (IData)(vlSelfRef.d2_small_top__DOT__conv3__DOT__controller__DOT__state))) {
            vlSelfRef.__Vdly__d2_small_top__DOT__conv3__DOT__engine_start = 1U;
            vlSelfRef.__Vdly__d2_small_top__DOT__conv3__DOT__controller__DOT__state = 2U;
        } else {
            vlSelfRef.d2_small_top__DOT__conv3__DOT__controller_busy = 0U;
            if (vlSelfRef.conv3_start) {
                vlSelfRef.d2_small_top__DOT__conv3__DOT__out_row = 0U;
                vlSelfRef.d2_small_top__DOT__conv3__DOT__out_col = 0U;
                vlSelfRef.d2_small_top__DOT__conv3__DOT__out_channel = 0U;
                vlSelfRef.d2_small_top__DOT__conv3__DOT__controller_busy = 1U;
                vlSelfRef.__Vdly__d2_small_top__DOT__conv3__DOT__controller__DOT__state = 1U;
            }
        }
        vlSelfRef.d2_small_top__DOT__conv1__DOT__engine_done = 0U;
        if ((2U & (IData)(vlSelfRef.d2_small_top__DOT__conv1__DOT__engine__DOT__state))) {
            if ((1U & (IData)(vlSelfRef.d2_small_top__DOT__conv1__DOT__engine__DOT__state))) {
                vlSelfRef.d2_small_top__DOT__conv1__DOT__engine_done = 1U;
                vlSelfRef.__Vdly__d2_small_top__DOT__conv1__DOT__engine__DOT__state = 0U;
            } else {
                if ((1U & (~ (IData)(vlSelfRef.d2_small_top__DOT__conv1__DOT__engine__DOT__last_item)))) {
                    if ((2U != (IData)(vlSelfRef.d2_small_top__DOT__conv1__DOT__engine__DOT__kc))) {
                        vlSelfRef.d2_small_top__DOT__conv1__DOT__engine__DOT__kc 
                            = (3U & ((IData)(1U) + (IData)(vlSelfRef.d2_small_top__DOT__conv1__DOT__engine__DOT__kc)));
                    } else {
                        if ((2U != (IData)(vlSelfRef.d2_small_top__DOT__conv1__DOT__engine__DOT__kr))) {
                            vlSelfRef.d2_small_top__DOT__conv1__DOT__engine__DOT__kr 
                                = (3U & ((IData)(1U) 
                                         + (IData)(vlSelfRef.d2_small_top__DOT__conv1__DOT__engine__DOT__kr)));
                        } else {
                            vlSelfRef.__Vdly__d2_small_top__DOT__conv1__DOT__engine__DOT__ic 
                                = (0x0000ffffU & ((IData)(1U) 
                                                  + (IData)(vlSelfRef.d2_small_top__DOT__conv1__DOT__engine__DOT__ic)));
                            vlSelfRef.d2_small_top__DOT__conv1__DOT__engine__DOT__kr = 0U;
                        }
                        vlSelfRef.d2_small_top__DOT__conv1__DOT__engine__DOT__kc = 0U;
                    }
                }
                vlSelfRef.__Vdly__d2_small_top__DOT__conv1__DOT__engine__DOT__product_reg 
                    = (0x0000ffffU & (VL_MULS_III(16, 
                                                  (0x0000ffffU 
                                                   & VL_EXTENDS_II(16,8, (IData)(vlSelfRef.d2_small_top__DOT__buf_a_rdata))), 
                                                  (0x0000ffffU 
                                                   & VL_EXTENDS_II(16,8, (IData)(vlSelfRef.d2_small_top__DOT__conv1_weight_data)))) 
                                      & (- (IData)((IData)(vlSelfRef.d2_small_top__DOT__conv1__DOT__engine__DOT__valid_pixel)))));
                vlSelfRef.__Vdly__d2_small_top__DOT__conv1__DOT__engine__DOT__product_valid = 1U;
                vlSelfRef.__Vdly__d2_small_top__DOT__conv1__DOT__engine__DOT__product_last 
                    = vlSelfRef.d2_small_top__DOT__conv1__DOT__engine__DOT__last_item;
                vlSelfRef.__Vdly__d2_small_top__DOT__conv1__DOT__engine__DOT__state = 1U;
            }
        } else if ((1U & (IData)(vlSelfRef.d2_small_top__DOT__conv1__DOT__engine__DOT__state))) {
            if (vlSelfRef.d2_small_top__DOT__conv1__DOT__engine__DOT__product_valid) {
                vlSelfRef.d2_small_top__DOT__conv1__DOT__raw_sum 
                    = (vlSelfRef.d2_small_top__DOT__conv1__DOT__raw_sum 
                       + (((- (IData)((1U & ((IData)(vlSelfRef.d2_small_top__DOT__conv1__DOT__engine__DOT__product_reg) 
                                             >> 0x0fU)))) 
                           << 0x00000010U) | (IData)(vlSelfRef.d2_small_top__DOT__conv1__DOT__engine__DOT__product_reg)));
                if (vlSelfRef.d2_small_top__DOT__conv1__DOT__engine__DOT__product_last) {
                    vlSelfRef.__Vdly__d2_small_top__DOT__conv1__DOT__engine__DOT__product_valid = 0U;
                    vlSelfRef.__Vdly__d2_small_top__DOT__conv1__DOT__engine__DOT__state = 3U;
                } else {
                    vlSelfRef.__Vdly__d2_small_top__DOT__conv1__DOT__engine__DOT__state = 2U;
                }
            } else {
                vlSelfRef.__Vdly__d2_small_top__DOT__conv1__DOT__engine__DOT__state = 2U;
            }
        } else if (vlSelfRef.d2_small_top__DOT__conv1__DOT__engine_start) {
            vlSelfRef.__Vdly__d2_small_top__DOT__conv1__DOT__engine__DOT__ic = 0U;
            vlSelfRef.d2_small_top__DOT__conv1__DOT__engine__DOT__kr = 0U;
            vlSelfRef.d2_small_top__DOT__conv1__DOT__engine__DOT__kc = 0U;
            vlSelfRef.d2_small_top__DOT__conv1__DOT__raw_sum = 0U;
            vlSelfRef.__Vdly__d2_small_top__DOT__conv1__DOT__engine__DOT__product_valid = 0U;
            vlSelfRef.__Vdly__d2_small_top__DOT__conv1__DOT__engine__DOT__product_last = 0U;
            vlSelfRef.__Vdly__d2_small_top__DOT__conv1__DOT__engine__DOT__state = 1U;
        }
        vlSelfRef.d2_small_top__DOT__conv2__DOT__engine_done = 0U;
        if ((2U & (IData)(vlSelfRef.d2_small_top__DOT__conv2__DOT__engine__DOT__state))) {
            if ((1U & (IData)(vlSelfRef.d2_small_top__DOT__conv2__DOT__engine__DOT__state))) {
                vlSelfRef.d2_small_top__DOT__conv2__DOT__engine_done = 1U;
                vlSelfRef.__Vdly__d2_small_top__DOT__conv2__DOT__engine__DOT__state = 0U;
            } else {
                if ((1U & (~ (IData)(vlSelfRef.d2_small_top__DOT__conv2__DOT__engine__DOT__last_item)))) {
                    if ((2U != (IData)(vlSelfRef.d2_small_top__DOT__conv2__DOT__engine__DOT__kc))) {
                        vlSelfRef.d2_small_top__DOT__conv2__DOT__engine__DOT__kc 
                            = (3U & ((IData)(1U) + (IData)(vlSelfRef.d2_small_top__DOT__conv2__DOT__engine__DOT__kc)));
                    } else {
                        if ((2U != (IData)(vlSelfRef.d2_small_top__DOT__conv2__DOT__engine__DOT__kr))) {
                            vlSelfRef.d2_small_top__DOT__conv2__DOT__engine__DOT__kr 
                                = (3U & ((IData)(1U) 
                                         + (IData)(vlSelfRef.d2_small_top__DOT__conv2__DOT__engine__DOT__kr)));
                        } else {
                            vlSelfRef.__Vdly__d2_small_top__DOT__conv2__DOT__engine__DOT__ic 
                                = (0x0000ffffU & ((IData)(1U) 
                                                  + (IData)(vlSelfRef.d2_small_top__DOT__conv2__DOT__engine__DOT__ic)));
                            vlSelfRef.d2_small_top__DOT__conv2__DOT__engine__DOT__kr = 0U;
                        }
                        vlSelfRef.d2_small_top__DOT__conv2__DOT__engine__DOT__kc = 0U;
                    }
                }
                vlSelfRef.__Vdly__d2_small_top__DOT__conv2__DOT__engine__DOT__product_reg 
                    = (0x0000ffffU & (VL_MULS_III(16, 
                                                  (0x0000ffffU 
                                                   & VL_EXTENDS_II(16,8, (IData)(vlSelfRef.d2_small_top__DOT__buf_a_rdata))), 
                                                  (0x0000ffffU 
                                                   & VL_EXTENDS_II(16,8, (IData)(vlSelfRef.d2_small_top__DOT__conv2_weight_data)))) 
                                      & (- (IData)((IData)(vlSelfRef.d2_small_top__DOT__conv2__DOT__engine__DOT__valid_pixel)))));
                vlSelfRef.__Vdly__d2_small_top__DOT__conv2__DOT__engine__DOT__product_valid = 1U;
                vlSelfRef.__Vdly__d2_small_top__DOT__conv2__DOT__engine__DOT__product_last 
                    = vlSelfRef.d2_small_top__DOT__conv2__DOT__engine__DOT__last_item;
                vlSelfRef.__Vdly__d2_small_top__DOT__conv2__DOT__engine__DOT__state = 1U;
            }
        } else if ((1U & (IData)(vlSelfRef.d2_small_top__DOT__conv2__DOT__engine__DOT__state))) {
            if (vlSelfRef.d2_small_top__DOT__conv2__DOT__engine__DOT__product_valid) {
                vlSelfRef.d2_small_top__DOT__conv2__DOT__raw_sum 
                    = (vlSelfRef.d2_small_top__DOT__conv2__DOT__raw_sum 
                       + (((- (IData)((1U & ((IData)(vlSelfRef.d2_small_top__DOT__conv2__DOT__engine__DOT__product_reg) 
                                             >> 0x0fU)))) 
                           << 0x00000010U) | (IData)(vlSelfRef.d2_small_top__DOT__conv2__DOT__engine__DOT__product_reg)));
                if (vlSelfRef.d2_small_top__DOT__conv2__DOT__engine__DOT__product_last) {
                    vlSelfRef.__Vdly__d2_small_top__DOT__conv2__DOT__engine__DOT__product_valid = 0U;
                    vlSelfRef.__Vdly__d2_small_top__DOT__conv2__DOT__engine__DOT__state = 3U;
                } else {
                    vlSelfRef.__Vdly__d2_small_top__DOT__conv2__DOT__engine__DOT__state = 2U;
                }
            } else {
                vlSelfRef.__Vdly__d2_small_top__DOT__conv2__DOT__engine__DOT__state = 2U;
            }
        } else if (vlSelfRef.d2_small_top__DOT__conv2__DOT__engine_start) {
            vlSelfRef.__Vdly__d2_small_top__DOT__conv2__DOT__engine__DOT__ic = 0U;
            vlSelfRef.d2_small_top__DOT__conv2__DOT__engine__DOT__kr = 0U;
            vlSelfRef.d2_small_top__DOT__conv2__DOT__engine__DOT__kc = 0U;
            vlSelfRef.d2_small_top__DOT__conv2__DOT__raw_sum = 0U;
            vlSelfRef.__Vdly__d2_small_top__DOT__conv2__DOT__engine__DOT__product_valid = 0U;
            vlSelfRef.__Vdly__d2_small_top__DOT__conv2__DOT__engine__DOT__product_last = 0U;
            vlSelfRef.__Vdly__d2_small_top__DOT__conv2__DOT__engine__DOT__state = 1U;
        }
        vlSelfRef.d2_small_top__DOT__conv3__DOT__engine_done = 0U;
        if ((2U & (IData)(vlSelfRef.d2_small_top__DOT__conv3__DOT__engine__DOT__state))) {
            if ((1U & (IData)(vlSelfRef.d2_small_top__DOT__conv3__DOT__engine__DOT__state))) {
                vlSelfRef.d2_small_top__DOT__conv3__DOT__engine_done = 1U;
                vlSelfRef.__Vdly__d2_small_top__DOT__conv3__DOT__engine__DOT__state = 0U;
            } else {
                if ((1U & (~ (IData)(vlSelfRef.d2_small_top__DOT__conv3__DOT__engine__DOT__last_item)))) {
                    if ((2U != (IData)(vlSelfRef.d2_small_top__DOT__conv3__DOT__engine__DOT__kc))) {
                        vlSelfRef.d2_small_top__DOT__conv3__DOT__engine__DOT__kc 
                            = (3U & ((IData)(1U) + (IData)(vlSelfRef.d2_small_top__DOT__conv3__DOT__engine__DOT__kc)));
                    } else {
                        if ((2U != (IData)(vlSelfRef.d2_small_top__DOT__conv3__DOT__engine__DOT__kr))) {
                            vlSelfRef.d2_small_top__DOT__conv3__DOT__engine__DOT__kr 
                                = (3U & ((IData)(1U) 
                                         + (IData)(vlSelfRef.d2_small_top__DOT__conv3__DOT__engine__DOT__kr)));
                        } else {
                            vlSelfRef.__Vdly__d2_small_top__DOT__conv3__DOT__engine__DOT__ic 
                                = (0x0000ffffU & ((IData)(1U) 
                                                  + (IData)(vlSelfRef.d2_small_top__DOT__conv3__DOT__engine__DOT__ic)));
                            vlSelfRef.d2_small_top__DOT__conv3__DOT__engine__DOT__kr = 0U;
                        }
                        vlSelfRef.d2_small_top__DOT__conv3__DOT__engine__DOT__kc = 0U;
                    }
                }
                vlSelfRef.__Vdly__d2_small_top__DOT__conv3__DOT__engine__DOT__product_reg 
                    = (0x0000ffffU & (VL_MULS_III(16, 
                                                  (0x0000ffffU 
                                                   & VL_EXTENDS_II(16,8, (IData)(vlSelfRef.d2_small_top__DOT__buf_a_rdata))), 
                                                  (0x0000ffffU 
                                                   & VL_EXTENDS_II(16,8, (IData)(vlSelfRef.d2_small_top__DOT__conv3_weight_data)))) 
                                      & (- (IData)((IData)(vlSelfRef.d2_small_top__DOT__conv3__DOT__engine__DOT__valid_pixel)))));
                vlSelfRef.__Vdly__d2_small_top__DOT__conv3__DOT__engine__DOT__product_valid = 1U;
                vlSelfRef.__Vdly__d2_small_top__DOT__conv3__DOT__engine__DOT__product_last 
                    = vlSelfRef.d2_small_top__DOT__conv3__DOT__engine__DOT__last_item;
                vlSelfRef.__Vdly__d2_small_top__DOT__conv3__DOT__engine__DOT__state = 1U;
            }
        } else if ((1U & (IData)(vlSelfRef.d2_small_top__DOT__conv3__DOT__engine__DOT__state))) {
            if (vlSelfRef.d2_small_top__DOT__conv3__DOT__engine__DOT__product_valid) {
                vlSelfRef.d2_small_top__DOT__conv3__DOT__raw_sum 
                    = (vlSelfRef.d2_small_top__DOT__conv3__DOT__raw_sum 
                       + (((- (IData)((1U & ((IData)(vlSelfRef.d2_small_top__DOT__conv3__DOT__engine__DOT__product_reg) 
                                             >> 0x0fU)))) 
                           << 0x00000010U) | (IData)(vlSelfRef.d2_small_top__DOT__conv3__DOT__engine__DOT__product_reg)));
                if (vlSelfRef.d2_small_top__DOT__conv3__DOT__engine__DOT__product_last) {
                    vlSelfRef.__Vdly__d2_small_top__DOT__conv3__DOT__engine__DOT__product_valid = 0U;
                    vlSelfRef.__Vdly__d2_small_top__DOT__conv3__DOT__engine__DOT__state = 3U;
                } else {
                    vlSelfRef.__Vdly__d2_small_top__DOT__conv3__DOT__engine__DOT__state = 2U;
                }
            } else {
                vlSelfRef.__Vdly__d2_small_top__DOT__conv3__DOT__engine__DOT__state = 2U;
            }
        } else if (vlSelfRef.d2_small_top__DOT__conv3__DOT__engine_start) {
            vlSelfRef.__Vdly__d2_small_top__DOT__conv3__DOT__engine__DOT__ic = 0U;
            vlSelfRef.d2_small_top__DOT__conv3__DOT__engine__DOT__kr = 0U;
            vlSelfRef.d2_small_top__DOT__conv3__DOT__engine__DOT__kc = 0U;
            vlSelfRef.d2_small_top__DOT__conv3__DOT__raw_sum = 0U;
            vlSelfRef.__Vdly__d2_small_top__DOT__conv3__DOT__engine__DOT__product_valid = 0U;
            vlSelfRef.__Vdly__d2_small_top__DOT__conv3__DOT__engine__DOT__product_last = 0U;
            vlSelfRef.__Vdly__d2_small_top__DOT__conv3__DOT__engine__DOT__state = 1U;
        }
    } else {
        vlSelfRef.d2_small_top__DOT__conv1__DOT__biased_sum_reg = 0U;
        vlSelfRef.__Vdly__d2_small_top__DOT__conv1__DOT__controller__DOT__state = 0U;
        vlSelfRef.d2_small_top__DOT__conv1__DOT__out_row = 0U;
        vlSelfRef.d2_small_top__DOT__conv1__DOT__out_col = 0U;
        vlSelfRef.d2_small_top__DOT__conv1__DOT__out_channel = 0U;
        vlSelfRef.__Vdly__d2_small_top__DOT__conv1__DOT__engine_start = 0U;
        vlSelfRef.d2_small_top__DOT__conv1__DOT__controller_out_addr = 0U;
        vlSelfRef.d2_small_top__DOT__conv1__DOT__controller_busy = 0U;
        vlSelfRef.d2_small_top__DOT__conv1__DOT__controller_done = 0U;
        vlSelfRef.d2_small_top__DOT__conv2__DOT__biased_sum_reg = 0U;
        vlSelfRef.__Vdly__d2_small_top__DOT__conv2__DOT__controller__DOT__state = 0U;
        vlSelfRef.d2_small_top__DOT__conv2__DOT__out_row = 0U;
        vlSelfRef.d2_small_top__DOT__conv2__DOT__out_col = 0U;
        vlSelfRef.d2_small_top__DOT__conv2__DOT__out_channel = 0U;
        vlSelfRef.__Vdly__d2_small_top__DOT__conv2__DOT__engine_start = 0U;
        vlSelfRef.d2_small_top__DOT__conv2__DOT__controller_out_addr = 0U;
        vlSelfRef.d2_small_top__DOT__conv2__DOT__controller_busy = 0U;
        vlSelfRef.d2_small_top__DOT__conv2__DOT__controller_done = 0U;
        vlSelfRef.d2_small_top__DOT__conv3__DOT__biased_sum_reg = 0U;
        vlSelfRef.__Vdly__d2_small_top__DOT__conv3__DOT__controller__DOT__state = 0U;
        vlSelfRef.d2_small_top__DOT__conv3__DOT__out_row = 0U;
        vlSelfRef.d2_small_top__DOT__conv3__DOT__out_col = 0U;
        vlSelfRef.d2_small_top__DOT__conv3__DOT__out_channel = 0U;
        vlSelfRef.__Vdly__d2_small_top__DOT__conv3__DOT__engine_start = 0U;
        vlSelfRef.d2_small_top__DOT__conv3__DOT__controller_out_addr = 0U;
        vlSelfRef.d2_small_top__DOT__conv3__DOT__controller_busy = 0U;
        vlSelfRef.d2_small_top__DOT__conv3__DOT__controller_done = 0U;
        vlSelfRef.__Vdly__d2_small_top__DOT__conv1__DOT__engine__DOT__ic = 0U;
        vlSelfRef.d2_small_top__DOT__conv1__DOT__engine__DOT__kr = 0U;
        vlSelfRef.d2_small_top__DOT__conv1__DOT__engine__DOT__kc = 0U;
        vlSelfRef.d2_small_top__DOT__conv1__DOT__raw_sum = 0U;
        vlSelfRef.__Vdly__d2_small_top__DOT__conv1__DOT__engine__DOT__state = 0U;
        vlSelfRef.__Vdly__d2_small_top__DOT__conv1__DOT__engine__DOT__product_reg = 0U;
        vlSelfRef.__Vdly__d2_small_top__DOT__conv1__DOT__engine__DOT__product_valid = 0U;
        vlSelfRef.__Vdly__d2_small_top__DOT__conv1__DOT__engine__DOT__product_last = 0U;
        vlSelfRef.d2_small_top__DOT__conv1__DOT__engine_done = 0U;
        vlSelfRef.__Vdly__d2_small_top__DOT__conv2__DOT__engine__DOT__ic = 0U;
        vlSelfRef.d2_small_top__DOT__conv2__DOT__engine__DOT__kr = 0U;
        vlSelfRef.d2_small_top__DOT__conv2__DOT__engine__DOT__kc = 0U;
        vlSelfRef.d2_small_top__DOT__conv2__DOT__raw_sum = 0U;
        vlSelfRef.__Vdly__d2_small_top__DOT__conv2__DOT__engine__DOT__state = 0U;
        vlSelfRef.__Vdly__d2_small_top__DOT__conv2__DOT__engine__DOT__product_reg = 0U;
        vlSelfRef.__Vdly__d2_small_top__DOT__conv2__DOT__engine__DOT__product_valid = 0U;
        vlSelfRef.__Vdly__d2_small_top__DOT__conv2__DOT__engine__DOT__product_last = 0U;
        vlSelfRef.d2_small_top__DOT__conv2__DOT__engine_done = 0U;
        vlSelfRef.__Vdly__d2_small_top__DOT__conv3__DOT__engine__DOT__ic = 0U;
        vlSelfRef.d2_small_top__DOT__conv3__DOT__engine__DOT__kr = 0U;
        vlSelfRef.d2_small_top__DOT__conv3__DOT__engine__DOT__kc = 0U;
        vlSelfRef.d2_small_top__DOT__conv3__DOT__raw_sum = 0U;
        vlSelfRef.__Vdly__d2_small_top__DOT__conv3__DOT__engine__DOT__state = 0U;
        vlSelfRef.__Vdly__d2_small_top__DOT__conv3__DOT__engine__DOT__product_reg = 0U;
        vlSelfRef.__Vdly__d2_small_top__DOT__conv3__DOT__engine__DOT__product_valid = 0U;
        vlSelfRef.__Vdly__d2_small_top__DOT__conv3__DOT__engine__DOT__product_last = 0U;
        vlSelfRef.d2_small_top__DOT__conv3__DOT__engine_done = 0U;
    }
    d2_small_top__DOT__conv1__DOT__quant__DOT__result43 
        = (0x000007ffffffffffULL & ((1U & (IData)((vlSelfRef.d2_small_top__DOT__conv1__DOT__product_reg 
                                                   >> 0x0000002aU)))
                                     ? (- d2_small_top__DOT__conv1__DOT__quant__DOT__gen_round_shift__DOT__shifted_magnitude)
                                     : d2_small_top__DOT__conv1__DOT__quant__DOT__gen_round_shift__DOT__shifted_magnitude));
    vlSelfRef.conv1_start = vlSelfRef.__Vdly__conv1_start;
    vlSelfRef.d2_small_top__DOT__conv1__DOT__controller__DOT__state 
        = vlSelfRef.__Vdly__d2_small_top__DOT__conv1__DOT__controller__DOT__state;
    d2_small_top__DOT__conv2__DOT__quant__DOT__result43 
        = (0x000007ffffffffffULL & ((1U & (IData)((vlSelfRef.d2_small_top__DOT__conv2__DOT__product_reg 
                                                   >> 0x0000002aU)))
                                     ? (- d2_small_top__DOT__conv2__DOT__quant__DOT__gen_round_shift__DOT__shifted_magnitude)
                                     : d2_small_top__DOT__conv2__DOT__quant__DOT__gen_round_shift__DOT__shifted_magnitude));
    vlSelfRef.conv2_start = vlSelfRef.__Vdly__conv2_start;
    vlSelfRef.d2_small_top__DOT__conv2__DOT__controller__DOT__state 
        = vlSelfRef.__Vdly__d2_small_top__DOT__conv2__DOT__controller__DOT__state;
    d2_small_top__DOT__conv3__DOT__quant__DOT__result43 
        = (0x000007ffffffffffULL & ((1U & (IData)((vlSelfRef.d2_small_top__DOT__conv3__DOT__product_reg 
                                                   >> 0x0000002aU)))
                                     ? (- d2_small_top__DOT__conv3__DOT__quant__DOT__gen_round_shift__DOT__shifted_magnitude)
                                     : d2_small_top__DOT__conv3__DOT__quant__DOT__gen_round_shift__DOT__shifted_magnitude));
    vlSelfRef.conv3_start = vlSelfRef.__Vdly__conv3_start;
    vlSelfRef.d2_small_top__DOT__conv3__DOT__controller__DOT__state 
        = vlSelfRef.__Vdly__d2_small_top__DOT__conv3__DOT__controller__DOT__state;
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
    vlSelfRef.d2_small_top__DOT__conv1__DOT__engine_start 
        = vlSelfRef.__Vdly__d2_small_top__DOT__conv1__DOT__engine_start;
    vlSelfRef.d2_small_top__DOT__conv1__DOT__engine__DOT__state 
        = vlSelfRef.__Vdly__d2_small_top__DOT__conv1__DOT__engine__DOT__state;
    vlSelfRef.d2_small_top__DOT__conv1__DOT__engine__DOT__product_reg 
        = vlSelfRef.__Vdly__d2_small_top__DOT__conv1__DOT__engine__DOT__product_reg;
    vlSelfRef.d2_small_top__DOT__conv1__DOT__engine__DOT__product_valid 
        = vlSelfRef.__Vdly__d2_small_top__DOT__conv1__DOT__engine__DOT__product_valid;
    vlSelfRef.d2_small_top__DOT__conv1__DOT__engine__DOT__product_last 
        = vlSelfRef.__Vdly__d2_small_top__DOT__conv1__DOT__engine__DOT__product_last;
    vlSelfRef.d2_small_top__DOT__conv2__DOT__engine_start 
        = vlSelfRef.__Vdly__d2_small_top__DOT__conv2__DOT__engine_start;
    vlSelfRef.d2_small_top__DOT__conv2__DOT__engine__DOT__state 
        = vlSelfRef.__Vdly__d2_small_top__DOT__conv2__DOT__engine__DOT__state;
    vlSelfRef.d2_small_top__DOT__conv2__DOT__engine__DOT__product_reg 
        = vlSelfRef.__Vdly__d2_small_top__DOT__conv2__DOT__engine__DOT__product_reg;
    vlSelfRef.d2_small_top__DOT__conv2__DOT__engine__DOT__product_valid 
        = vlSelfRef.__Vdly__d2_small_top__DOT__conv2__DOT__engine__DOT__product_valid;
    vlSelfRef.d2_small_top__DOT__conv2__DOT__engine__DOT__product_last 
        = vlSelfRef.__Vdly__d2_small_top__DOT__conv2__DOT__engine__DOT__product_last;
    vlSelfRef.d2_small_top__DOT__conv3__DOT__engine_start 
        = vlSelfRef.__Vdly__d2_small_top__DOT__conv3__DOT__engine_start;
    vlSelfRef.d2_small_top__DOT__conv3__DOT__engine__DOT__state 
        = vlSelfRef.__Vdly__d2_small_top__DOT__conv3__DOT__engine__DOT__state;
    vlSelfRef.d2_small_top__DOT__conv3__DOT__engine__DOT__product_reg 
        = vlSelfRef.__Vdly__d2_small_top__DOT__conv3__DOT__engine__DOT__product_reg;
    vlSelfRef.d2_small_top__DOT__conv3__DOT__engine__DOT__product_valid 
        = vlSelfRef.__Vdly__d2_small_top__DOT__conv3__DOT__engine__DOT__product_valid;
    vlSelfRef.d2_small_top__DOT__conv3__DOT__engine__DOT__product_last 
        = vlSelfRef.__Vdly__d2_small_top__DOT__conv3__DOT__engine__DOT__product_last;
}

void Vd2_small_top___024root___nba_sequent__TOP__3(Vd2_small_top___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vd2_small_top___024root___nba_sequent__TOP__3\n"); );
    Vd2_small_top__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    vlSelfRef.d2_small_top__DOT__buf_b_rdata = vlSelfRef.d2_small_top__DOT__buffer_b__DOT__mem
        [(0x00003fffU & ((2U == (IData)(vlSelfRef.state_dbg))
                          ? (VL_SHIFTL_III(14,32,32, (IData)(vlSelfRef.d2_small_top__DOT__pool1__DOT__ch), 0x0000000aU) 
                             + ((0x0000ffffU & (VL_SHIFTL_III(16,16,32, (IData)(vlSelfRef.d2_small_top__DOT__pool1__DOT__col), 1U) 
                                                + (1U 
                                                   & (IData)(vlSelfRef.d2_small_top__DOT__pool1__DOT__tap)))) 
                                + (0x001fffe0U & ((
                                                   VL_SHIFTL_III(16,16,32, (IData)(vlSelfRef.d2_small_top__DOT__pool1__DOT__row), 1U) 
                                                   + 
                                                   (1U 
                                                    & ((IData)(vlSelfRef.d2_small_top__DOT__pool1__DOT__tap) 
                                                       >> 1U))) 
                                                  << 5U))))
                          : ((4U == (IData)(vlSelfRef.state_dbg))
                              ? (VL_SHIFTL_III(14,32,32, (IData)(vlSelfRef.d2_small_top__DOT__pool2__DOT__ch), 8U) 
                                 + ((0x0000ffffU & 
                                     (VL_SHIFTL_III(16,16,32, (IData)(vlSelfRef.d2_small_top__DOT__pool2__DOT__col), 1U) 
                                      + (1U & (IData)(vlSelfRef.d2_small_top__DOT__pool2__DOT__tap)))) 
                                    + (0x000ffff0U 
                                       & ((VL_SHIFTL_III(16,16,32, (IData)(vlSelfRef.d2_small_top__DOT__pool2__DOT__row), 1U) 
                                           + (1U & 
                                              ((IData)(vlSelfRef.d2_small_top__DOT__pool2__DOT__tap) 
                                               >> 1U))) 
                                          << 4U))))
                              : (0x0001ffffU & ((VL_SHIFTL_III(14,32,32, (IData)(vlSelfRef.d2_small_top__DOT__gap__DOT__ch), 6U) 
                                                 + (IData)(vlSelfRef.d2_small_top__DOT__gap__DOT__idx)) 
                                                & (- (IData)(
                                                             (6U 
                                                              == (IData)(vlSelfRef.state_dbg)))))))))];
    vlSelfRef.d2_small_top__DOT__conv1_weight_data 
        = ((0x01afU >= (IData)(vlSelfRef.d2_small_top__DOT__conv1_weight_rom__DOT__rom_addr))
            ? vlSelfRef.d2_small_top__DOT__conv1_weight_rom__DOT__rom
           [vlSelfRef.d2_small_top__DOT__conv1_weight_rom__DOT__rom_addr]
            : 0U);
    vlSelfRef.d2_small_top__DOT__conv2_weight_data 
        = ((0x11ffU >= (IData)(vlSelfRef.d2_small_top__DOT__conv2_weight_rom__DOT__rom_addr))
            ? vlSelfRef.d2_small_top__DOT__conv2_weight_rom__DOT__rom
           [vlSelfRef.d2_small_top__DOT__conv2_weight_rom__DOT__rom_addr]
            : 0U);
    vlSelfRef.d2_small_top__DOT__conv3_weight_data 
        = ((0x47ffU >= (IData)(vlSelfRef.d2_small_top__DOT__conv3_weight_rom__DOT__rom_addr))
            ? vlSelfRef.d2_small_top__DOT__conv3_weight_rom__DOT__rom
           [vlSelfRef.d2_small_top__DOT__conv3_weight_rom__DOT__rom_addr]
            : 0U);
    vlSelfRef.d2_small_top__DOT__buf_a_rdata = vlSelfRef.d2_small_top__DOT__buffer_a__DOT__mem
        [(0x00001fffU & ((1U == (IData)(vlSelfRef.state_dbg))
                          ? (0x0001ffffU & ((vlSelfRef.__VdfgRegularize_h6e95ff9d_0_3 
                                             + (VL_SHIFTL_III(13,32,32, (IData)(vlSelfRef.d2_small_top__DOT__conv1__DOT__engine__DOT__ic), 0x0000000aU) 
                                                + VL_SHIFTL_III(13,32,32, vlSelfRef.__VdfgRegularize_h6e95ff9d_0_2, 5U))) 
                                            & (- (IData)((IData)(vlSelfRef.d2_small_top__DOT__conv1__DOT__engine__DOT__valid_pixel)))))
                          : ((3U == (IData)(vlSelfRef.state_dbg))
                              ? (0x0001ffffU & ((vlSelfRef.__VdfgRegularize_h6e95ff9d_0_5 
                                                 + 
                                                 (VL_SHIFTL_III(13,32,32, (IData)(vlSelfRef.d2_small_top__DOT__conv2__DOT__engine__DOT__ic), 8U) 
                                                  + 
                                                  VL_SHIFTL_III(13,32,32, vlSelfRef.__VdfgRegularize_h6e95ff9d_0_4, 4U))) 
                                                & (- (IData)((IData)(vlSelfRef.d2_small_top__DOT__conv2__DOT__engine__DOT__valid_pixel)))))
                              : ((5U == (IData)(vlSelfRef.state_dbg))
                                  ? (0x0001ffffU & 
                                     ((vlSelfRef.__VdfgRegularize_h6e95ff9d_0_7 
                                       + (VL_SHIFTL_III(13,32,32, (IData)(vlSelfRef.d2_small_top__DOT__conv3__DOT__engine__DOT__ic), 6U) 
                                          + VL_SHIFTL_III(13,32,32, vlSelfRef.__VdfgRegularize_h6e95ff9d_0_6, 3U))) 
                                      & (- (IData)((IData)(vlSelfRef.d2_small_top__DOT__conv3__DOT__engine__DOT__valid_pixel)))))
                                  : (0x0001ffffU & 
                                     ((IData)(vlSelfRef.d2_small_top__DOT__linear__DOT__feature_idx) 
                                      & (- (IData)(
                                                   (7U 
                                                    == (IData)(vlSelfRef.state_dbg))))))))))];
    if (vlSelfRef.__VdlySet__d2_small_top__DOT__buffer_b__DOT__mem__v0) {
        vlSelfRef.d2_small_top__DOT__buffer_b__DOT__mem[vlSelfRef.__VdlyDim0__d2_small_top__DOT__buffer_b__DOT__mem__v0] 
            = vlSelfRef.__VdlyVal__d2_small_top__DOT__buffer_b__DOT__mem__v0;
    }
    if (vlSelfRef.__VdlySet__d2_small_top__DOT__buffer_a__DOT__mem__v0) {
        vlSelfRef.d2_small_top__DOT__buffer_a__DOT__mem[vlSelfRef.__VdlyDim0__d2_small_top__DOT__buffer_a__DOT__mem__v0] 
            = vlSelfRef.__VdlyVal__d2_small_top__DOT__buffer_a__DOT__mem__v0;
    }
}

void Vd2_small_top___024root___nba_sequent__TOP__4(Vd2_small_top___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vd2_small_top___024root___nba_sequent__TOP__4\n"); );
    Vd2_small_top__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    vlSelfRef.d2_small_top__DOT__pool1__DOT__ch = vlSelfRef.__Vdly__d2_small_top__DOT__pool1__DOT__ch;
    vlSelfRef.d2_small_top__DOT__pool1__DOT__col = vlSelfRef.__Vdly__d2_small_top__DOT__pool1__DOT__col;
    vlSelfRef.d2_small_top__DOT__pool1__DOT__tap = vlSelfRef.__Vdly__d2_small_top__DOT__pool1__DOT__tap;
    vlSelfRef.d2_small_top__DOT__pool1__DOT__row = vlSelfRef.__Vdly__d2_small_top__DOT__pool1__DOT__row;
    vlSelfRef.d2_small_top__DOT__pool2__DOT__ch = vlSelfRef.__Vdly__d2_small_top__DOT__pool2__DOT__ch;
    vlSelfRef.d2_small_top__DOT__pool2__DOT__col = vlSelfRef.__Vdly__d2_small_top__DOT__pool2__DOT__col;
    vlSelfRef.d2_small_top__DOT__pool2__DOT__tap = vlSelfRef.__Vdly__d2_small_top__DOT__pool2__DOT__tap;
    vlSelfRef.d2_small_top__DOT__pool2__DOT__row = vlSelfRef.__Vdly__d2_small_top__DOT__pool2__DOT__row;
    vlSelfRef.d2_small_top__DOT__gap__DOT__ch = vlSelfRef.__Vdly__d2_small_top__DOT__gap__DOT__ch;
    vlSelfRef.d2_small_top__DOT__gap__DOT__idx = vlSelfRef.__Vdly__d2_small_top__DOT__gap__DOT__idx;
    vlSelfRef.state_dbg = vlSelfRef.__Vdly__state_dbg;
    vlSelfRef.__VdfgRegularize_h6e95ff9d_0_3 = (((IData)(vlSelfRef.d2_small_top__DOT__conv1__DOT__engine__DOT__kc) 
                                                 + (IData)(vlSelfRef.d2_small_top__DOT__conv1__DOT__out_col)) 
                                                - (IData)(1U));
    vlSelfRef.__VdfgRegularize_h6e95ff9d_0_2 = (((IData)(vlSelfRef.d2_small_top__DOT__conv1__DOT__engine__DOT__kr) 
                                                 + (IData)(vlSelfRef.d2_small_top__DOT__conv1__DOT__out_row)) 
                                                - (IData)(1U));
    vlSelfRef.__VdfgRegularize_h6e95ff9d_0_5 = (((IData)(vlSelfRef.d2_small_top__DOT__conv2__DOT__engine__DOT__kc) 
                                                 + (IData)(vlSelfRef.d2_small_top__DOT__conv2__DOT__out_col)) 
                                                - (IData)(1U));
    vlSelfRef.__VdfgRegularize_h6e95ff9d_0_4 = (((IData)(vlSelfRef.d2_small_top__DOT__conv2__DOT__engine__DOT__kr) 
                                                 + (IData)(vlSelfRef.d2_small_top__DOT__conv2__DOT__out_row)) 
                                                - (IData)(1U));
    vlSelfRef.__VdfgRegularize_h6e95ff9d_0_7 = (((IData)(vlSelfRef.d2_small_top__DOT__conv3__DOT__engine__DOT__kc) 
                                                 + (IData)(vlSelfRef.d2_small_top__DOT__conv3__DOT__out_col)) 
                                                - (IData)(1U));
    vlSelfRef.__VdfgRegularize_h6e95ff9d_0_6 = (((IData)(vlSelfRef.d2_small_top__DOT__conv3__DOT__engine__DOT__kr) 
                                                 + (IData)(vlSelfRef.d2_small_top__DOT__conv3__DOT__out_row)) 
                                                - (IData)(1U));
    vlSelfRef.d2_small_top__DOT__linear__DOT__feature_idx 
        = vlSelfRef.__Vdly__d2_small_top__DOT__linear__DOT__feature_idx;
    vlSelfRef.d2_small_top__DOT__conv1__DOT__engine__DOT__ic 
        = vlSelfRef.__Vdly__d2_small_top__DOT__conv1__DOT__engine__DOT__ic;
    vlSelfRef.d2_small_top__DOT__conv2__DOT__engine__DOT__ic 
        = vlSelfRef.__Vdly__d2_small_top__DOT__conv2__DOT__engine__DOT__ic;
    vlSelfRef.d2_small_top__DOT__conv3__DOT__engine__DOT__ic 
        = vlSelfRef.__Vdly__d2_small_top__DOT__conv3__DOT__engine__DOT__ic;
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
    vlSelfRef.d2_small_top__DOT__linear_weight_rom__DOT__rom_addr 
        = (0x000003ffU & ((IData)(vlSelfRef.d2_small_top__DOT__linear__DOT__feature_idx) 
                          + ((IData)(vlSelfRef.d2_small_top__DOT__linear__DOT__class_idx) 
                             << 6U)));
    vlSelfRef.d2_small_top__DOT__conv1__DOT__engine__DOT__last_item 
        = ((2U == (IData)(vlSelfRef.d2_small_top__DOT__conv1__DOT__engine__DOT__ic)) 
           & ((2U == (IData)(vlSelfRef.d2_small_top__DOT__conv1__DOT__engine__DOT__kr)) 
              & (2U == (IData)(vlSelfRef.d2_small_top__DOT__conv1__DOT__engine__DOT__kc))));
    vlSelfRef.d2_small_top__DOT__conv1_weight_rom__DOT__rom_addr 
        = (0x000001ffU & ((IData)(vlSelfRef.d2_small_top__DOT__conv1__DOT__engine__DOT__kc) 
                          + (((IData)(0x0000001bU) 
                              * (IData)(vlSelfRef.d2_small_top__DOT__conv1__DOT__out_channel)) 
                             + (((IData)(9U) * (IData)(vlSelfRef.d2_small_top__DOT__conv1__DOT__engine__DOT__ic)) 
                                + ((IData)(3U) * (IData)(vlSelfRef.d2_small_top__DOT__conv1__DOT__engine__DOT__kr))))));
    vlSelfRef.d2_small_top__DOT__conv2__DOT__engine__DOT__last_item 
        = ((0x000fU == (IData)(vlSelfRef.d2_small_top__DOT__conv2__DOT__engine__DOT__ic)) 
           & ((2U == (IData)(vlSelfRef.d2_small_top__DOT__conv2__DOT__engine__DOT__kr)) 
              & (2U == (IData)(vlSelfRef.d2_small_top__DOT__conv2__DOT__engine__DOT__kc))));
    vlSelfRef.d2_small_top__DOT__conv2_weight_rom__DOT__rom_addr 
        = (0x00001fffU & ((IData)(vlSelfRef.d2_small_top__DOT__conv2__DOT__engine__DOT__kc) 
                          + (((IData)(9U) * VL_SHIFTL_III(32,32,32, (IData)(vlSelfRef.d2_small_top__DOT__conv2__DOT__out_channel), 4U)) 
                             + (((IData)(9U) * (IData)(vlSelfRef.d2_small_top__DOT__conv2__DOT__engine__DOT__ic)) 
                                + ((IData)(3U) * (IData)(vlSelfRef.d2_small_top__DOT__conv2__DOT__engine__DOT__kr))))));
    vlSelfRef.d2_small_top__DOT__conv3__DOT__engine__DOT__last_item 
        = ((0x001fU == (IData)(vlSelfRef.d2_small_top__DOT__conv3__DOT__engine__DOT__ic)) 
           & ((2U == (IData)(vlSelfRef.d2_small_top__DOT__conv3__DOT__engine__DOT__kr)) 
              & (2U == (IData)(vlSelfRef.d2_small_top__DOT__conv3__DOT__engine__DOT__kc))));
    vlSelfRef.d2_small_top__DOT__conv3_weight_rom__DOT__rom_addr 
        = (0x00007fffU & ((IData)(vlSelfRef.d2_small_top__DOT__conv3__DOT__engine__DOT__kc) 
                          + (((IData)(9U) * VL_SHIFTL_III(32,32,32, (IData)(vlSelfRef.d2_small_top__DOT__conv3__DOT__out_channel), 5U)) 
                             + (((IData)(9U) * (IData)(vlSelfRef.d2_small_top__DOT__conv3__DOT__engine__DOT__ic)) 
                                + ((IData)(3U) * (IData)(vlSelfRef.d2_small_top__DOT__conv3__DOT__engine__DOT__kr))))));
}

void Vd2_small_top___024root___eval_nba(Vd2_small_top___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vd2_small_top___024root___eval_nba\n"); );
    Vd2_small_top__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    if ((3ULL & vlSelfRef.__VnbaTriggered[0U])) {
        Vd2_small_top___024root___nba_sequent__TOP__0(vlSelf);
    }
    if ((1ULL & vlSelfRef.__VnbaTriggered[0U])) {
        vlSelfRef.__VdlySet__d2_small_top__DOT__buffer_a__DOT__mem__v0 = 0U;
        vlSelfRef.__VdlySet__d2_small_top__DOT__buffer_b__DOT__mem__v0 = 0U;
        if (((IData)(vlSelfRef.__VdfgRegularize_h6e95ff9d_0_0) 
             | ((IData)(vlSelfRef.d2_small_top__DOT__pool1_out_we) 
                | ((IData)(vlSelfRef.d2_small_top__DOT__pool2_out_we) 
                   | (IData)(vlSelfRef.d2_small_top__DOT__gap_out_we))))) {
            if (vlSelfRef.__VdfgRegularize_h6e95ff9d_0_0) {
                vlSelfRef.__VdlyVal__d2_small_top__DOT__buffer_a__DOT__mem__v0 
                    = vlSelfRef.input_data;
                vlSelfRef.__VdlyDim0__d2_small_top__DOT__buffer_a__DOT__mem__v0 
                    = (0x00001fffU & vlSelfRef.input_addr);
            } else if (vlSelfRef.d2_small_top__DOT__pool1_out_we) {
                vlSelfRef.__VdlyVal__d2_small_top__DOT__buffer_a__DOT__mem__v0 
                    = vlSelfRef.d2_small_top__DOT__pool1_out_data;
                vlSelfRef.__VdlyDim0__d2_small_top__DOT__buffer_a__DOT__mem__v0 
                    = (0x00001fffU & vlSelfRef.d2_small_top__DOT__pool1_out_addr);
            } else if (vlSelfRef.d2_small_top__DOT__pool2_out_we) {
                vlSelfRef.__VdlyVal__d2_small_top__DOT__buffer_a__DOT__mem__v0 
                    = vlSelfRef.d2_small_top__DOT__pool2_out_data;
                vlSelfRef.__VdlyDim0__d2_small_top__DOT__buffer_a__DOT__mem__v0 
                    = (0x00001fffU & vlSelfRef.d2_small_top__DOT__pool2_out_addr);
            } else {
                vlSelfRef.__VdlyVal__d2_small_top__DOT__buffer_a__DOT__mem__v0 
                    = vlSelfRef.d2_small_top__DOT__gap_out_data;
                vlSelfRef.__VdlyDim0__d2_small_top__DOT__buffer_a__DOT__mem__v0 
                    = (0x00001fffU & vlSelfRef.d2_small_top__DOT__gap_out_addr);
            }
            vlSelfRef.__VdlySet__d2_small_top__DOT__buffer_a__DOT__mem__v0 = 1U;
        }
        if (((IData)(vlSelfRef.d2_small_top__DOT__conv1__DOT__postprocess_valid) 
             | ((IData)(vlSelfRef.d2_small_top__DOT__conv2__DOT__postprocess_valid) 
                | (IData)(vlSelfRef.d2_small_top__DOT__conv3__DOT__postprocess_valid)))) {
            if (vlSelfRef.d2_small_top__DOT__conv1__DOT__postprocess_valid) {
                vlSelfRef.__VdlyVal__d2_small_top__DOT__buffer_b__DOT__mem__v0 
                    = vlSelfRef.d2_small_top__DOT__conv1__DOT__out_data_reg;
                vlSelfRef.__VdlyDim0__d2_small_top__DOT__buffer_b__DOT__mem__v0 
                    = (0x00003fffU & vlSelfRef.d2_small_top__DOT__conv1__DOT__out_addr_reg);
            } else if (vlSelfRef.d2_small_top__DOT__conv2__DOT__postprocess_valid) {
                vlSelfRef.__VdlyVal__d2_small_top__DOT__buffer_b__DOT__mem__v0 
                    = vlSelfRef.d2_small_top__DOT__conv2__DOT__out_data_reg;
                vlSelfRef.__VdlyDim0__d2_small_top__DOT__buffer_b__DOT__mem__v0 
                    = (0x00003fffU & vlSelfRef.d2_small_top__DOT__conv2__DOT__out_addr_reg);
            } else {
                vlSelfRef.__VdlyVal__d2_small_top__DOT__buffer_b__DOT__mem__v0 
                    = vlSelfRef.d2_small_top__DOT__conv3__DOT__out_data_reg;
                vlSelfRef.__VdlyDim0__d2_small_top__DOT__buffer_b__DOT__mem__v0 
                    = (0x00003fffU & vlSelfRef.d2_small_top__DOT__conv3__DOT__out_addr_reg);
            }
            vlSelfRef.__VdlySet__d2_small_top__DOT__buffer_b__DOT__mem__v0 = 1U;
        }
        vlSelfRef.d2_small_top__DOT__linear_weight_data 
            = ((0x027fU >= (IData)(vlSelfRef.d2_small_top__DOT__linear_weight_rom__DOT__rom_addr))
                ? vlSelfRef.d2_small_top__DOT__linear_weight_rom__DOT__rom
               [vlSelfRef.d2_small_top__DOT__linear_weight_rom__DOT__rom_addr]
                : 0U);
    }
    if ((3ULL & vlSelfRef.__VnbaTriggered[0U])) {
        Vd2_small_top___024root___nba_sequent__TOP__2(vlSelf);
    }
    if ((1ULL & vlSelfRef.__VnbaTriggered[0U])) {
        vlSelfRef.d2_small_top__DOT__buf_b_rdata = vlSelfRef.d2_small_top__DOT__buffer_b__DOT__mem
            [(0x00003fffU & ((2U == (IData)(vlSelfRef.state_dbg))
                              ? (VL_SHIFTL_III(14,32,32, (IData)(vlSelfRef.d2_small_top__DOT__pool1__DOT__ch), 0x0000000aU) 
                                 + ((0x0000ffffU & 
                                     (VL_SHIFTL_III(16,16,32, (IData)(vlSelfRef.d2_small_top__DOT__pool1__DOT__col), 1U) 
                                      + (1U & (IData)(vlSelfRef.d2_small_top__DOT__pool1__DOT__tap)))) 
                                    + (0x001fffe0U 
                                       & ((VL_SHIFTL_III(16,16,32, (IData)(vlSelfRef.d2_small_top__DOT__pool1__DOT__row), 1U) 
                                           + (1U & 
                                              ((IData)(vlSelfRef.d2_small_top__DOT__pool1__DOT__tap) 
                                               >> 1U))) 
                                          << 5U))))
                              : ((4U == (IData)(vlSelfRef.state_dbg))
                                  ? (VL_SHIFTL_III(14,32,32, (IData)(vlSelfRef.d2_small_top__DOT__pool2__DOT__ch), 8U) 
                                     + ((0x0000ffffU 
                                         & (VL_SHIFTL_III(16,16,32, (IData)(vlSelfRef.d2_small_top__DOT__pool2__DOT__col), 1U) 
                                            + (1U & (IData)(vlSelfRef.d2_small_top__DOT__pool2__DOT__tap)))) 
                                        + (0x000ffff0U 
                                           & ((VL_SHIFTL_III(16,16,32, (IData)(vlSelfRef.d2_small_top__DOT__pool2__DOT__row), 1U) 
                                               + (1U 
                                                  & ((IData)(vlSelfRef.d2_small_top__DOT__pool2__DOT__tap) 
                                                     >> 1U))) 
                                              << 4U))))
                                  : (0x0001ffffU & 
                                     ((VL_SHIFTL_III(14,32,32, (IData)(vlSelfRef.d2_small_top__DOT__gap__DOT__ch), 6U) 
                                       + (IData)(vlSelfRef.d2_small_top__DOT__gap__DOT__idx)) 
                                      & (- (IData)(
                                                   (6U 
                                                    == (IData)(vlSelfRef.state_dbg)))))))))];
        vlSelfRef.d2_small_top__DOT__conv1_weight_data 
            = ((0x01afU >= (IData)(vlSelfRef.d2_small_top__DOT__conv1_weight_rom__DOT__rom_addr))
                ? vlSelfRef.d2_small_top__DOT__conv1_weight_rom__DOT__rom
               [vlSelfRef.d2_small_top__DOT__conv1_weight_rom__DOT__rom_addr]
                : 0U);
        vlSelfRef.d2_small_top__DOT__conv2_weight_data 
            = ((0x11ffU >= (IData)(vlSelfRef.d2_small_top__DOT__conv2_weight_rom__DOT__rom_addr))
                ? vlSelfRef.d2_small_top__DOT__conv2_weight_rom__DOT__rom
               [vlSelfRef.d2_small_top__DOT__conv2_weight_rom__DOT__rom_addr]
                : 0U);
        vlSelfRef.d2_small_top__DOT__conv3_weight_data 
            = ((0x47ffU >= (IData)(vlSelfRef.d2_small_top__DOT__conv3_weight_rom__DOT__rom_addr))
                ? vlSelfRef.d2_small_top__DOT__conv3_weight_rom__DOT__rom
               [vlSelfRef.d2_small_top__DOT__conv3_weight_rom__DOT__rom_addr]
                : 0U);
        vlSelfRef.d2_small_top__DOT__buf_a_rdata = vlSelfRef.d2_small_top__DOT__buffer_a__DOT__mem
            [(0x00001fffU & ((1U == (IData)(vlSelfRef.state_dbg))
                              ? (0x0001ffffU & ((vlSelfRef.__VdfgRegularize_h6e95ff9d_0_3 
                                                 + 
                                                 (VL_SHIFTL_III(13,32,32, (IData)(vlSelfRef.d2_small_top__DOT__conv1__DOT__engine__DOT__ic), 0x0000000aU) 
                                                  + 
                                                  VL_SHIFTL_III(13,32,32, vlSelfRef.__VdfgRegularize_h6e95ff9d_0_2, 5U))) 
                                                & (- (IData)((IData)(vlSelfRef.d2_small_top__DOT__conv1__DOT__engine__DOT__valid_pixel)))))
                              : ((3U == (IData)(vlSelfRef.state_dbg))
                                  ? (0x0001ffffU & 
                                     ((vlSelfRef.__VdfgRegularize_h6e95ff9d_0_5 
                                       + (VL_SHIFTL_III(13,32,32, (IData)(vlSelfRef.d2_small_top__DOT__conv2__DOT__engine__DOT__ic), 8U) 
                                          + VL_SHIFTL_III(13,32,32, vlSelfRef.__VdfgRegularize_h6e95ff9d_0_4, 4U))) 
                                      & (- (IData)((IData)(vlSelfRef.d2_small_top__DOT__conv2__DOT__engine__DOT__valid_pixel)))))
                                  : ((5U == (IData)(vlSelfRef.state_dbg))
                                      ? (0x0001ffffU 
                                         & ((vlSelfRef.__VdfgRegularize_h6e95ff9d_0_7 
                                             + (VL_SHIFTL_III(13,32,32, (IData)(vlSelfRef.d2_small_top__DOT__conv3__DOT__engine__DOT__ic), 6U) 
                                                + VL_SHIFTL_III(13,32,32, vlSelfRef.__VdfgRegularize_h6e95ff9d_0_6, 3U))) 
                                            & (- (IData)((IData)(vlSelfRef.d2_small_top__DOT__conv3__DOT__engine__DOT__valid_pixel)))))
                                      : (0x0001ffffU 
                                         & ((IData)(vlSelfRef.d2_small_top__DOT__linear__DOT__feature_idx) 
                                            & (- (IData)(
                                                         (7U 
                                                          == (IData)(vlSelfRef.state_dbg))))))))))];
        if (vlSelfRef.__VdlySet__d2_small_top__DOT__buffer_b__DOT__mem__v0) {
            vlSelfRef.d2_small_top__DOT__buffer_b__DOT__mem[vlSelfRef.__VdlyDim0__d2_small_top__DOT__buffer_b__DOT__mem__v0] 
                = vlSelfRef.__VdlyVal__d2_small_top__DOT__buffer_b__DOT__mem__v0;
        }
        if (vlSelfRef.__VdlySet__d2_small_top__DOT__buffer_a__DOT__mem__v0) {
            vlSelfRef.d2_small_top__DOT__buffer_a__DOT__mem[vlSelfRef.__VdlyDim0__d2_small_top__DOT__buffer_a__DOT__mem__v0] 
                = vlSelfRef.__VdlyVal__d2_small_top__DOT__buffer_a__DOT__mem__v0;
        }
    }
    if ((3ULL & vlSelfRef.__VnbaTriggered[0U])) {
        Vd2_small_top___024root___nba_sequent__TOP__4(vlSelf);
    }
}

void Vd2_small_top___024root___trigger_orInto__act_vec_vec(VlUnpacked<QData/*63:0*/, 1> &out, const VlUnpacked<QData/*63:0*/, 1> &in) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vd2_small_top___024root___trigger_orInto__act_vec_vec\n"); );
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
VL_ATTR_COLD void Vd2_small_top___024root___dump_triggers__act(const VlUnpacked<QData/*63:0*/, 1> &triggers, const std::string &tag);
#endif  // VL_DEBUG

bool Vd2_small_top___024root___eval_phase__act(Vd2_small_top___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vd2_small_top___024root___eval_phase__act\n"); );
    Vd2_small_top__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    Vd2_small_top___024root___eval_triggers_vec__act(vlSelf);
#ifdef VL_DEBUG
    if (VL_UNLIKELY(vlSymsp->_vm_contextp__->debug())) {
        Vd2_small_top___024root___dump_triggers__act(vlSelfRef.__VactTriggered, "act"s);
    }
#endif
    Vd2_small_top___024root___trigger_orInto__act_vec_vec(vlSelfRef.__VnbaTriggered, vlSelfRef.__VactTriggered);
    return (0U);
}

void Vd2_small_top___024root___trigger_clear__act(VlUnpacked<QData/*63:0*/, 1> &out) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vd2_small_top___024root___trigger_clear__act\n"); );
    // Locals
    IData/*31:0*/ n;
    // Body
    n = 0U;
    do {
        out[n] = 0ULL;
        n = ((IData)(1U) + n);
    } while ((1U > n));
}

bool Vd2_small_top___024root___eval_phase__nba(Vd2_small_top___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vd2_small_top___024root___eval_phase__nba\n"); );
    Vd2_small_top__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Locals
    CData/*0:0*/ __VnbaExecute;
    // Body
    __VnbaExecute = Vd2_small_top___024root___trigger_anySet__act(vlSelfRef.__VnbaTriggered);
    if (__VnbaExecute) {
        Vd2_small_top___024root___eval_nba(vlSelf);
        Vd2_small_top___024root___trigger_clear__act(vlSelfRef.__VnbaTriggered);
    }
    return (__VnbaExecute);
}

void Vd2_small_top___024root___eval(Vd2_small_top___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vd2_small_top___024root___eval\n"); );
    Vd2_small_top__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Locals
    IData/*31:0*/ __VicoIterCount;
    IData/*31:0*/ __VnbaIterCount;
    // Body
    __VicoIterCount = 0U;
    vlSelfRef.__VicoFirstIteration = 1U;
    do {
        if (VL_UNLIKELY(((0x00002710U < __VicoIterCount)))) {
#ifdef VL_DEBUG
            Vd2_small_top___024root___dump_triggers__ico(vlSelfRef.__VicoTriggered, "ico"s);
#endif
            VL_FATAL_MT("rtl/src/d2_small_top.v", 21, "", "DIDNOTCONVERGE: Input combinational region did not converge after '--converge-limit' of 10000 tries");
        }
        __VicoIterCount = ((IData)(1U) + __VicoIterCount);
        vlSelfRef.__VicoPhaseResult = Vd2_small_top___024root___eval_phase__ico(vlSelf);
        vlSelfRef.__VicoFirstIteration = 0U;
    } while (vlSelfRef.__VicoPhaseResult);
    __VnbaIterCount = 0U;
    do {
        if (VL_UNLIKELY(((0x00002710U < __VnbaIterCount)))) {
#ifdef VL_DEBUG
            Vd2_small_top___024root___dump_triggers__act(vlSelfRef.__VnbaTriggered, "nba"s);
#endif
            VL_FATAL_MT("rtl/src/d2_small_top.v", 21, "", "DIDNOTCONVERGE: NBA region did not converge after '--converge-limit' of 10000 tries");
        }
        __VnbaIterCount = ((IData)(1U) + __VnbaIterCount);
        vlSelfRef.__VactIterCount = 0U;
        do {
            if (VL_UNLIKELY(((0x00002710U < vlSelfRef.__VactIterCount)))) {
#ifdef VL_DEBUG
                Vd2_small_top___024root___dump_triggers__act(vlSelfRef.__VactTriggered, "act"s);
#endif
                VL_FATAL_MT("rtl/src/d2_small_top.v", 21, "", "DIDNOTCONVERGE: Active region did not converge after '--converge-limit' of 10000 tries");
            }
            vlSelfRef.__VactIterCount = ((IData)(1U) 
                                         + vlSelfRef.__VactIterCount);
            vlSelfRef.__VactPhaseResult = Vd2_small_top___024root___eval_phase__act(vlSelf);
        } while (vlSelfRef.__VactPhaseResult);
        vlSelfRef.__VnbaPhaseResult = Vd2_small_top___024root___eval_phase__nba(vlSelf);
    } while (vlSelfRef.__VnbaPhaseResult);
}

#ifdef VL_DEBUG
void Vd2_small_top___024root___eval_debug_assertions(Vd2_small_top___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vd2_small_top___024root___eval_debug_assertions\n"); );
    Vd2_small_top__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    if (VL_UNLIKELY(((vlSelfRef.clk & 0xfeU)))) {
        Verilated::overWidthError("clk");
    }
    if (VL_UNLIKELY(((vlSelfRef.rst_n & 0xfeU)))) {
        Verilated::overWidthError("rst_n");
    }
    if (VL_UNLIKELY(((vlSelfRef.start & 0xfeU)))) {
        Verilated::overWidthError("start");
    }
    if (VL_UNLIKELY(((vlSelfRef.input_we & 0xfeU)))) {
        Verilated::overWidthError("input_we");
    }
    if (VL_UNLIKELY(((vlSelfRef.input_addr & 0xfffe0000U)))) {
        Verilated::overWidthError("input_addr");
    }
}
#endif  // VL_DEBUG
