// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design implementation internals
// See Vedge_cnn_top.h for the primary calling header

#include "Vedge_cnn_top__pch.h"

void Vedge_cnn_top___024root___eval_triggers_vec__ico(Vedge_cnn_top___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vedge_cnn_top___024root___eval_triggers_vec__ico\n"); );
    Vedge_cnn_top__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    vlSelfRef.__VicoTriggered[0U] = ((0xfffffffffffffffeULL 
                                      & vlSelfRef.__VicoTriggered[0U]) 
                                     | (IData)((IData)(vlSelfRef.__VicoFirstIteration)));
}

bool Vedge_cnn_top___024root___trigger_anySet__ico(const VlUnpacked<QData/*63:0*/, 1> &in) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vedge_cnn_top___024root___trigger_anySet__ico\n"); );
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

void Vedge_cnn_top___024root___ico_sequent__TOP__0(Vedge_cnn_top___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vedge_cnn_top___024root___ico_sequent__TOP__0\n"); );
    Vedge_cnn_top__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
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
}

void Vedge_cnn_top___024root___eval_ico(Vedge_cnn_top___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vedge_cnn_top___024root___eval_ico\n"); );
    Vedge_cnn_top__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    if ((1ULL & vlSelfRef.__VicoTriggered[0U])) {
        vlSelfRef.__VdfgRegularize_h6e95ff9d_0_0 = 
            ((~ (IData)(vlSelfRef.busy)) & ((~ ((IData)(vlSelfRef.edge_cnn_top__DOT__conv1__DOT__postprocess_pending) 
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
    }
}

#ifdef VL_DEBUG
VL_ATTR_COLD void Vedge_cnn_top___024root___dump_triggers__ico(const VlUnpacked<QData/*63:0*/, 1> &triggers, const std::string &tag);
#endif  // VL_DEBUG

bool Vedge_cnn_top___024root___eval_phase__ico(Vedge_cnn_top___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vedge_cnn_top___024root___eval_phase__ico\n"); );
    Vedge_cnn_top__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Locals
    CData/*0:0*/ __VicoExecute;
    // Body
    Vedge_cnn_top___024root___eval_triggers_vec__ico(vlSelf);
#ifdef VL_DEBUG
    if (VL_UNLIKELY(vlSymsp->_vm_contextp__->debug())) {
        Vedge_cnn_top___024root___dump_triggers__ico(vlSelfRef.__VicoTriggered, "ico"s);
    }
#endif
    __VicoExecute = Vedge_cnn_top___024root___trigger_anySet__ico(vlSelfRef.__VicoTriggered);
    if (__VicoExecute) {
        Vedge_cnn_top___024root___eval_ico(vlSelf);
    }
    return (__VicoExecute);
}

void Vedge_cnn_top___024root___eval_triggers_vec__act(Vedge_cnn_top___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vedge_cnn_top___024root___eval_triggers_vec__act\n"); );
    Vedge_cnn_top__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
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

bool Vedge_cnn_top___024root___trigger_anySet__act(const VlUnpacked<QData/*63:0*/, 1> &in) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vedge_cnn_top___024root___trigger_anySet__act\n"); );
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

void Vedge_cnn_top___024root___nba_sequent__TOP__0(Vedge_cnn_top___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vedge_cnn_top___024root___nba_sequent__TOP__0\n"); );
    Vedge_cnn_top__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Locals
    IData/*31:0*/ __Vdly__edge_cnn_top__DOT__logit0;
    __Vdly__edge_cnn_top__DOT__logit0 = 0;
    CData/*0:0*/ __Vdly__linear_start;
    __Vdly__linear_start = 0;
    CData/*3:0*/ __Vdly__edge_cnn_top__DOT__fsm__DOT__state;
    __Vdly__edge_cnn_top__DOT__fsm__DOT__state = 0;
    CData/*2:0*/ __Vdly__edge_cnn_top__DOT__linear__DOT__state;
    __Vdly__edge_cnn_top__DOT__linear__DOT__state = 0;
    IData/*31:0*/ __Vdly__edge_cnn_top__DOT__linear__DOT__acc;
    __Vdly__edge_cnn_top__DOT__linear__DOT__acc = 0;
    // Body
    vlSelfRef.__Vdly__edge_cnn_top__DOT__pool1__DOT__state 
        = vlSelfRef.edge_cnn_top__DOT__pool1__DOT__state;
    vlSelfRef.__Vdly__edge_cnn_top__DOT__pool1__DOT__tap 
        = vlSelfRef.edge_cnn_top__DOT__pool1__DOT__tap;
    vlSelfRef.__Vdly__edge_cnn_top__DOT__pool1__DOT__col 
        = vlSelfRef.edge_cnn_top__DOT__pool1__DOT__col;
    vlSelfRef.__Vdly__edge_cnn_top__DOT__pool1__DOT__row 
        = vlSelfRef.edge_cnn_top__DOT__pool1__DOT__row;
    vlSelfRef.__Vdly__edge_cnn_top__DOT__pool1__DOT__ch 
        = vlSelfRef.edge_cnn_top__DOT__pool1__DOT__ch;
    vlSelfRef.__Vdly__edge_cnn_top__DOT__pool1__DOT__current_max 
        = vlSelfRef.edge_cnn_top__DOT__pool1__DOT__current_max;
    vlSelfRef.__Vdly__edge_cnn_top__DOT__pool2__DOT__state 
        = vlSelfRef.edge_cnn_top__DOT__pool2__DOT__state;
    vlSelfRef.__Vdly__edge_cnn_top__DOT__pool2__DOT__tap 
        = vlSelfRef.edge_cnn_top__DOT__pool2__DOT__tap;
    vlSelfRef.__Vdly__edge_cnn_top__DOT__pool2__DOT__col 
        = vlSelfRef.edge_cnn_top__DOT__pool2__DOT__col;
    vlSelfRef.__Vdly__edge_cnn_top__DOT__pool2__DOT__row 
        = vlSelfRef.edge_cnn_top__DOT__pool2__DOT__row;
    vlSelfRef.__Vdly__edge_cnn_top__DOT__pool2__DOT__ch 
        = vlSelfRef.edge_cnn_top__DOT__pool2__DOT__ch;
    vlSelfRef.__Vdly__edge_cnn_top__DOT__pool2__DOT__current_max 
        = vlSelfRef.edge_cnn_top__DOT__pool2__DOT__current_max;
    vlSelfRef.__Vdly__edge_cnn_top__DOT__gap__DOT__state 
        = vlSelfRef.edge_cnn_top__DOT__gap__DOT__state;
    vlSelfRef.__Vdly__edge_cnn_top__DOT__gap__DOT__idx 
        = vlSelfRef.edge_cnn_top__DOT__gap__DOT__idx;
    vlSelfRef.__Vdly__edge_cnn_top__DOT__gap__DOT__ch 
        = vlSelfRef.edge_cnn_top__DOT__gap__DOT__ch;
    vlSelfRef.__Vdly__edge_cnn_top__DOT__gap__DOT__sum 
        = vlSelfRef.edge_cnn_top__DOT__gap__DOT__sum;
    __Vdly__edge_cnn_top__DOT__linear__DOT__state = vlSelfRef.edge_cnn_top__DOT__linear__DOT__state;
    __Vdly__edge_cnn_top__DOT__linear__DOT__acc = vlSelfRef.edge_cnn_top__DOT__linear__DOT__acc;
    vlSelfRef.__Vdly__edge_cnn_top__DOT__linear__DOT__feature_idx 
        = vlSelfRef.edge_cnn_top__DOT__linear__DOT__feature_idx;
    vlSelfRef.__Vdly__edge_cnn_top__DOT__conv4__DOT__engine__DOT__state 
        = vlSelfRef.edge_cnn_top__DOT__conv4__DOT__engine__DOT__state;
    vlSelfRef.__Vdly__edge_cnn_top__DOT__conv4__DOT__engine__DOT__product_reg 
        = vlSelfRef.edge_cnn_top__DOT__conv4__DOT__engine__DOT__product_reg;
    vlSelfRef.__Vdly__edge_cnn_top__DOT__conv4__DOT__engine__DOT__product_valid 
        = vlSelfRef.edge_cnn_top__DOT__conv4__DOT__engine__DOT__product_valid;
    vlSelfRef.__Vdly__edge_cnn_top__DOT__conv4__DOT__engine__DOT__product_last 
        = vlSelfRef.edge_cnn_top__DOT__conv4__DOT__engine__DOT__product_last;
    vlSelfRef.__Vdly__edge_cnn_top__DOT__conv4__DOT__engine__DOT__ic 
        = vlSelfRef.edge_cnn_top__DOT__conv4__DOT__engine__DOT__ic;
    vlSelfRef.__Vdly__edge_cnn_top__DOT__conv1__DOT__engine__DOT__state 
        = vlSelfRef.edge_cnn_top__DOT__conv1__DOT__engine__DOT__state;
    vlSelfRef.__Vdly__edge_cnn_top__DOT__conv1__DOT__engine__DOT__product_reg 
        = vlSelfRef.edge_cnn_top__DOT__conv1__DOT__engine__DOT__product_reg;
    vlSelfRef.__Vdly__edge_cnn_top__DOT__conv1__DOT__engine__DOT__product_valid 
        = vlSelfRef.edge_cnn_top__DOT__conv1__DOT__engine__DOT__product_valid;
    vlSelfRef.__Vdly__edge_cnn_top__DOT__conv1__DOT__engine__DOT__product_last 
        = vlSelfRef.edge_cnn_top__DOT__conv1__DOT__engine__DOT__product_last;
    vlSelfRef.__Vdly__edge_cnn_top__DOT__conv2__DOT__engine__DOT__state 
        = vlSelfRef.edge_cnn_top__DOT__conv2__DOT__engine__DOT__state;
    vlSelfRef.__Vdly__edge_cnn_top__DOT__conv2__DOT__engine__DOT__product_reg 
        = vlSelfRef.edge_cnn_top__DOT__conv2__DOT__engine__DOT__product_reg;
    vlSelfRef.__Vdly__edge_cnn_top__DOT__conv2__DOT__engine__DOT__product_valid 
        = vlSelfRef.edge_cnn_top__DOT__conv2__DOT__engine__DOT__product_valid;
    vlSelfRef.__Vdly__edge_cnn_top__DOT__conv2__DOT__engine__DOT__product_last 
        = vlSelfRef.edge_cnn_top__DOT__conv2__DOT__engine__DOT__product_last;
    vlSelfRef.__Vdly__edge_cnn_top__DOT__conv3__DOT__engine__DOT__state 
        = vlSelfRef.edge_cnn_top__DOT__conv3__DOT__engine__DOT__state;
    vlSelfRef.__Vdly__edge_cnn_top__DOT__conv3__DOT__engine__DOT__product_reg 
        = vlSelfRef.edge_cnn_top__DOT__conv3__DOT__engine__DOT__product_reg;
    vlSelfRef.__Vdly__edge_cnn_top__DOT__conv3__DOT__engine__DOT__product_valid 
        = vlSelfRef.edge_cnn_top__DOT__conv3__DOT__engine__DOT__product_valid;
    vlSelfRef.__Vdly__edge_cnn_top__DOT__conv3__DOT__engine__DOT__product_last 
        = vlSelfRef.edge_cnn_top__DOT__conv3__DOT__engine__DOT__product_last;
    vlSelfRef.__Vdly__edge_cnn_top__DOT__conv1__DOT__engine__DOT__ic 
        = vlSelfRef.edge_cnn_top__DOT__conv1__DOT__engine__DOT__ic;
    vlSelfRef.__Vdly__edge_cnn_top__DOT__conv2__DOT__engine__DOT__ic 
        = vlSelfRef.edge_cnn_top__DOT__conv2__DOT__engine__DOT__ic;
    vlSelfRef.__Vdly__edge_cnn_top__DOT__conv3__DOT__engine__DOT__ic 
        = vlSelfRef.edge_cnn_top__DOT__conv3__DOT__engine__DOT__ic;
    vlSelfRef.__Vdly__edge_cnn_top__DOT__conv4__DOT__engine_start 
        = vlSelfRef.edge_cnn_top__DOT__conv4__DOT__engine_start;
    vlSelfRef.__Vdly__edge_cnn_top__DOT__conv4__DOT__controller__DOT__state 
        = vlSelfRef.edge_cnn_top__DOT__conv4__DOT__controller__DOT__state;
    vlSelfRef.__Vdly__edge_cnn_top__DOT__conv1__DOT__engine_start 
        = vlSelfRef.edge_cnn_top__DOT__conv1__DOT__engine_start;
    vlSelfRef.__Vdly__edge_cnn_top__DOT__conv1__DOT__controller__DOT__state 
        = vlSelfRef.edge_cnn_top__DOT__conv1__DOT__controller__DOT__state;
    vlSelfRef.__Vdly__edge_cnn_top__DOT__conv2__DOT__engine_start 
        = vlSelfRef.edge_cnn_top__DOT__conv2__DOT__engine_start;
    vlSelfRef.__Vdly__edge_cnn_top__DOT__conv2__DOT__controller__DOT__state 
        = vlSelfRef.edge_cnn_top__DOT__conv2__DOT__controller__DOT__state;
    vlSelfRef.__Vdly__edge_cnn_top__DOT__conv3__DOT__engine_start 
        = vlSelfRef.edge_cnn_top__DOT__conv3__DOT__engine_start;
    vlSelfRef.__Vdly__edge_cnn_top__DOT__conv3__DOT__controller__DOT__state 
        = vlSelfRef.edge_cnn_top__DOT__conv3__DOT__controller__DOT__state;
    __Vdly__edge_cnn_top__DOT__logit0 = vlSelfRef.edge_cnn_top__DOT__logit0;
    vlSelfRef.__Vdly__conv1_start = vlSelfRef.conv1_start;
    vlSelfRef.__Vdly__pool1_start = vlSelfRef.pool1_start;
    vlSelfRef.__Vdly__conv2_start = vlSelfRef.conv2_start;
    vlSelfRef.__Vdly__pool2_start = vlSelfRef.pool2_start;
    vlSelfRef.__Vdly__conv3_start = vlSelfRef.conv3_start;
    vlSelfRef.__Vdly__conv4_start = vlSelfRef.conv4_start;
    vlSelfRef.__Vdly__gap_start = vlSelfRef.gap_start;
    __Vdly__linear_start = vlSelfRef.linear_start;
    vlSelfRef.__Vdly__state_dbg = vlSelfRef.state_dbg;
    __Vdly__edge_cnn_top__DOT__fsm__DOT__state = vlSelfRef.edge_cnn_top__DOT__fsm__DOT__state;
    if (vlSelfRef.rst_n) {
        if (vlSelfRef.edge_cnn_top__DOT__linear_logit_valid) {
            if ((8U & (IData)(vlSelfRef.edge_cnn_top__DOT__linear_logit_index))) {
                if ((4U & (IData)(vlSelfRef.edge_cnn_top__DOT__linear_logit_index))) {
                    __Vdly__edge_cnn_top__DOT__logit0 
                        = vlSelfRef.edge_cnn_top__DOT__logit0;
                } else if ((2U & (IData)(vlSelfRef.edge_cnn_top__DOT__linear_logit_index))) {
                    __Vdly__edge_cnn_top__DOT__logit0 
                        = vlSelfRef.edge_cnn_top__DOT__logit0;
                }
                if ((1U & (~ ((IData)(vlSelfRef.edge_cnn_top__DOT__linear_logit_index) 
                              >> 2U)))) {
                    if ((1U & (~ ((IData)(vlSelfRef.edge_cnn_top__DOT__linear_logit_index) 
                                  >> 1U)))) {
                        if ((1U & (~ (IData)(vlSelfRef.edge_cnn_top__DOT__linear_logit_index)))) {
                            vlSelfRef.edge_cnn_top__DOT__logit8 
                                = vlSelfRef.edge_cnn_top__DOT__linear_logit_data;
                        }
                        if ((1U & (IData)(vlSelfRef.edge_cnn_top__DOT__linear_logit_index))) {
                            vlSelfRef.edge_cnn_top__DOT__logit9 
                                = vlSelfRef.edge_cnn_top__DOT__linear_logit_data;
                        }
                    }
                }
            } else if ((1U & (~ ((IData)(vlSelfRef.edge_cnn_top__DOT__linear_logit_index) 
                                 >> 2U)))) {
                if ((1U & (~ ((IData)(vlSelfRef.edge_cnn_top__DOT__linear_logit_index) 
                              >> 1U)))) {
                    if ((1U & (~ (IData)(vlSelfRef.edge_cnn_top__DOT__linear_logit_index)))) {
                        __Vdly__edge_cnn_top__DOT__logit0 
                            = vlSelfRef.edge_cnn_top__DOT__linear_logit_data;
                    }
                }
            }
            if ((1U & (~ ((IData)(vlSelfRef.edge_cnn_top__DOT__linear_logit_index) 
                          >> 3U)))) {
                if ((1U & (~ ((IData)(vlSelfRef.edge_cnn_top__DOT__linear_logit_index) 
                              >> 2U)))) {
                    if ((2U & (IData)(vlSelfRef.edge_cnn_top__DOT__linear_logit_index))) {
                        if ((1U & (~ (IData)(vlSelfRef.edge_cnn_top__DOT__linear_logit_index)))) {
                            vlSelfRef.edge_cnn_top__DOT__logit2 
                                = vlSelfRef.edge_cnn_top__DOT__linear_logit_data;
                        }
                        if ((1U & (IData)(vlSelfRef.edge_cnn_top__DOT__linear_logit_index))) {
                            vlSelfRef.edge_cnn_top__DOT__logit3 
                                = vlSelfRef.edge_cnn_top__DOT__linear_logit_data;
                        }
                    }
                    if ((1U & (~ ((IData)(vlSelfRef.edge_cnn_top__DOT__linear_logit_index) 
                                  >> 1U)))) {
                        if ((1U & (IData)(vlSelfRef.edge_cnn_top__DOT__linear_logit_index))) {
                            vlSelfRef.edge_cnn_top__DOT__logit1 
                                = vlSelfRef.edge_cnn_top__DOT__linear_logit_data;
                        }
                    }
                }
                if ((4U & (IData)(vlSelfRef.edge_cnn_top__DOT__linear_logit_index))) {
                    if ((2U & (IData)(vlSelfRef.edge_cnn_top__DOT__linear_logit_index))) {
                        if ((1U & (IData)(vlSelfRef.edge_cnn_top__DOT__linear_logit_index))) {
                            vlSelfRef.edge_cnn_top__DOT__logit7 
                                = vlSelfRef.edge_cnn_top__DOT__linear_logit_data;
                        }
                        if ((1U & (~ (IData)(vlSelfRef.edge_cnn_top__DOT__linear_logit_index)))) {
                            vlSelfRef.edge_cnn_top__DOT__logit6 
                                = vlSelfRef.edge_cnn_top__DOT__linear_logit_data;
                        }
                    }
                    if ((1U & (~ ((IData)(vlSelfRef.edge_cnn_top__DOT__linear_logit_index) 
                                  >> 1U)))) {
                        if ((1U & (~ (IData)(vlSelfRef.edge_cnn_top__DOT__linear_logit_index)))) {
                            vlSelfRef.edge_cnn_top__DOT__logit4 
                                = vlSelfRef.edge_cnn_top__DOT__linear_logit_data;
                        }
                        if ((1U & (IData)(vlSelfRef.edge_cnn_top__DOT__linear_logit_index))) {
                            vlSelfRef.edge_cnn_top__DOT__logit5 
                                = vlSelfRef.edge_cnn_top__DOT__linear_logit_data;
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
        vlSelfRef.__Vdly__conv4_start = 0U;
        vlSelfRef.__Vdly__gap_start = 0U;
        __Vdly__linear_start = 0U;
        vlSelfRef.argmax_valid = 0U;
        vlSelfRef.done = 0U;
        vlSelfRef.__Vdly__state_dbg = vlSelfRef.edge_cnn_top__DOT__fsm__DOT__state;
        if ((8U & (IData)(vlSelfRef.edge_cnn_top__DOT__fsm__DOT__state))) {
            if ((4U & (IData)(vlSelfRef.edge_cnn_top__DOT__fsm__DOT__state))) {
                __Vdly__edge_cnn_top__DOT__fsm__DOT__state = 0U;
            } else if ((2U & (IData)(vlSelfRef.edge_cnn_top__DOT__fsm__DOT__state))) {
                if ((1U & (IData)(vlSelfRef.edge_cnn_top__DOT__fsm__DOT__state))) {
                    __Vdly__edge_cnn_top__DOT__fsm__DOT__state = 0U;
                } else {
                    vlSelfRef.busy = 0U;
                    vlSelfRef.done = 1U;
                    __Vdly__edge_cnn_top__DOT__fsm__DOT__state = 0U;
                }
            } else if ((1U & (IData)(vlSelfRef.edge_cnn_top__DOT__fsm__DOT__state))) {
                __Vdly__edge_cnn_top__DOT__fsm__DOT__state = 0x0aU;
            } else if (vlSelfRef.edge_cnn_top__DOT__linear_done) {
                vlSelfRef.argmax_valid = 1U;
                __Vdly__edge_cnn_top__DOT__fsm__DOT__state = 9U;
            }
        } else if ((4U & (IData)(vlSelfRef.edge_cnn_top__DOT__fsm__DOT__state))) {
            if ((2U & (IData)(vlSelfRef.edge_cnn_top__DOT__fsm__DOT__state))) {
                if ((1U & (IData)(vlSelfRef.edge_cnn_top__DOT__fsm__DOT__state))) {
                    if (vlSelfRef.edge_cnn_top__DOT__gap_done) {
                        __Vdly__linear_start = 1U;
                        __Vdly__edge_cnn_top__DOT__fsm__DOT__state = 8U;
                    }
                } else if (vlSelfRef.edge_cnn_top__DOT__conv4__DOT__controller_done_d2) {
                    vlSelfRef.__Vdly__gap_start = 1U;
                    __Vdly__edge_cnn_top__DOT__fsm__DOT__state = 7U;
                }
            } else if ((1U & (IData)(vlSelfRef.edge_cnn_top__DOT__fsm__DOT__state))) {
                if (vlSelfRef.edge_cnn_top__DOT__conv3__DOT__controller_done_d2) {
                    vlSelfRef.__Vdly__conv4_start = 1U;
                    __Vdly__edge_cnn_top__DOT__fsm__DOT__state = 6U;
                }
            } else if (vlSelfRef.edge_cnn_top__DOT__pool2_done) {
                vlSelfRef.__Vdly__conv3_start = 1U;
                __Vdly__edge_cnn_top__DOT__fsm__DOT__state = 5U;
            }
        } else if ((2U & (IData)(vlSelfRef.edge_cnn_top__DOT__fsm__DOT__state))) {
            if ((1U & (IData)(vlSelfRef.edge_cnn_top__DOT__fsm__DOT__state))) {
                if (vlSelfRef.edge_cnn_top__DOT__conv2__DOT__controller_done_d2) {
                    vlSelfRef.__Vdly__pool2_start = 1U;
                    __Vdly__edge_cnn_top__DOT__fsm__DOT__state = 4U;
                }
            } else if (vlSelfRef.edge_cnn_top__DOT__pool1_done) {
                vlSelfRef.__Vdly__conv2_start = 1U;
                __Vdly__edge_cnn_top__DOT__fsm__DOT__state = 3U;
            }
        } else if ((1U & (IData)(vlSelfRef.edge_cnn_top__DOT__fsm__DOT__state))) {
            if (vlSelfRef.edge_cnn_top__DOT__conv1__DOT__controller_done_d2) {
                vlSelfRef.__Vdly__pool1_start = 1U;
                __Vdly__edge_cnn_top__DOT__fsm__DOT__state = 2U;
            }
        } else {
            vlSelfRef.busy = 0U;
            if (vlSelfRef.start) {
                vlSelfRef.busy = 1U;
                vlSelfRef.__Vdly__conv1_start = 1U;
                __Vdly__edge_cnn_top__DOT__fsm__DOT__state = 1U;
            }
        }
        vlSelfRef.edge_cnn_top__DOT__linear_logit_valid = 0U;
        vlSelfRef.edge_cnn_top__DOT__linear_done = 0U;
        if ((4U & (IData)(vlSelfRef.edge_cnn_top__DOT__linear__DOT__state))) {
            if ((2U & (IData)(vlSelfRef.edge_cnn_top__DOT__linear__DOT__state))) {
                __Vdly__edge_cnn_top__DOT__linear__DOT__state = 0U;
            } else if ((1U & (IData)(vlSelfRef.edge_cnn_top__DOT__linear__DOT__state))) {
                __Vdly__edge_cnn_top__DOT__linear__DOT__state = 0U;
            } else {
                vlSelfRef.edge_cnn_top__DOT__linear_busy = 0U;
                vlSelfRef.edge_cnn_top__DOT__linear_done = 1U;
                __Vdly__edge_cnn_top__DOT__linear__DOT__state = 0U;
            }
        } else if ((2U & (IData)(vlSelfRef.edge_cnn_top__DOT__linear__DOT__state))) {
            if ((1U & (IData)(vlSelfRef.edge_cnn_top__DOT__linear__DOT__state))) {
                vlSelfRef.edge_cnn_top__DOT__linear_logit_index 
                    = vlSelfRef.edge_cnn_top__DOT__linear__DOT__class_idx;
                vlSelfRef.edge_cnn_top__DOT__linear_logit_valid = 1U;
                vlSelfRef.edge_cnn_top__DOT__linear_logit_data 
                    = (vlSelfRef.edge_cnn_top__DOT__linear__DOT__acc 
                       + (vlSelfRef.edge_cnn_top__DOT__linear_bias_rom__DOT__rom
                          [vlSelfRef.edge_cnn_top__DOT__linear__DOT__class_idx] 
                          & (- (IData)((9U >= (IData)(vlSelfRef.edge_cnn_top__DOT__linear__DOT__class_idx))))));
                if ((9U == (IData)(vlSelfRef.edge_cnn_top__DOT__linear__DOT__class_idx))) {
                    __Vdly__edge_cnn_top__DOT__linear__DOT__state = 4U;
                } else {
                    vlSelfRef.edge_cnn_top__DOT__linear__DOT__class_idx 
                        = (0x0000000fU & ((IData)(1U) 
                                          + (IData)(vlSelfRef.edge_cnn_top__DOT__linear__DOT__class_idx)));
                    vlSelfRef.__Vdly__edge_cnn_top__DOT__linear__DOT__feature_idx = 0U;
                    __Vdly__edge_cnn_top__DOT__linear__DOT__acc = 0U;
                    __Vdly__edge_cnn_top__DOT__linear__DOT__state = 1U;
                }
            } else {
                __Vdly__edge_cnn_top__DOT__linear__DOT__acc 
                    = (vlSelfRef.edge_cnn_top__DOT__linear__DOT__acc 
                       + VL_MULS_III(32, VL_EXTENDS_II(32,8, (IData)(vlSelfRef.edge_cnn_top__DOT__buf_b_rdata)), 
                                     VL_EXTENDS_II(32,8, (IData)(vlSelfRef.edge_cnn_top__DOT__linear_weight_data))));
                if ((0xffU == (IData)(vlSelfRef.edge_cnn_top__DOT__linear__DOT__feature_idx))) {
                    __Vdly__edge_cnn_top__DOT__linear__DOT__state = 3U;
                } else {
                    vlSelfRef.__Vdly__edge_cnn_top__DOT__linear__DOT__feature_idx 
                        = (0x000000ffU & ((IData)(1U) 
                                          + (IData)(vlSelfRef.edge_cnn_top__DOT__linear__DOT__feature_idx)));
                    __Vdly__edge_cnn_top__DOT__linear__DOT__state = 1U;
                }
            }
        } else if ((1U & (IData)(vlSelfRef.edge_cnn_top__DOT__linear__DOT__state))) {
            __Vdly__edge_cnn_top__DOT__linear__DOT__state = 2U;
        } else {
            vlSelfRef.edge_cnn_top__DOT__linear_busy = 0U;
            if (vlSelfRef.linear_start) {
                vlSelfRef.edge_cnn_top__DOT__linear__DOT__class_idx = 0U;
                vlSelfRef.__Vdly__edge_cnn_top__DOT__linear__DOT__feature_idx = 0U;
                __Vdly__edge_cnn_top__DOT__linear__DOT__acc = 0U;
                vlSelfRef.edge_cnn_top__DOT__linear_busy = 1U;
                __Vdly__edge_cnn_top__DOT__linear__DOT__state = 1U;
            }
        }
    } else {
        __Vdly__edge_cnn_top__DOT__logit0 = 0U;
        vlSelfRef.edge_cnn_top__DOT__logit2 = 0U;
        vlSelfRef.edge_cnn_top__DOT__logit7 = 0U;
        vlSelfRef.edge_cnn_top__DOT__logit8 = 0U;
        vlSelfRef.edge_cnn_top__DOT__logit4 = 0U;
        vlSelfRef.edge_cnn_top__DOT__logit9 = 0U;
        vlSelfRef.edge_cnn_top__DOT__logit6 = 0U;
        vlSelfRef.edge_cnn_top__DOT__logit5 = 0U;
        vlSelfRef.edge_cnn_top__DOT__logit3 = 0U;
        vlSelfRef.edge_cnn_top__DOT__logit1 = 0U;
        __Vdly__edge_cnn_top__DOT__fsm__DOT__state = 0U;
        vlSelfRef.__Vdly__conv1_start = 0U;
        vlSelfRef.__Vdly__pool1_start = 0U;
        vlSelfRef.__Vdly__conv2_start = 0U;
        vlSelfRef.__Vdly__pool2_start = 0U;
        vlSelfRef.__Vdly__conv3_start = 0U;
        vlSelfRef.__Vdly__conv4_start = 0U;
        vlSelfRef.__Vdly__gap_start = 0U;
        __Vdly__linear_start = 0U;
        vlSelfRef.argmax_valid = 0U;
        vlSelfRef.busy = 0U;
        vlSelfRef.done = 0U;
        vlSelfRef.__Vdly__state_dbg = 0U;
        vlSelfRef.edge_cnn_top__DOT__linear__DOT__class_idx = 0U;
        __Vdly__edge_cnn_top__DOT__linear__DOT__state = 0U;
        vlSelfRef.__Vdly__edge_cnn_top__DOT__linear__DOT__feature_idx = 0U;
        __Vdly__edge_cnn_top__DOT__linear__DOT__acc = 0U;
        vlSelfRef.edge_cnn_top__DOT__linear_logit_index = 0U;
        vlSelfRef.edge_cnn_top__DOT__linear_logit_data = 0U;
        vlSelfRef.edge_cnn_top__DOT__linear_logit_valid = 0U;
        vlSelfRef.edge_cnn_top__DOT__linear_busy = 0U;
        vlSelfRef.edge_cnn_top__DOT__linear_done = 0U;
    }
    vlSelfRef.edge_cnn_top__DOT__logit0 = __Vdly__edge_cnn_top__DOT__logit0;
    vlSelfRef.edge_cnn_top__DOT__fsm__DOT__state = __Vdly__edge_cnn_top__DOT__fsm__DOT__state;
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
    vlSelfRef.edge_cnn_top__DOT__conv4__DOT__controller_done_d2 
        = ((IData)(vlSelfRef.rst_n) && (IData)(vlSelfRef.edge_cnn_top__DOT__conv4__DOT__controller_done_d1));
    vlSelfRef.edge_cnn_top__DOT__conv2__DOT__controller_done_d2 
        = ((IData)(vlSelfRef.rst_n) && (IData)(vlSelfRef.edge_cnn_top__DOT__conv2__DOT__controller_done_d1));
    vlSelfRef.edge_cnn_top__DOT__conv1__DOT__controller_done_d2 
        = ((IData)(vlSelfRef.rst_n) && (IData)(vlSelfRef.edge_cnn_top__DOT__conv1__DOT__controller_done_d1));
    vlSelfRef.edge_cnn_top__DOT__conv3__DOT__controller_done_d2 
        = ((IData)(vlSelfRef.rst_n) && (IData)(vlSelfRef.edge_cnn_top__DOT__conv3__DOT__controller_done_d1));
    vlSelfRef.linear_start = __Vdly__linear_start;
    vlSelfRef.edge_cnn_top__DOT__linear__DOT__state 
        = __Vdly__edge_cnn_top__DOT__linear__DOT__state;
    vlSelfRef.edge_cnn_top__DOT__linear__DOT__acc = __Vdly__edge_cnn_top__DOT__linear__DOT__acc;
    vlSelfRef.edge_cnn_top__DOT__conv4__DOT__controller_done_d1 
        = ((IData)(vlSelfRef.rst_n) && (IData)(vlSelfRef.edge_cnn_top__DOT__conv4__DOT__controller_done));
    vlSelfRef.edge_cnn_top__DOT__conv2__DOT__controller_done_d1 
        = ((IData)(vlSelfRef.rst_n) && (IData)(vlSelfRef.edge_cnn_top__DOT__conv2__DOT__controller_done));
    vlSelfRef.edge_cnn_top__DOT__conv1__DOT__controller_done_d1 
        = ((IData)(vlSelfRef.rst_n) && (IData)(vlSelfRef.edge_cnn_top__DOT__conv1__DOT__controller_done));
    vlSelfRef.edge_cnn_top__DOT__conv3__DOT__controller_done_d1 
        = ((IData)(vlSelfRef.rst_n) && (IData)(vlSelfRef.edge_cnn_top__DOT__conv3__DOT__controller_done));
}

void Vedge_cnn_top___024root___nba_sequent__TOP__1(Vedge_cnn_top___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vedge_cnn_top___024root___nba_sequent__TOP__1\n"); );
    Vedge_cnn_top__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    vlSelfRef.__VdlySet__edge_cnn_top__DOT__buffer_a__DOT__mem__v0 = 0U;
    vlSelfRef.__VdlySet__edge_cnn_top__DOT__buffer_b__DOT__mem__v0 = 0U;
    if (((IData)(vlSelfRef.__VdfgRegularize_h6e95ff9d_0_0) 
         | ((IData)(vlSelfRef.edge_cnn_top__DOT__pool1_out_we) 
            | ((IData)(vlSelfRef.edge_cnn_top__DOT__pool2_out_we) 
               | (IData)(vlSelfRef.edge_cnn_top__DOT__conv4__DOT__postprocess_valid))))) {
        if (vlSelfRef.__VdfgRegularize_h6e95ff9d_0_0) {
            vlSelfRef.__VdlyVal__edge_cnn_top__DOT__buffer_a__DOT__mem__v0 
                = vlSelfRef.input_data;
            vlSelfRef.__VdlyDim0__edge_cnn_top__DOT__buffer_a__DOT__mem__v0 
                = (0x00007fffU & vlSelfRef.input_addr);
        } else if (vlSelfRef.edge_cnn_top__DOT__pool1_out_we) {
            vlSelfRef.__VdlyVal__edge_cnn_top__DOT__buffer_a__DOT__mem__v0 
                = vlSelfRef.edge_cnn_top__DOT__pool1_out_data;
            vlSelfRef.__VdlyDim0__edge_cnn_top__DOT__buffer_a__DOT__mem__v0 
                = (0x00007fffU & vlSelfRef.edge_cnn_top__DOT__pool1_out_addr);
        } else if (vlSelfRef.edge_cnn_top__DOT__pool2_out_we) {
            vlSelfRef.__VdlyVal__edge_cnn_top__DOT__buffer_a__DOT__mem__v0 
                = vlSelfRef.edge_cnn_top__DOT__pool2_out_data;
            vlSelfRef.__VdlyDim0__edge_cnn_top__DOT__buffer_a__DOT__mem__v0 
                = (0x00007fffU & vlSelfRef.edge_cnn_top__DOT__pool2_out_addr);
        } else {
            vlSelfRef.__VdlyVal__edge_cnn_top__DOT__buffer_a__DOT__mem__v0 
                = vlSelfRef.edge_cnn_top__DOT__conv4__DOT__out_data_reg;
            vlSelfRef.__VdlyDim0__edge_cnn_top__DOT__buffer_a__DOT__mem__v0 
                = (0x00007fffU & vlSelfRef.edge_cnn_top__DOT__conv4__DOT__out_addr_reg);
        }
        vlSelfRef.__VdlySet__edge_cnn_top__DOT__buffer_a__DOT__mem__v0 = 1U;
    }
    if (((IData)(vlSelfRef.edge_cnn_top__DOT__conv1__DOT__postprocess_valid) 
         | ((IData)(vlSelfRef.edge_cnn_top__DOT__conv2__DOT__postprocess_valid) 
            | ((IData)(vlSelfRef.edge_cnn_top__DOT__gap_out_we) 
               | (IData)(vlSelfRef.edge_cnn_top__DOT__conv3__DOT__postprocess_valid))))) {
        if (vlSelfRef.edge_cnn_top__DOT__conv1__DOT__postprocess_valid) {
            vlSelfRef.__VdlyVal__edge_cnn_top__DOT__buffer_b__DOT__mem__v0 
                = vlSelfRef.edge_cnn_top__DOT__conv1__DOT__out_data_reg;
            vlSelfRef.__VdlyDim0__edge_cnn_top__DOT__buffer_b__DOT__mem__v0 
                = (0x00007fffU & vlSelfRef.edge_cnn_top__DOT__conv1__DOT__out_addr_reg);
        } else if (vlSelfRef.edge_cnn_top__DOT__conv2__DOT__postprocess_valid) {
            vlSelfRef.__VdlyVal__edge_cnn_top__DOT__buffer_b__DOT__mem__v0 
                = vlSelfRef.edge_cnn_top__DOT__conv2__DOT__out_data_reg;
            vlSelfRef.__VdlyDim0__edge_cnn_top__DOT__buffer_b__DOT__mem__v0 
                = (0x00007fffU & vlSelfRef.edge_cnn_top__DOT__conv2__DOT__out_addr_reg);
        } else if (vlSelfRef.edge_cnn_top__DOT__conv3__DOT__postprocess_valid) {
            vlSelfRef.__VdlyVal__edge_cnn_top__DOT__buffer_b__DOT__mem__v0 
                = vlSelfRef.edge_cnn_top__DOT__conv3__DOT__out_data_reg;
            vlSelfRef.__VdlyDim0__edge_cnn_top__DOT__buffer_b__DOT__mem__v0 
                = (0x00007fffU & vlSelfRef.edge_cnn_top__DOT__conv3__DOT__out_addr_reg);
        } else {
            vlSelfRef.__VdlyVal__edge_cnn_top__DOT__buffer_b__DOT__mem__v0 
                = vlSelfRef.edge_cnn_top__DOT__gap_out_data;
            vlSelfRef.__VdlyDim0__edge_cnn_top__DOT__buffer_b__DOT__mem__v0 
                = (0x00007fffU & vlSelfRef.edge_cnn_top__DOT__gap_out_addr);
        }
        vlSelfRef.__VdlySet__edge_cnn_top__DOT__buffer_b__DOT__mem__v0 = 1U;
    }
    vlSelfRef.edge_cnn_top__DOT__linear_weight_data 
        = ((0x09ffU >= (IData)(vlSelfRef.edge_cnn_top__DOT__linear_weight_rom__DOT__rom_addr))
            ? vlSelfRef.edge_cnn_top__DOT__linear_weight_rom__DOT__rom
           [vlSelfRef.edge_cnn_top__DOT__linear_weight_rom__DOT__rom_addr]
            : 0U);
}

void Vedge_cnn_top___024root___nba_sequent__TOP__2(Vedge_cnn_top___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vedge_cnn_top___024root___nba_sequent__TOP__2\n"); );
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
    if (vlSelfRef.rst_n) {
        vlSelfRef.edge_cnn_top__DOT__pool1_out_we = 0U;
        vlSelfRef.edge_cnn_top__DOT__pool1_done = 0U;
        if ((4U & (IData)(vlSelfRef.edge_cnn_top__DOT__pool1__DOT__state))) {
            if ((2U & (IData)(vlSelfRef.edge_cnn_top__DOT__pool1__DOT__state))) {
                vlSelfRef.__Vdly__edge_cnn_top__DOT__pool1__DOT__state = 0U;
            } else if ((1U & (IData)(vlSelfRef.edge_cnn_top__DOT__pool1__DOT__state))) {
                vlSelfRef.__Vdly__edge_cnn_top__DOT__pool1__DOT__state = 0U;
            } else {
                vlSelfRef.edge_cnn_top__DOT__pool1_busy = 0U;
                vlSelfRef.edge_cnn_top__DOT__pool1_done = 1U;
                vlSelfRef.__Vdly__edge_cnn_top__DOT__pool1__DOT__state = 0U;
            }
        } else if ((2U & (IData)(vlSelfRef.edge_cnn_top__DOT__pool1__DOT__state))) {
            if ((1U & (IData)(vlSelfRef.edge_cnn_top__DOT__pool1__DOT__state))) {
                vlSelfRef.edge_cnn_top__DOT__pool1_out_addr 
                    = (0x0007ffffU & (VL_SHIFTL_III(19,32,32, (IData)(vlSelfRef.edge_cnn_top__DOT__pool1__DOT__ch), 8U) 
                                      + ((IData)(vlSelfRef.edge_cnn_top__DOT__pool1__DOT__col) 
                                         + ((IData)(vlSelfRef.edge_cnn_top__DOT__pool1__DOT__row) 
                                            << 4U))));
                vlSelfRef.edge_cnn_top__DOT__pool1_out_data 
                    = vlSelfRef.edge_cnn_top__DOT__pool1__DOT__current_max;
                vlSelfRef.edge_cnn_top__DOT__pool1_out_we = 1U;
                vlSelfRef.__Vdly__edge_cnn_top__DOT__pool1__DOT__tap = 0U;
                if (((0x001fU == (IData)(vlSelfRef.edge_cnn_top__DOT__pool1__DOT__ch)) 
                     & ((0x000fU == (IData)(vlSelfRef.edge_cnn_top__DOT__pool1__DOT__row)) 
                        & (0x000fU == (IData)(vlSelfRef.edge_cnn_top__DOT__pool1__DOT__col))))) {
                    vlSelfRef.__Vdly__edge_cnn_top__DOT__pool1__DOT__state = 4U;
                } else {
                    if ((0x000fU != (IData)(vlSelfRef.edge_cnn_top__DOT__pool1__DOT__col))) {
                        vlSelfRef.__Vdly__edge_cnn_top__DOT__pool1__DOT__col 
                            = (0x0000ffffU & ((IData)(1U) 
                                              + (IData)(vlSelfRef.edge_cnn_top__DOT__pool1__DOT__col)));
                    } else {
                        if ((0x000fU != (IData)(vlSelfRef.edge_cnn_top__DOT__pool1__DOT__row))) {
                            vlSelfRef.__Vdly__edge_cnn_top__DOT__pool1__DOT__row 
                                = (0x0000ffffU & ((IData)(1U) 
                                                  + (IData)(vlSelfRef.edge_cnn_top__DOT__pool1__DOT__row)));
                        } else {
                            vlSelfRef.__Vdly__edge_cnn_top__DOT__pool1__DOT__ch 
                                = (0x0000ffffU & ((IData)(1U) 
                                                  + (IData)(vlSelfRef.edge_cnn_top__DOT__pool1__DOT__ch)));
                            vlSelfRef.__Vdly__edge_cnn_top__DOT__pool1__DOT__row = 0U;
                        }
                        vlSelfRef.__Vdly__edge_cnn_top__DOT__pool1__DOT__col = 0U;
                    }
                    vlSelfRef.__Vdly__edge_cnn_top__DOT__pool1__DOT__state = 1U;
                }
                vlSelfRef.__Vdly__edge_cnn_top__DOT__pool1__DOT__current_max = 0x80U;
            } else {
                if (((0U == (IData)(vlSelfRef.edge_cnn_top__DOT__pool1__DOT__tap)) 
                     | VL_GTS_III(8, (IData)(vlSelfRef.edge_cnn_top__DOT__buf_b_rdata), (IData)(vlSelfRef.edge_cnn_top__DOT__pool1__DOT__current_max)))) {
                    vlSelfRef.__Vdly__edge_cnn_top__DOT__pool1__DOT__current_max 
                        = vlSelfRef.edge_cnn_top__DOT__buf_b_rdata;
                }
                if ((3U == (IData)(vlSelfRef.edge_cnn_top__DOT__pool1__DOT__tap))) {
                    vlSelfRef.__Vdly__edge_cnn_top__DOT__pool1__DOT__state = 3U;
                } else {
                    vlSelfRef.__Vdly__edge_cnn_top__DOT__pool1__DOT__tap 
                        = (3U & ((IData)(1U) + (IData)(vlSelfRef.edge_cnn_top__DOT__pool1__DOT__tap)));
                    vlSelfRef.__Vdly__edge_cnn_top__DOT__pool1__DOT__state = 1U;
                }
            }
        } else if ((1U & (IData)(vlSelfRef.edge_cnn_top__DOT__pool1__DOT__state))) {
            vlSelfRef.__Vdly__edge_cnn_top__DOT__pool1__DOT__state = 2U;
        } else {
            vlSelfRef.edge_cnn_top__DOT__pool1_busy = 0U;
            if (vlSelfRef.pool1_start) {
                vlSelfRef.__Vdly__edge_cnn_top__DOT__pool1__DOT__ch = 0U;
                vlSelfRef.__Vdly__edge_cnn_top__DOT__pool1__DOT__row = 0U;
                vlSelfRef.__Vdly__edge_cnn_top__DOT__pool1__DOT__col = 0U;
                vlSelfRef.__Vdly__edge_cnn_top__DOT__pool1__DOT__tap = 0U;
                vlSelfRef.__Vdly__edge_cnn_top__DOT__pool1__DOT__current_max = 0x80U;
                vlSelfRef.edge_cnn_top__DOT__pool1_busy = 1U;
                vlSelfRef.__Vdly__edge_cnn_top__DOT__pool1__DOT__state = 1U;
            }
        }
        vlSelfRef.pool1_start = vlSelfRef.__Vdly__pool1_start;
        vlSelfRef.edge_cnn_top__DOT__pool1__DOT__state 
            = vlSelfRef.__Vdly__edge_cnn_top__DOT__pool1__DOT__state;
        vlSelfRef.edge_cnn_top__DOT__pool1__DOT__current_max 
            = vlSelfRef.__Vdly__edge_cnn_top__DOT__pool1__DOT__current_max;
        vlSelfRef.edge_cnn_top__DOT__pool2_out_we = 0U;
        vlSelfRef.edge_cnn_top__DOT__pool2_done = 0U;
        if ((4U & (IData)(vlSelfRef.edge_cnn_top__DOT__pool2__DOT__state))) {
            if ((2U & (IData)(vlSelfRef.edge_cnn_top__DOT__pool2__DOT__state))) {
                vlSelfRef.__Vdly__edge_cnn_top__DOT__pool2__DOT__state = 0U;
            } else if ((1U & (IData)(vlSelfRef.edge_cnn_top__DOT__pool2__DOT__state))) {
                vlSelfRef.__Vdly__edge_cnn_top__DOT__pool2__DOT__state = 0U;
            } else {
                vlSelfRef.edge_cnn_top__DOT__pool2_busy = 0U;
                vlSelfRef.edge_cnn_top__DOT__pool2_done = 1U;
                vlSelfRef.__Vdly__edge_cnn_top__DOT__pool2__DOT__state = 0U;
            }
        } else if ((2U & (IData)(vlSelfRef.edge_cnn_top__DOT__pool2__DOT__state))) {
            if ((1U & (IData)(vlSelfRef.edge_cnn_top__DOT__pool2__DOT__state))) {
                vlSelfRef.edge_cnn_top__DOT__pool2_out_addr 
                    = (0x0007ffffU & (VL_SHIFTL_III(19,32,32, (IData)(vlSelfRef.edge_cnn_top__DOT__pool2__DOT__ch), 6U) 
                                      + ((IData)(vlSelfRef.edge_cnn_top__DOT__pool2__DOT__col) 
                                         + ((IData)(vlSelfRef.edge_cnn_top__DOT__pool2__DOT__row) 
                                            << 3U))));
                vlSelfRef.edge_cnn_top__DOT__pool2_out_data 
                    = vlSelfRef.edge_cnn_top__DOT__pool2__DOT__current_max;
                vlSelfRef.edge_cnn_top__DOT__pool2_out_we = 1U;
                vlSelfRef.__Vdly__edge_cnn_top__DOT__pool2__DOT__tap = 0U;
                if (((0x003fU == (IData)(vlSelfRef.edge_cnn_top__DOT__pool2__DOT__ch)) 
                     & ((7U == (IData)(vlSelfRef.edge_cnn_top__DOT__pool2__DOT__row)) 
                        & (7U == (IData)(vlSelfRef.edge_cnn_top__DOT__pool2__DOT__col))))) {
                    vlSelfRef.__Vdly__edge_cnn_top__DOT__pool2__DOT__state = 4U;
                } else {
                    if ((7U != (IData)(vlSelfRef.edge_cnn_top__DOT__pool2__DOT__col))) {
                        vlSelfRef.__Vdly__edge_cnn_top__DOT__pool2__DOT__col 
                            = (0x0000ffffU & ((IData)(1U) 
                                              + (IData)(vlSelfRef.edge_cnn_top__DOT__pool2__DOT__col)));
                    } else {
                        if ((7U != (IData)(vlSelfRef.edge_cnn_top__DOT__pool2__DOT__row))) {
                            vlSelfRef.__Vdly__edge_cnn_top__DOT__pool2__DOT__row 
                                = (0x0000ffffU & ((IData)(1U) 
                                                  + (IData)(vlSelfRef.edge_cnn_top__DOT__pool2__DOT__row)));
                        } else {
                            vlSelfRef.__Vdly__edge_cnn_top__DOT__pool2__DOT__ch 
                                = (0x0000ffffU & ((IData)(1U) 
                                                  + (IData)(vlSelfRef.edge_cnn_top__DOT__pool2__DOT__ch)));
                            vlSelfRef.__Vdly__edge_cnn_top__DOT__pool2__DOT__row = 0U;
                        }
                        vlSelfRef.__Vdly__edge_cnn_top__DOT__pool2__DOT__col = 0U;
                    }
                    vlSelfRef.__Vdly__edge_cnn_top__DOT__pool2__DOT__state = 1U;
                }
                vlSelfRef.__Vdly__edge_cnn_top__DOT__pool2__DOT__current_max = 0x80U;
            } else {
                if (((0U == (IData)(vlSelfRef.edge_cnn_top__DOT__pool2__DOT__tap)) 
                     | VL_GTS_III(8, (IData)(vlSelfRef.edge_cnn_top__DOT__buf_b_rdata), (IData)(vlSelfRef.edge_cnn_top__DOT__pool2__DOT__current_max)))) {
                    vlSelfRef.__Vdly__edge_cnn_top__DOT__pool2__DOT__current_max 
                        = vlSelfRef.edge_cnn_top__DOT__buf_b_rdata;
                }
                if ((3U == (IData)(vlSelfRef.edge_cnn_top__DOT__pool2__DOT__tap))) {
                    vlSelfRef.__Vdly__edge_cnn_top__DOT__pool2__DOT__state = 3U;
                } else {
                    vlSelfRef.__Vdly__edge_cnn_top__DOT__pool2__DOT__tap 
                        = (3U & ((IData)(1U) + (IData)(vlSelfRef.edge_cnn_top__DOT__pool2__DOT__tap)));
                    vlSelfRef.__Vdly__edge_cnn_top__DOT__pool2__DOT__state = 1U;
                }
            }
        } else if ((1U & (IData)(vlSelfRef.edge_cnn_top__DOT__pool2__DOT__state))) {
            vlSelfRef.__Vdly__edge_cnn_top__DOT__pool2__DOT__state = 2U;
        } else {
            vlSelfRef.edge_cnn_top__DOT__pool2_busy = 0U;
            if (vlSelfRef.pool2_start) {
                vlSelfRef.__Vdly__edge_cnn_top__DOT__pool2__DOT__ch = 0U;
                vlSelfRef.__Vdly__edge_cnn_top__DOT__pool2__DOT__row = 0U;
                vlSelfRef.__Vdly__edge_cnn_top__DOT__pool2__DOT__col = 0U;
                vlSelfRef.__Vdly__edge_cnn_top__DOT__pool2__DOT__tap = 0U;
                vlSelfRef.__Vdly__edge_cnn_top__DOT__pool2__DOT__current_max = 0x80U;
                vlSelfRef.edge_cnn_top__DOT__pool2_busy = 1U;
                vlSelfRef.__Vdly__edge_cnn_top__DOT__pool2__DOT__state = 1U;
            }
        }
        vlSelfRef.pool2_start = vlSelfRef.__Vdly__pool2_start;
        vlSelfRef.edge_cnn_top__DOT__pool2__DOT__state 
            = vlSelfRef.__Vdly__edge_cnn_top__DOT__pool2__DOT__state;
        vlSelfRef.edge_cnn_top__DOT__pool2__DOT__current_max 
            = vlSelfRef.__Vdly__edge_cnn_top__DOT__pool2__DOT__current_max;
        if (vlSelfRef.edge_cnn_top__DOT__conv4__DOT__product_valid) {
            vlSelfRef.edge_cnn_top__DOT__conv4__DOT__postprocess_valid = 1U;
            vlSelfRef.edge_cnn_top__DOT__conv4__DOT__out_data_reg 
                = (VL_LTS_III(32, 0x0000007fU, vlSelfRef.edge_cnn_top__DOT__conv4__DOT__activated)
                    ? 0x0000007fU : (VL_GTS_III(32, 0xffffff80U, vlSelfRef.edge_cnn_top__DOT__conv4__DOT__activated)
                                      ? 0x00000080U
                                      : (0x000000ffU 
                                         & vlSelfRef.edge_cnn_top__DOT__conv4__DOT__activated)));
            vlSelfRef.edge_cnn_top__DOT__conv4__DOT__out_addr_reg 
                = vlSelfRef.edge_cnn_top__DOT__conv4__DOT__product_addr_reg;
        } else {
            vlSelfRef.edge_cnn_top__DOT__conv4__DOT__postprocess_valid = 0U;
        }
        vlSelfRef.edge_cnn_top__DOT__gap_out_we = 0U;
        vlSelfRef.edge_cnn_top__DOT__gap_done = 0U;
        if ((4U & (IData)(vlSelfRef.edge_cnn_top__DOT__gap__DOT__state))) {
            if ((2U & (IData)(vlSelfRef.edge_cnn_top__DOT__gap__DOT__state))) {
                vlSelfRef.__Vdly__edge_cnn_top__DOT__gap__DOT__state = 0U;
            } else if ((1U & (IData)(vlSelfRef.edge_cnn_top__DOT__gap__DOT__state))) {
                vlSelfRef.__Vdly__edge_cnn_top__DOT__gap__DOT__state = 0U;
            } else {
                vlSelfRef.edge_cnn_top__DOT__gap_busy = 0U;
                vlSelfRef.edge_cnn_top__DOT__gap_done = 1U;
                vlSelfRef.__Vdly__edge_cnn_top__DOT__gap__DOT__state = 0U;
            }
        } else if ((2U & (IData)(vlSelfRef.edge_cnn_top__DOT__gap__DOT__state))) {
            if ((1U & (IData)(vlSelfRef.edge_cnn_top__DOT__gap__DOT__state))) {
                vlSelfRef.edge_cnn_top__DOT__gap_out_addr 
                    = (0x0007ffffU & (IData)(vlSelfRef.edge_cnn_top__DOT__gap__DOT__ch));
                vlSelfRef.edge_cnn_top__DOT__gap_out_data 
                    = (0x000000ffU & (vlSelfRef.edge_cnn_top__DOT__gap__DOT__sum 
                                      >> 6U));
                vlSelfRef.edge_cnn_top__DOT__gap_out_we = 1U;
                vlSelfRef.__Vdly__edge_cnn_top__DOT__gap__DOT__idx = 0U;
                if ((0x00ffU == (IData)(vlSelfRef.edge_cnn_top__DOT__gap__DOT__ch))) {
                    vlSelfRef.__Vdly__edge_cnn_top__DOT__gap__DOT__state = 4U;
                } else {
                    vlSelfRef.__Vdly__edge_cnn_top__DOT__gap__DOT__ch 
                        = (0x0000ffffU & ((IData)(1U) 
                                          + (IData)(vlSelfRef.edge_cnn_top__DOT__gap__DOT__ch)));
                    vlSelfRef.__Vdly__edge_cnn_top__DOT__gap__DOT__state = 1U;
                }
                vlSelfRef.__Vdly__edge_cnn_top__DOT__gap__DOT__sum = 0U;
            } else {
                vlSelfRef.__Vdly__edge_cnn_top__DOT__gap__DOT__sum 
                    = (vlSelfRef.edge_cnn_top__DOT__gap__DOT__sum 
                       + (((- (IData)((1U & ((IData)(vlSelfRef.edge_cnn_top__DOT__buf_a_rdata) 
                                             >> 7U)))) 
                           << 8U) | (IData)(vlSelfRef.edge_cnn_top__DOT__buf_a_rdata)));
                if ((0x003fU == (IData)(vlSelfRef.edge_cnn_top__DOT__gap__DOT__idx))) {
                    vlSelfRef.__Vdly__edge_cnn_top__DOT__gap__DOT__state = 3U;
                } else {
                    vlSelfRef.__Vdly__edge_cnn_top__DOT__gap__DOT__idx 
                        = (0x0000ffffU & ((IData)(1U) 
                                          + (IData)(vlSelfRef.edge_cnn_top__DOT__gap__DOT__idx)));
                    vlSelfRef.__Vdly__edge_cnn_top__DOT__gap__DOT__state = 1U;
                }
            }
        } else if ((1U & (IData)(vlSelfRef.edge_cnn_top__DOT__gap__DOT__state))) {
            vlSelfRef.__Vdly__edge_cnn_top__DOT__gap__DOT__state = 2U;
        } else {
            vlSelfRef.edge_cnn_top__DOT__gap_busy = 0U;
            if (vlSelfRef.gap_start) {
                vlSelfRef.__Vdly__edge_cnn_top__DOT__gap__DOT__ch = 0U;
                vlSelfRef.__Vdly__edge_cnn_top__DOT__gap__DOT__idx = 0U;
                vlSelfRef.__Vdly__edge_cnn_top__DOT__gap__DOT__sum = 0U;
                vlSelfRef.edge_cnn_top__DOT__gap_busy = 1U;
                vlSelfRef.__Vdly__edge_cnn_top__DOT__gap__DOT__state = 1U;
            }
        }
        vlSelfRef.gap_start = vlSelfRef.__Vdly__gap_start;
        vlSelfRef.edge_cnn_top__DOT__gap__DOT__state 
            = vlSelfRef.__Vdly__edge_cnn_top__DOT__gap__DOT__state;
        vlSelfRef.edge_cnn_top__DOT__gap__DOT__sum 
            = vlSelfRef.__Vdly__edge_cnn_top__DOT__gap__DOT__sum;
        if (vlSelfRef.edge_cnn_top__DOT__conv1__DOT__product_valid) {
            vlSelfRef.edge_cnn_top__DOT__conv1__DOT__postprocess_valid = 1U;
            vlSelfRef.edge_cnn_top__DOT__conv1__DOT__out_data_reg 
                = (VL_LTS_III(32, 0x0000007fU, vlSelfRef.edge_cnn_top__DOT__conv1__DOT__activated)
                    ? 0x0000007fU : (VL_GTS_III(32, 0xffffff80U, vlSelfRef.edge_cnn_top__DOT__conv1__DOT__activated)
                                      ? 0x00000080U
                                      : (0x000000ffU 
                                         & vlSelfRef.edge_cnn_top__DOT__conv1__DOT__activated)));
            vlSelfRef.edge_cnn_top__DOT__conv1__DOT__out_addr_reg 
                = vlSelfRef.edge_cnn_top__DOT__conv1__DOT__product_addr_reg;
        } else {
            vlSelfRef.edge_cnn_top__DOT__conv1__DOT__postprocess_valid = 0U;
        }
        if (vlSelfRef.edge_cnn_top__DOT__conv2__DOT__product_valid) {
            vlSelfRef.edge_cnn_top__DOT__conv2__DOT__postprocess_valid = 1U;
            vlSelfRef.edge_cnn_top__DOT__conv2__DOT__out_data_reg 
                = (VL_LTS_III(32, 0x0000007fU, vlSelfRef.edge_cnn_top__DOT__conv2__DOT__activated)
                    ? 0x0000007fU : (VL_GTS_III(32, 0xffffff80U, vlSelfRef.edge_cnn_top__DOT__conv2__DOT__activated)
                                      ? 0x00000080U
                                      : (0x000000ffU 
                                         & vlSelfRef.edge_cnn_top__DOT__conv2__DOT__activated)));
            vlSelfRef.edge_cnn_top__DOT__conv2__DOT__out_addr_reg 
                = vlSelfRef.edge_cnn_top__DOT__conv2__DOT__product_addr_reg;
        } else {
            vlSelfRef.edge_cnn_top__DOT__conv2__DOT__postprocess_valid = 0U;
        }
        if (vlSelfRef.edge_cnn_top__DOT__conv3__DOT__product_valid) {
            vlSelfRef.edge_cnn_top__DOT__conv3__DOT__postprocess_valid = 1U;
            vlSelfRef.edge_cnn_top__DOT__conv3__DOT__out_data_reg 
                = (VL_LTS_III(32, 0x0000007fU, vlSelfRef.edge_cnn_top__DOT__conv3__DOT__activated)
                    ? 0x0000007fU : (VL_GTS_III(32, 0xffffff80U, vlSelfRef.edge_cnn_top__DOT__conv3__DOT__activated)
                                      ? 0x00000080U
                                      : (0x000000ffU 
                                         & vlSelfRef.edge_cnn_top__DOT__conv3__DOT__activated)));
            vlSelfRef.edge_cnn_top__DOT__conv3__DOT__out_addr_reg 
                = vlSelfRef.edge_cnn_top__DOT__conv3__DOT__product_addr_reg;
        } else {
            vlSelfRef.edge_cnn_top__DOT__conv3__DOT__postprocess_valid = 0U;
        }
        if (vlSelfRef.edge_cnn_top__DOT__conv4__DOT__postprocess_pending) {
            vlSelfRef.edge_cnn_top__DOT__conv4__DOT__product_valid = 1U;
            vlSelfRef.edge_cnn_top__DOT__conv4__DOT__product_reg 
                = (0x000007ffffffffffULL & VL_MULS_QQQ(43, 0x000000000000dbe9ULL, 
                                                       (0x000007ffffffffffULL 
                                                        & VL_EXTENDS_QI(43,25, 
                                                                        (0x01ffffffU 
                                                                         & vlSelfRef.edge_cnn_top__DOT__conv4__DOT__biased_sum_reg)))));
            vlSelfRef.edge_cnn_top__DOT__conv4__DOT__product_addr_reg 
                = vlSelfRef.edge_cnn_top__DOT__conv4__DOT__controller_out_addr;
        } else {
            vlSelfRef.edge_cnn_top__DOT__conv4__DOT__product_valid = 0U;
        }
    } else {
        vlSelfRef.__Vdly__edge_cnn_top__DOT__pool1__DOT__ch = 0U;
        vlSelfRef.__Vdly__edge_cnn_top__DOT__pool1__DOT__row = 0U;
        vlSelfRef.__Vdly__edge_cnn_top__DOT__pool1__DOT__col = 0U;
        vlSelfRef.__Vdly__edge_cnn_top__DOT__pool1__DOT__state = 0U;
        vlSelfRef.__Vdly__edge_cnn_top__DOT__pool1__DOT__tap = 0U;
        vlSelfRef.__Vdly__edge_cnn_top__DOT__pool1__DOT__current_max = 0x80U;
        vlSelfRef.edge_cnn_top__DOT__pool1_out_data = 0U;
        vlSelfRef.edge_cnn_top__DOT__pool1_out_addr = 0U;
        vlSelfRef.edge_cnn_top__DOT__pool1_out_we = 0U;
        vlSelfRef.edge_cnn_top__DOT__pool1_busy = 0U;
        vlSelfRef.edge_cnn_top__DOT__pool1_done = 0U;
        vlSelfRef.pool1_start = vlSelfRef.__Vdly__pool1_start;
        vlSelfRef.edge_cnn_top__DOT__pool1__DOT__state 
            = vlSelfRef.__Vdly__edge_cnn_top__DOT__pool1__DOT__state;
        vlSelfRef.edge_cnn_top__DOT__pool1__DOT__current_max 
            = vlSelfRef.__Vdly__edge_cnn_top__DOT__pool1__DOT__current_max;
        vlSelfRef.__Vdly__edge_cnn_top__DOT__pool2__DOT__ch = 0U;
        vlSelfRef.__Vdly__edge_cnn_top__DOT__pool2__DOT__row = 0U;
        vlSelfRef.__Vdly__edge_cnn_top__DOT__pool2__DOT__col = 0U;
        vlSelfRef.__Vdly__edge_cnn_top__DOT__pool2__DOT__state = 0U;
        vlSelfRef.__Vdly__edge_cnn_top__DOT__pool2__DOT__tap = 0U;
        vlSelfRef.__Vdly__edge_cnn_top__DOT__pool2__DOT__current_max = 0x80U;
        vlSelfRef.edge_cnn_top__DOT__pool2_out_data = 0U;
        vlSelfRef.edge_cnn_top__DOT__pool2_out_addr = 0U;
        vlSelfRef.edge_cnn_top__DOT__pool2_out_we = 0U;
        vlSelfRef.edge_cnn_top__DOT__pool2_busy = 0U;
        vlSelfRef.edge_cnn_top__DOT__pool2_done = 0U;
        vlSelfRef.pool2_start = vlSelfRef.__Vdly__pool2_start;
        vlSelfRef.edge_cnn_top__DOT__pool2__DOT__state 
            = vlSelfRef.__Vdly__edge_cnn_top__DOT__pool2__DOT__state;
        vlSelfRef.edge_cnn_top__DOT__pool2__DOT__current_max 
            = vlSelfRef.__Vdly__edge_cnn_top__DOT__pool2__DOT__current_max;
        vlSelfRef.edge_cnn_top__DOT__conv4__DOT__postprocess_valid = 0U;
        vlSelfRef.edge_cnn_top__DOT__conv4__DOT__out_data_reg = 0U;
        vlSelfRef.edge_cnn_top__DOT__conv4__DOT__out_addr_reg = 0U;
        vlSelfRef.__Vdly__edge_cnn_top__DOT__gap__DOT__ch = 0U;
        vlSelfRef.__Vdly__edge_cnn_top__DOT__gap__DOT__state = 0U;
        vlSelfRef.__Vdly__edge_cnn_top__DOT__gap__DOT__idx = 0U;
        vlSelfRef.__Vdly__edge_cnn_top__DOT__gap__DOT__sum = 0U;
        vlSelfRef.edge_cnn_top__DOT__gap_out_we = 0U;
        vlSelfRef.edge_cnn_top__DOT__gap_out_addr = 0U;
        vlSelfRef.edge_cnn_top__DOT__gap_out_data = 0U;
        vlSelfRef.edge_cnn_top__DOT__gap_busy = 0U;
        vlSelfRef.edge_cnn_top__DOT__gap_done = 0U;
        vlSelfRef.gap_start = vlSelfRef.__Vdly__gap_start;
        vlSelfRef.edge_cnn_top__DOT__gap__DOT__state 
            = vlSelfRef.__Vdly__edge_cnn_top__DOT__gap__DOT__state;
        vlSelfRef.edge_cnn_top__DOT__gap__DOT__sum 
            = vlSelfRef.__Vdly__edge_cnn_top__DOT__gap__DOT__sum;
        vlSelfRef.edge_cnn_top__DOT__conv1__DOT__postprocess_valid = 0U;
        vlSelfRef.edge_cnn_top__DOT__conv1__DOT__out_data_reg = 0U;
        vlSelfRef.edge_cnn_top__DOT__conv1__DOT__out_addr_reg = 0U;
        vlSelfRef.edge_cnn_top__DOT__conv2__DOT__postprocess_valid = 0U;
        vlSelfRef.edge_cnn_top__DOT__conv2__DOT__out_data_reg = 0U;
        vlSelfRef.edge_cnn_top__DOT__conv2__DOT__out_addr_reg = 0U;
        vlSelfRef.edge_cnn_top__DOT__conv3__DOT__postprocess_valid = 0U;
        vlSelfRef.edge_cnn_top__DOT__conv3__DOT__out_data_reg = 0U;
        vlSelfRef.edge_cnn_top__DOT__conv3__DOT__out_addr_reg = 0U;
        vlSelfRef.edge_cnn_top__DOT__conv4__DOT__product_reg = 0ULL;
        vlSelfRef.edge_cnn_top__DOT__conv4__DOT__product_valid = 0U;
        vlSelfRef.edge_cnn_top__DOT__conv4__DOT__product_addr_reg = 0U;
    }
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
    vlSelfRef.edge_cnn_top__DOT__conv4__DOT__postprocess_pending 
        = ((IData)(vlSelfRef.rst_n) && (IData)(vlSelfRef.edge_cnn_top__DOT__conv4__DOT__engine_done));
    if (vlSelfRef.rst_n) {
        if (vlSelfRef.edge_cnn_top__DOT__conv1__DOT__postprocess_pending) {
            vlSelfRef.edge_cnn_top__DOT__conv1__DOT__product_valid = 1U;
            vlSelfRef.edge_cnn_top__DOT__conv1__DOT__product_reg 
                = (0x000007ffffffffffULL & VL_MULS_QQQ(43, 0x000000000000ae9eULL, 
                                                       (0x000007ffffffffffULL 
                                                        & VL_EXTENDS_QI(43,25, 
                                                                        (0x01ffffffU 
                                                                         & vlSelfRef.edge_cnn_top__DOT__conv1__DOT__biased_sum_reg)))));
            vlSelfRef.edge_cnn_top__DOT__conv1__DOT__product_addr_reg 
                = vlSelfRef.edge_cnn_top__DOT__conv1__DOT__controller_out_addr;
        } else {
            vlSelfRef.edge_cnn_top__DOT__conv1__DOT__product_valid = 0U;
        }
    } else {
        vlSelfRef.edge_cnn_top__DOT__conv1__DOT__product_reg = 0ULL;
        vlSelfRef.edge_cnn_top__DOT__conv1__DOT__product_valid = 0U;
        vlSelfRef.edge_cnn_top__DOT__conv1__DOT__product_addr_reg = 0U;
    }
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
    vlSelfRef.edge_cnn_top__DOT__conv1__DOT__postprocess_pending 
        = ((IData)(vlSelfRef.rst_n) && (IData)(vlSelfRef.edge_cnn_top__DOT__conv1__DOT__engine_done));
    if (vlSelfRef.rst_n) {
        if (vlSelfRef.edge_cnn_top__DOT__conv2__DOT__postprocess_pending) {
            vlSelfRef.edge_cnn_top__DOT__conv2__DOT__product_valid = 1U;
            vlSelfRef.edge_cnn_top__DOT__conv2__DOT__product_reg 
                = (0x000007ffffffffffULL & VL_MULS_QQQ(43, 0x000000000000f788ULL, 
                                                       (0x000007ffffffffffULL 
                                                        & VL_EXTENDS_QI(43,25, 
                                                                        (0x01ffffffU 
                                                                         & vlSelfRef.edge_cnn_top__DOT__conv2__DOT__biased_sum_reg)))));
            vlSelfRef.edge_cnn_top__DOT__conv2__DOT__product_addr_reg 
                = vlSelfRef.edge_cnn_top__DOT__conv2__DOT__controller_out_addr;
        } else {
            vlSelfRef.edge_cnn_top__DOT__conv2__DOT__product_valid = 0U;
        }
    } else {
        vlSelfRef.edge_cnn_top__DOT__conv2__DOT__product_reg = 0ULL;
        vlSelfRef.edge_cnn_top__DOT__conv2__DOT__product_valid = 0U;
        vlSelfRef.edge_cnn_top__DOT__conv2__DOT__product_addr_reg = 0U;
    }
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
    vlSelfRef.edge_cnn_top__DOT__conv2__DOT__postprocess_pending 
        = ((IData)(vlSelfRef.rst_n) && (IData)(vlSelfRef.edge_cnn_top__DOT__conv2__DOT__engine_done));
    if (vlSelfRef.rst_n) {
        if (vlSelfRef.edge_cnn_top__DOT__conv3__DOT__postprocess_pending) {
            vlSelfRef.edge_cnn_top__DOT__conv3__DOT__product_valid = 1U;
            vlSelfRef.edge_cnn_top__DOT__conv3__DOT__product_reg 
                = (0x000007ffffffffffULL & VL_MULS_QQQ(43, 0x000000000000c8d7ULL, 
                                                       (0x000007ffffffffffULL 
                                                        & VL_EXTENDS_QI(43,25, 
                                                                        (0x01ffffffU 
                                                                         & vlSelfRef.edge_cnn_top__DOT__conv3__DOT__biased_sum_reg)))));
            vlSelfRef.edge_cnn_top__DOT__conv3__DOT__product_addr_reg 
                = vlSelfRef.edge_cnn_top__DOT__conv3__DOT__controller_out_addr;
        } else {
            vlSelfRef.edge_cnn_top__DOT__conv3__DOT__product_valid = 0U;
        }
    } else {
        vlSelfRef.edge_cnn_top__DOT__conv3__DOT__product_reg = 0ULL;
        vlSelfRef.edge_cnn_top__DOT__conv3__DOT__product_valid = 0U;
        vlSelfRef.edge_cnn_top__DOT__conv3__DOT__product_addr_reg = 0U;
    }
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
    vlSelfRef.edge_cnn_top__DOT__conv3__DOT__postprocess_pending 
        = ((IData)(vlSelfRef.rst_n) && (IData)(vlSelfRef.edge_cnn_top__DOT__conv3__DOT__engine_done));
    if (vlSelfRef.rst_n) {
        if (vlSelfRef.edge_cnn_top__DOT__conv4__DOT__engine_done) {
            vlSelfRef.edge_cnn_top__DOT__conv4__DOT__biased_sum_reg 
                = (vlSelfRef.edge_cnn_top__DOT__conv4__DOT__raw_sum 
                   + vlSelfRef.edge_cnn_top__DOT__conv4_bias_rom__DOT__rom
                   [(0x000000ffU & (IData)(vlSelfRef.edge_cnn_top__DOT__conv4__DOT__out_channel))]);
        }
        vlSelfRef.__Vdly__edge_cnn_top__DOT__conv4__DOT__engine_start = 0U;
        vlSelfRef.edge_cnn_top__DOT__conv4__DOT__controller_done = 0U;
        if ((2U & (IData)(vlSelfRef.edge_cnn_top__DOT__conv4__DOT__controller__DOT__state))) {
            if ((1U & (IData)(vlSelfRef.edge_cnn_top__DOT__conv4__DOT__controller__DOT__state))) {
                vlSelfRef.edge_cnn_top__DOT__conv4__DOT__controller_busy = 0U;
                vlSelfRef.edge_cnn_top__DOT__conv4__DOT__controller_done = 1U;
                vlSelfRef.__Vdly__edge_cnn_top__DOT__conv4__DOT__controller__DOT__state = 0U;
            } else if (vlSelfRef.edge_cnn_top__DOT__conv4__DOT__engine_done) {
                vlSelfRef.edge_cnn_top__DOT__conv4__DOT__controller_out_addr 
                    = (0x0007ffffU & (VL_SHIFTL_III(19,32,32, (IData)(vlSelfRef.edge_cnn_top__DOT__conv4__DOT__out_channel), 6U) 
                                      + (VL_SHIFTL_III(19,32,32, (IData)(vlSelfRef.edge_cnn_top__DOT__conv4__DOT__out_row), 3U) 
                                         + (IData)(vlSelfRef.edge_cnn_top__DOT__conv4__DOT__out_col))));
                if (((0x00ffU == (IData)(vlSelfRef.edge_cnn_top__DOT__conv4__DOT__out_channel)) 
                     & ((7U == (IData)(vlSelfRef.edge_cnn_top__DOT__conv4__DOT__out_row)) 
                        & (7U == (IData)(vlSelfRef.edge_cnn_top__DOT__conv4__DOT__out_col))))) {
                    vlSelfRef.__Vdly__edge_cnn_top__DOT__conv4__DOT__controller__DOT__state = 3U;
                } else {
                    if ((7U != (IData)(vlSelfRef.edge_cnn_top__DOT__conv4__DOT__out_col))) {
                        vlSelfRef.edge_cnn_top__DOT__conv4__DOT__out_col 
                            = (0x0000ffffU & ((IData)(1U) 
                                              + (IData)(vlSelfRef.edge_cnn_top__DOT__conv4__DOT__out_col)));
                    } else {
                        if ((7U != (IData)(vlSelfRef.edge_cnn_top__DOT__conv4__DOT__out_row))) {
                            vlSelfRef.edge_cnn_top__DOT__conv4__DOT__out_row 
                                = (0x0000ffffU & ((IData)(1U) 
                                                  + (IData)(vlSelfRef.edge_cnn_top__DOT__conv4__DOT__out_row)));
                        } else {
                            vlSelfRef.edge_cnn_top__DOT__conv4__DOT__out_channel 
                                = (0x0000ffffU & ((IData)(1U) 
                                                  + (IData)(vlSelfRef.edge_cnn_top__DOT__conv4__DOT__out_channel)));
                            vlSelfRef.edge_cnn_top__DOT__conv4__DOT__out_row = 0U;
                        }
                        vlSelfRef.edge_cnn_top__DOT__conv4__DOT__out_col = 0U;
                    }
                    vlSelfRef.__Vdly__edge_cnn_top__DOT__conv4__DOT__controller__DOT__state = 1U;
                }
            }
        } else if ((1U & (IData)(vlSelfRef.edge_cnn_top__DOT__conv4__DOT__controller__DOT__state))) {
            vlSelfRef.__Vdly__edge_cnn_top__DOT__conv4__DOT__engine_start = 1U;
            vlSelfRef.__Vdly__edge_cnn_top__DOT__conv4__DOT__controller__DOT__state = 2U;
        } else {
            vlSelfRef.edge_cnn_top__DOT__conv4__DOT__controller_busy = 0U;
            if (vlSelfRef.conv4_start) {
                vlSelfRef.edge_cnn_top__DOT__conv4__DOT__out_row = 0U;
                vlSelfRef.edge_cnn_top__DOT__conv4__DOT__out_col = 0U;
                vlSelfRef.edge_cnn_top__DOT__conv4__DOT__out_channel = 0U;
                vlSelfRef.edge_cnn_top__DOT__conv4__DOT__controller_busy = 1U;
                vlSelfRef.__Vdly__edge_cnn_top__DOT__conv4__DOT__controller__DOT__state = 1U;
            }
        }
        if (vlSelfRef.edge_cnn_top__DOT__conv1__DOT__engine_done) {
            vlSelfRef.edge_cnn_top__DOT__conv1__DOT__biased_sum_reg 
                = (vlSelfRef.edge_cnn_top__DOT__conv1__DOT__raw_sum 
                   + vlSelfRef.edge_cnn_top__DOT__conv1_bias_rom__DOT__rom
                   [(0x0000001fU & (IData)(vlSelfRef.edge_cnn_top__DOT__conv1__DOT__out_channel))]);
        }
        vlSelfRef.__Vdly__edge_cnn_top__DOT__conv1__DOT__engine_start = 0U;
        vlSelfRef.edge_cnn_top__DOT__conv1__DOT__controller_done = 0U;
        if ((2U & (IData)(vlSelfRef.edge_cnn_top__DOT__conv1__DOT__controller__DOT__state))) {
            if ((1U & (IData)(vlSelfRef.edge_cnn_top__DOT__conv1__DOT__controller__DOT__state))) {
                vlSelfRef.edge_cnn_top__DOT__conv1__DOT__controller_busy = 0U;
                vlSelfRef.edge_cnn_top__DOT__conv1__DOT__controller_done = 1U;
                vlSelfRef.__Vdly__edge_cnn_top__DOT__conv1__DOT__controller__DOT__state = 0U;
            } else if (vlSelfRef.edge_cnn_top__DOT__conv1__DOT__engine_done) {
                vlSelfRef.edge_cnn_top__DOT__conv1__DOT__controller_out_addr 
                    = (0x0007ffffU & (VL_SHIFTL_III(19,32,32, (IData)(vlSelfRef.edge_cnn_top__DOT__conv1__DOT__out_channel), 0x0000000aU) 
                                      + (VL_SHIFTL_III(19,32,32, (IData)(vlSelfRef.edge_cnn_top__DOT__conv1__DOT__out_row), 5U) 
                                         + (IData)(vlSelfRef.edge_cnn_top__DOT__conv1__DOT__out_col))));
                if (((0x001fU == (IData)(vlSelfRef.edge_cnn_top__DOT__conv1__DOT__out_channel)) 
                     & ((0x001fU == (IData)(vlSelfRef.edge_cnn_top__DOT__conv1__DOT__out_row)) 
                        & (0x001fU == (IData)(vlSelfRef.edge_cnn_top__DOT__conv1__DOT__out_col))))) {
                    vlSelfRef.__Vdly__edge_cnn_top__DOT__conv1__DOT__controller__DOT__state = 3U;
                } else {
                    if ((0x001fU != (IData)(vlSelfRef.edge_cnn_top__DOT__conv1__DOT__out_col))) {
                        vlSelfRef.edge_cnn_top__DOT__conv1__DOT__out_col 
                            = (0x0000ffffU & ((IData)(1U) 
                                              + (IData)(vlSelfRef.edge_cnn_top__DOT__conv1__DOT__out_col)));
                    } else {
                        if ((0x001fU != (IData)(vlSelfRef.edge_cnn_top__DOT__conv1__DOT__out_row))) {
                            vlSelfRef.edge_cnn_top__DOT__conv1__DOT__out_row 
                                = (0x0000ffffU & ((IData)(1U) 
                                                  + (IData)(vlSelfRef.edge_cnn_top__DOT__conv1__DOT__out_row)));
                        } else {
                            vlSelfRef.edge_cnn_top__DOT__conv1__DOT__out_channel 
                                = (0x0000ffffU & ((IData)(1U) 
                                                  + (IData)(vlSelfRef.edge_cnn_top__DOT__conv1__DOT__out_channel)));
                            vlSelfRef.edge_cnn_top__DOT__conv1__DOT__out_row = 0U;
                        }
                        vlSelfRef.edge_cnn_top__DOT__conv1__DOT__out_col = 0U;
                    }
                    vlSelfRef.__Vdly__edge_cnn_top__DOT__conv1__DOT__controller__DOT__state = 1U;
                }
            }
        } else if ((1U & (IData)(vlSelfRef.edge_cnn_top__DOT__conv1__DOT__controller__DOT__state))) {
            vlSelfRef.__Vdly__edge_cnn_top__DOT__conv1__DOT__engine_start = 1U;
            vlSelfRef.__Vdly__edge_cnn_top__DOT__conv1__DOT__controller__DOT__state = 2U;
        } else {
            vlSelfRef.edge_cnn_top__DOT__conv1__DOT__controller_busy = 0U;
            if (vlSelfRef.conv1_start) {
                vlSelfRef.edge_cnn_top__DOT__conv1__DOT__out_row = 0U;
                vlSelfRef.edge_cnn_top__DOT__conv1__DOT__out_col = 0U;
                vlSelfRef.edge_cnn_top__DOT__conv1__DOT__out_channel = 0U;
                vlSelfRef.edge_cnn_top__DOT__conv1__DOT__controller_busy = 1U;
                vlSelfRef.__Vdly__edge_cnn_top__DOT__conv1__DOT__controller__DOT__state = 1U;
            }
        }
        if (vlSelfRef.edge_cnn_top__DOT__conv2__DOT__engine_done) {
            vlSelfRef.edge_cnn_top__DOT__conv2__DOT__biased_sum_reg 
                = (vlSelfRef.edge_cnn_top__DOT__conv2__DOT__raw_sum 
                   + vlSelfRef.edge_cnn_top__DOT__conv2_bias_rom__DOT__rom
                   [(0x0000003fU & (IData)(vlSelfRef.edge_cnn_top__DOT__conv2__DOT__out_channel))]);
        }
        vlSelfRef.__Vdly__edge_cnn_top__DOT__conv2__DOT__engine_start = 0U;
        vlSelfRef.edge_cnn_top__DOT__conv2__DOT__controller_done = 0U;
        if ((2U & (IData)(vlSelfRef.edge_cnn_top__DOT__conv2__DOT__controller__DOT__state))) {
            if ((1U & (IData)(vlSelfRef.edge_cnn_top__DOT__conv2__DOT__controller__DOT__state))) {
                vlSelfRef.edge_cnn_top__DOT__conv2__DOT__controller_busy = 0U;
                vlSelfRef.edge_cnn_top__DOT__conv2__DOT__controller_done = 1U;
                vlSelfRef.__Vdly__edge_cnn_top__DOT__conv2__DOT__controller__DOT__state = 0U;
            } else if (vlSelfRef.edge_cnn_top__DOT__conv2__DOT__engine_done) {
                vlSelfRef.edge_cnn_top__DOT__conv2__DOT__controller_out_addr 
                    = (0x0007ffffU & (VL_SHIFTL_III(19,32,32, (IData)(vlSelfRef.edge_cnn_top__DOT__conv2__DOT__out_channel), 8U) 
                                      + (VL_SHIFTL_III(19,32,32, (IData)(vlSelfRef.edge_cnn_top__DOT__conv2__DOT__out_row), 4U) 
                                         + (IData)(vlSelfRef.edge_cnn_top__DOT__conv2__DOT__out_col))));
                if (((0x003fU == (IData)(vlSelfRef.edge_cnn_top__DOT__conv2__DOT__out_channel)) 
                     & ((0x000fU == (IData)(vlSelfRef.edge_cnn_top__DOT__conv2__DOT__out_row)) 
                        & (0x000fU == (IData)(vlSelfRef.edge_cnn_top__DOT__conv2__DOT__out_col))))) {
                    vlSelfRef.__Vdly__edge_cnn_top__DOT__conv2__DOT__controller__DOT__state = 3U;
                } else {
                    if ((0x000fU != (IData)(vlSelfRef.edge_cnn_top__DOT__conv2__DOT__out_col))) {
                        vlSelfRef.edge_cnn_top__DOT__conv2__DOT__out_col 
                            = (0x0000ffffU & ((IData)(1U) 
                                              + (IData)(vlSelfRef.edge_cnn_top__DOT__conv2__DOT__out_col)));
                    } else {
                        if ((0x000fU != (IData)(vlSelfRef.edge_cnn_top__DOT__conv2__DOT__out_row))) {
                            vlSelfRef.edge_cnn_top__DOT__conv2__DOT__out_row 
                                = (0x0000ffffU & ((IData)(1U) 
                                                  + (IData)(vlSelfRef.edge_cnn_top__DOT__conv2__DOT__out_row)));
                        } else {
                            vlSelfRef.edge_cnn_top__DOT__conv2__DOT__out_channel 
                                = (0x0000ffffU & ((IData)(1U) 
                                                  + (IData)(vlSelfRef.edge_cnn_top__DOT__conv2__DOT__out_channel)));
                            vlSelfRef.edge_cnn_top__DOT__conv2__DOT__out_row = 0U;
                        }
                        vlSelfRef.edge_cnn_top__DOT__conv2__DOT__out_col = 0U;
                    }
                    vlSelfRef.__Vdly__edge_cnn_top__DOT__conv2__DOT__controller__DOT__state = 1U;
                }
            }
        } else if ((1U & (IData)(vlSelfRef.edge_cnn_top__DOT__conv2__DOT__controller__DOT__state))) {
            vlSelfRef.__Vdly__edge_cnn_top__DOT__conv2__DOT__engine_start = 1U;
            vlSelfRef.__Vdly__edge_cnn_top__DOT__conv2__DOT__controller__DOT__state = 2U;
        } else {
            vlSelfRef.edge_cnn_top__DOT__conv2__DOT__controller_busy = 0U;
            if (vlSelfRef.conv2_start) {
                vlSelfRef.edge_cnn_top__DOT__conv2__DOT__out_row = 0U;
                vlSelfRef.edge_cnn_top__DOT__conv2__DOT__out_col = 0U;
                vlSelfRef.edge_cnn_top__DOT__conv2__DOT__out_channel = 0U;
                vlSelfRef.edge_cnn_top__DOT__conv2__DOT__controller_busy = 1U;
                vlSelfRef.__Vdly__edge_cnn_top__DOT__conv2__DOT__controller__DOT__state = 1U;
            }
        }
        if (vlSelfRef.edge_cnn_top__DOT__conv3__DOT__engine_done) {
            vlSelfRef.edge_cnn_top__DOT__conv3__DOT__biased_sum_reg 
                = (vlSelfRef.edge_cnn_top__DOT__conv3__DOT__raw_sum 
                   + vlSelfRef.edge_cnn_top__DOT__conv3_bias_rom__DOT__rom
                   [(0x0000007fU & (IData)(vlSelfRef.edge_cnn_top__DOT__conv3__DOT__out_channel))]);
        }
        vlSelfRef.__Vdly__edge_cnn_top__DOT__conv3__DOT__engine_start = 0U;
        vlSelfRef.edge_cnn_top__DOT__conv3__DOT__controller_done = 0U;
        if ((2U & (IData)(vlSelfRef.edge_cnn_top__DOT__conv3__DOT__controller__DOT__state))) {
            if ((1U & (IData)(vlSelfRef.edge_cnn_top__DOT__conv3__DOT__controller__DOT__state))) {
                vlSelfRef.edge_cnn_top__DOT__conv3__DOT__controller_busy = 0U;
                vlSelfRef.edge_cnn_top__DOT__conv3__DOT__controller_done = 1U;
                vlSelfRef.__Vdly__edge_cnn_top__DOT__conv3__DOT__controller__DOT__state = 0U;
            } else if (vlSelfRef.edge_cnn_top__DOT__conv3__DOT__engine_done) {
                vlSelfRef.edge_cnn_top__DOT__conv3__DOT__controller_out_addr 
                    = (0x0007ffffU & (vlSelfRef.__VdfgRegularize_h6e95ff9d_0_10 
                                      + (VL_SHIFTL_III(19,32,32, (IData)(vlSelfRef.edge_cnn_top__DOT__conv3__DOT__out_row), 3U) 
                                         + (IData)(vlSelfRef.edge_cnn_top__DOT__conv3__DOT__out_col))));
                if (((0x007fU == (IData)(vlSelfRef.edge_cnn_top__DOT__conv3__DOT__out_channel)) 
                     & ((7U == (IData)(vlSelfRef.edge_cnn_top__DOT__conv3__DOT__out_row)) 
                        & (7U == (IData)(vlSelfRef.edge_cnn_top__DOT__conv3__DOT__out_col))))) {
                    vlSelfRef.__Vdly__edge_cnn_top__DOT__conv3__DOT__controller__DOT__state = 3U;
                } else {
                    if ((7U != (IData)(vlSelfRef.edge_cnn_top__DOT__conv3__DOT__out_col))) {
                        vlSelfRef.edge_cnn_top__DOT__conv3__DOT__out_col 
                            = (0x0000ffffU & ((IData)(1U) 
                                              + (IData)(vlSelfRef.edge_cnn_top__DOT__conv3__DOT__out_col)));
                    } else {
                        if ((7U != (IData)(vlSelfRef.edge_cnn_top__DOT__conv3__DOT__out_row))) {
                            vlSelfRef.edge_cnn_top__DOT__conv3__DOT__out_row 
                                = (0x0000ffffU & ((IData)(1U) 
                                                  + (IData)(vlSelfRef.edge_cnn_top__DOT__conv3__DOT__out_row)));
                        } else {
                            vlSelfRef.edge_cnn_top__DOT__conv3__DOT__out_channel 
                                = (0x0000ffffU & ((IData)(1U) 
                                                  + (IData)(vlSelfRef.edge_cnn_top__DOT__conv3__DOT__out_channel)));
                            vlSelfRef.edge_cnn_top__DOT__conv3__DOT__out_row = 0U;
                        }
                        vlSelfRef.edge_cnn_top__DOT__conv3__DOT__out_col = 0U;
                    }
                    vlSelfRef.__Vdly__edge_cnn_top__DOT__conv3__DOT__controller__DOT__state = 1U;
                }
            }
        } else if ((1U & (IData)(vlSelfRef.edge_cnn_top__DOT__conv3__DOT__controller__DOT__state))) {
            vlSelfRef.__Vdly__edge_cnn_top__DOT__conv3__DOT__engine_start = 1U;
            vlSelfRef.__Vdly__edge_cnn_top__DOT__conv3__DOT__controller__DOT__state = 2U;
        } else {
            vlSelfRef.edge_cnn_top__DOT__conv3__DOT__controller_busy = 0U;
            if (vlSelfRef.conv3_start) {
                vlSelfRef.edge_cnn_top__DOT__conv3__DOT__out_row = 0U;
                vlSelfRef.edge_cnn_top__DOT__conv3__DOT__out_col = 0U;
                vlSelfRef.edge_cnn_top__DOT__conv3__DOT__out_channel = 0U;
                vlSelfRef.edge_cnn_top__DOT__conv3__DOT__controller_busy = 1U;
                vlSelfRef.__Vdly__edge_cnn_top__DOT__conv3__DOT__controller__DOT__state = 1U;
            }
        }
        vlSelfRef.edge_cnn_top__DOT__conv4__DOT__engine_done = 0U;
        if ((2U & (IData)(vlSelfRef.edge_cnn_top__DOT__conv4__DOT__engine__DOT__state))) {
            if ((1U & (IData)(vlSelfRef.edge_cnn_top__DOT__conv4__DOT__engine__DOT__state))) {
                vlSelfRef.edge_cnn_top__DOT__conv4__DOT__engine_done = 1U;
                vlSelfRef.__Vdly__edge_cnn_top__DOT__conv4__DOT__engine__DOT__state = 0U;
            } else {
                if ((1U & (~ (IData)(vlSelfRef.edge_cnn_top__DOT__conv4__DOT__engine__DOT__last_item)))) {
                    if ((2U != (IData)(vlSelfRef.edge_cnn_top__DOT__conv4__DOT__engine__DOT__kc))) {
                        vlSelfRef.edge_cnn_top__DOT__conv4__DOT__engine__DOT__kc 
                            = (3U & ((IData)(1U) + (IData)(vlSelfRef.edge_cnn_top__DOT__conv4__DOT__engine__DOT__kc)));
                    } else {
                        if ((2U != (IData)(vlSelfRef.edge_cnn_top__DOT__conv4__DOT__engine__DOT__kr))) {
                            vlSelfRef.edge_cnn_top__DOT__conv4__DOT__engine__DOT__kr 
                                = (3U & ((IData)(1U) 
                                         + (IData)(vlSelfRef.edge_cnn_top__DOT__conv4__DOT__engine__DOT__kr)));
                        } else {
                            vlSelfRef.__Vdly__edge_cnn_top__DOT__conv4__DOT__engine__DOT__ic 
                                = (0x0000ffffU & ((IData)(1U) 
                                                  + (IData)(vlSelfRef.edge_cnn_top__DOT__conv4__DOT__engine__DOT__ic)));
                            vlSelfRef.edge_cnn_top__DOT__conv4__DOT__engine__DOT__kr = 0U;
                        }
                        vlSelfRef.edge_cnn_top__DOT__conv4__DOT__engine__DOT__kc = 0U;
                    }
                }
                vlSelfRef.__Vdly__edge_cnn_top__DOT__conv4__DOT__engine__DOT__product_reg 
                    = (0x0000ffffU & (VL_MULS_III(16, 
                                                  (0x0000ffffU 
                                                   & VL_EXTENDS_II(16,8, (IData)(vlSelfRef.edge_cnn_top__DOT__buf_b_rdata))), 
                                                  (0x0000ffffU 
                                                   & VL_EXTENDS_II(16,8, (IData)(vlSelfRef.edge_cnn_top__DOT__conv4_weight_data)))) 
                                      & (- (IData)((IData)(vlSelfRef.edge_cnn_top__DOT__conv4__DOT__engine__DOT__valid_pixel)))));
                vlSelfRef.__Vdly__edge_cnn_top__DOT__conv4__DOT__engine__DOT__product_valid = 1U;
                vlSelfRef.__Vdly__edge_cnn_top__DOT__conv4__DOT__engine__DOT__product_last 
                    = vlSelfRef.edge_cnn_top__DOT__conv4__DOT__engine__DOT__last_item;
                vlSelfRef.__Vdly__edge_cnn_top__DOT__conv4__DOT__engine__DOT__state = 1U;
            }
        } else if ((1U & (IData)(vlSelfRef.edge_cnn_top__DOT__conv4__DOT__engine__DOT__state))) {
            if (vlSelfRef.edge_cnn_top__DOT__conv4__DOT__engine__DOT__product_valid) {
                vlSelfRef.edge_cnn_top__DOT__conv4__DOT__raw_sum 
                    = (vlSelfRef.edge_cnn_top__DOT__conv4__DOT__raw_sum 
                       + (((- (IData)((1U & ((IData)(vlSelfRef.edge_cnn_top__DOT__conv4__DOT__engine__DOT__product_reg) 
                                             >> 0x0fU)))) 
                           << 0x00000010U) | (IData)(vlSelfRef.edge_cnn_top__DOT__conv4__DOT__engine__DOT__product_reg)));
                if (vlSelfRef.edge_cnn_top__DOT__conv4__DOT__engine__DOT__product_last) {
                    vlSelfRef.__Vdly__edge_cnn_top__DOT__conv4__DOT__engine__DOT__product_valid = 0U;
                    vlSelfRef.__Vdly__edge_cnn_top__DOT__conv4__DOT__engine__DOT__state = 3U;
                } else {
                    vlSelfRef.__Vdly__edge_cnn_top__DOT__conv4__DOT__engine__DOT__state = 2U;
                }
            } else {
                vlSelfRef.__Vdly__edge_cnn_top__DOT__conv4__DOT__engine__DOT__state = 2U;
            }
        } else if (vlSelfRef.edge_cnn_top__DOT__conv4__DOT__engine_start) {
            vlSelfRef.__Vdly__edge_cnn_top__DOT__conv4__DOT__engine__DOT__ic = 0U;
            vlSelfRef.edge_cnn_top__DOT__conv4__DOT__engine__DOT__kr = 0U;
            vlSelfRef.edge_cnn_top__DOT__conv4__DOT__engine__DOT__kc = 0U;
            vlSelfRef.edge_cnn_top__DOT__conv4__DOT__raw_sum = 0U;
            vlSelfRef.__Vdly__edge_cnn_top__DOT__conv4__DOT__engine__DOT__product_valid = 0U;
            vlSelfRef.__Vdly__edge_cnn_top__DOT__conv4__DOT__engine__DOT__product_last = 0U;
            vlSelfRef.__Vdly__edge_cnn_top__DOT__conv4__DOT__engine__DOT__state = 1U;
        }
        vlSelfRef.edge_cnn_top__DOT__conv1__DOT__engine_done = 0U;
        if ((2U & (IData)(vlSelfRef.edge_cnn_top__DOT__conv1__DOT__engine__DOT__state))) {
            if ((1U & (IData)(vlSelfRef.edge_cnn_top__DOT__conv1__DOT__engine__DOT__state))) {
                vlSelfRef.edge_cnn_top__DOT__conv1__DOT__engine_done = 1U;
                vlSelfRef.__Vdly__edge_cnn_top__DOT__conv1__DOT__engine__DOT__state = 0U;
            } else {
                if ((1U & (~ (IData)(vlSelfRef.edge_cnn_top__DOT__conv1__DOT__engine__DOT__last_item)))) {
                    if ((2U != (IData)(vlSelfRef.edge_cnn_top__DOT__conv1__DOT__engine__DOT__kc))) {
                        vlSelfRef.edge_cnn_top__DOT__conv1__DOT__engine__DOT__kc 
                            = (3U & ((IData)(1U) + (IData)(vlSelfRef.edge_cnn_top__DOT__conv1__DOT__engine__DOT__kc)));
                    } else {
                        if ((2U != (IData)(vlSelfRef.edge_cnn_top__DOT__conv1__DOT__engine__DOT__kr))) {
                            vlSelfRef.edge_cnn_top__DOT__conv1__DOT__engine__DOT__kr 
                                = (3U & ((IData)(1U) 
                                         + (IData)(vlSelfRef.edge_cnn_top__DOT__conv1__DOT__engine__DOT__kr)));
                        } else {
                            vlSelfRef.__Vdly__edge_cnn_top__DOT__conv1__DOT__engine__DOT__ic 
                                = (0x0000ffffU & ((IData)(1U) 
                                                  + (IData)(vlSelfRef.edge_cnn_top__DOT__conv1__DOT__engine__DOT__ic)));
                            vlSelfRef.edge_cnn_top__DOT__conv1__DOT__engine__DOT__kr = 0U;
                        }
                        vlSelfRef.edge_cnn_top__DOT__conv1__DOT__engine__DOT__kc = 0U;
                    }
                }
                vlSelfRef.__Vdly__edge_cnn_top__DOT__conv1__DOT__engine__DOT__product_reg 
                    = (0x0000ffffU & (VL_MULS_III(16, 
                                                  (0x0000ffffU 
                                                   & VL_EXTENDS_II(16,8, (IData)(vlSelfRef.edge_cnn_top__DOT__buf_a_rdata))), 
                                                  (0x0000ffffU 
                                                   & VL_EXTENDS_II(16,8, (IData)(vlSelfRef.edge_cnn_top__DOT__conv1_weight_data)))) 
                                      & (- (IData)((IData)(vlSelfRef.edge_cnn_top__DOT__conv1__DOT__engine__DOT__valid_pixel)))));
                vlSelfRef.__Vdly__edge_cnn_top__DOT__conv1__DOT__engine__DOT__product_valid = 1U;
                vlSelfRef.__Vdly__edge_cnn_top__DOT__conv1__DOT__engine__DOT__product_last 
                    = vlSelfRef.edge_cnn_top__DOT__conv1__DOT__engine__DOT__last_item;
                vlSelfRef.__Vdly__edge_cnn_top__DOT__conv1__DOT__engine__DOT__state = 1U;
            }
        } else if ((1U & (IData)(vlSelfRef.edge_cnn_top__DOT__conv1__DOT__engine__DOT__state))) {
            if (vlSelfRef.edge_cnn_top__DOT__conv1__DOT__engine__DOT__product_valid) {
                vlSelfRef.edge_cnn_top__DOT__conv1__DOT__raw_sum 
                    = (vlSelfRef.edge_cnn_top__DOT__conv1__DOT__raw_sum 
                       + (((- (IData)((1U & ((IData)(vlSelfRef.edge_cnn_top__DOT__conv1__DOT__engine__DOT__product_reg) 
                                             >> 0x0fU)))) 
                           << 0x00000010U) | (IData)(vlSelfRef.edge_cnn_top__DOT__conv1__DOT__engine__DOT__product_reg)));
                if (vlSelfRef.edge_cnn_top__DOT__conv1__DOT__engine__DOT__product_last) {
                    vlSelfRef.__Vdly__edge_cnn_top__DOT__conv1__DOT__engine__DOT__product_valid = 0U;
                    vlSelfRef.__Vdly__edge_cnn_top__DOT__conv1__DOT__engine__DOT__state = 3U;
                } else {
                    vlSelfRef.__Vdly__edge_cnn_top__DOT__conv1__DOT__engine__DOT__state = 2U;
                }
            } else {
                vlSelfRef.__Vdly__edge_cnn_top__DOT__conv1__DOT__engine__DOT__state = 2U;
            }
        } else if (vlSelfRef.edge_cnn_top__DOT__conv1__DOT__engine_start) {
            vlSelfRef.__Vdly__edge_cnn_top__DOT__conv1__DOT__engine__DOT__ic = 0U;
            vlSelfRef.edge_cnn_top__DOT__conv1__DOT__engine__DOT__kr = 0U;
            vlSelfRef.edge_cnn_top__DOT__conv1__DOT__engine__DOT__kc = 0U;
            vlSelfRef.edge_cnn_top__DOT__conv1__DOT__raw_sum = 0U;
            vlSelfRef.__Vdly__edge_cnn_top__DOT__conv1__DOT__engine__DOT__product_valid = 0U;
            vlSelfRef.__Vdly__edge_cnn_top__DOT__conv1__DOT__engine__DOT__product_last = 0U;
            vlSelfRef.__Vdly__edge_cnn_top__DOT__conv1__DOT__engine__DOT__state = 1U;
        }
        vlSelfRef.edge_cnn_top__DOT__conv2__DOT__engine_done = 0U;
        if ((2U & (IData)(vlSelfRef.edge_cnn_top__DOT__conv2__DOT__engine__DOT__state))) {
            if ((1U & (IData)(vlSelfRef.edge_cnn_top__DOT__conv2__DOT__engine__DOT__state))) {
                vlSelfRef.edge_cnn_top__DOT__conv2__DOT__engine_done = 1U;
                vlSelfRef.__Vdly__edge_cnn_top__DOT__conv2__DOT__engine__DOT__state = 0U;
            } else {
                if ((1U & (~ (IData)(vlSelfRef.edge_cnn_top__DOT__conv2__DOT__engine__DOT__last_item)))) {
                    if ((2U != (IData)(vlSelfRef.edge_cnn_top__DOT__conv2__DOT__engine__DOT__kc))) {
                        vlSelfRef.edge_cnn_top__DOT__conv2__DOT__engine__DOT__kc 
                            = (3U & ((IData)(1U) + (IData)(vlSelfRef.edge_cnn_top__DOT__conv2__DOT__engine__DOT__kc)));
                    } else {
                        if ((2U != (IData)(vlSelfRef.edge_cnn_top__DOT__conv2__DOT__engine__DOT__kr))) {
                            vlSelfRef.edge_cnn_top__DOT__conv2__DOT__engine__DOT__kr 
                                = (3U & ((IData)(1U) 
                                         + (IData)(vlSelfRef.edge_cnn_top__DOT__conv2__DOT__engine__DOT__kr)));
                        } else {
                            vlSelfRef.__Vdly__edge_cnn_top__DOT__conv2__DOT__engine__DOT__ic 
                                = (0x0000ffffU & ((IData)(1U) 
                                                  + (IData)(vlSelfRef.edge_cnn_top__DOT__conv2__DOT__engine__DOT__ic)));
                            vlSelfRef.edge_cnn_top__DOT__conv2__DOT__engine__DOT__kr = 0U;
                        }
                        vlSelfRef.edge_cnn_top__DOT__conv2__DOT__engine__DOT__kc = 0U;
                    }
                }
                vlSelfRef.__Vdly__edge_cnn_top__DOT__conv2__DOT__engine__DOT__product_reg 
                    = (0x0000ffffU & (VL_MULS_III(16, 
                                                  (0x0000ffffU 
                                                   & VL_EXTENDS_II(16,8, (IData)(vlSelfRef.edge_cnn_top__DOT__buf_a_rdata))), 
                                                  (0x0000ffffU 
                                                   & VL_EXTENDS_II(16,8, (IData)(vlSelfRef.edge_cnn_top__DOT__conv2_weight_data)))) 
                                      & (- (IData)((IData)(vlSelfRef.edge_cnn_top__DOT__conv2__DOT__engine__DOT__valid_pixel)))));
                vlSelfRef.__Vdly__edge_cnn_top__DOT__conv2__DOT__engine__DOT__product_valid = 1U;
                vlSelfRef.__Vdly__edge_cnn_top__DOT__conv2__DOT__engine__DOT__product_last 
                    = vlSelfRef.edge_cnn_top__DOT__conv2__DOT__engine__DOT__last_item;
                vlSelfRef.__Vdly__edge_cnn_top__DOT__conv2__DOT__engine__DOT__state = 1U;
            }
        } else if ((1U & (IData)(vlSelfRef.edge_cnn_top__DOT__conv2__DOT__engine__DOT__state))) {
            if (vlSelfRef.edge_cnn_top__DOT__conv2__DOT__engine__DOT__product_valid) {
                vlSelfRef.edge_cnn_top__DOT__conv2__DOT__raw_sum 
                    = (vlSelfRef.edge_cnn_top__DOT__conv2__DOT__raw_sum 
                       + (((- (IData)((1U & ((IData)(vlSelfRef.edge_cnn_top__DOT__conv2__DOT__engine__DOT__product_reg) 
                                             >> 0x0fU)))) 
                           << 0x00000010U) | (IData)(vlSelfRef.edge_cnn_top__DOT__conv2__DOT__engine__DOT__product_reg)));
                if (vlSelfRef.edge_cnn_top__DOT__conv2__DOT__engine__DOT__product_last) {
                    vlSelfRef.__Vdly__edge_cnn_top__DOT__conv2__DOT__engine__DOT__product_valid = 0U;
                    vlSelfRef.__Vdly__edge_cnn_top__DOT__conv2__DOT__engine__DOT__state = 3U;
                } else {
                    vlSelfRef.__Vdly__edge_cnn_top__DOT__conv2__DOT__engine__DOT__state = 2U;
                }
            } else {
                vlSelfRef.__Vdly__edge_cnn_top__DOT__conv2__DOT__engine__DOT__state = 2U;
            }
        } else if (vlSelfRef.edge_cnn_top__DOT__conv2__DOT__engine_start) {
            vlSelfRef.__Vdly__edge_cnn_top__DOT__conv2__DOT__engine__DOT__ic = 0U;
            vlSelfRef.edge_cnn_top__DOT__conv2__DOT__engine__DOT__kr = 0U;
            vlSelfRef.edge_cnn_top__DOT__conv2__DOT__engine__DOT__kc = 0U;
            vlSelfRef.edge_cnn_top__DOT__conv2__DOT__raw_sum = 0U;
            vlSelfRef.__Vdly__edge_cnn_top__DOT__conv2__DOT__engine__DOT__product_valid = 0U;
            vlSelfRef.__Vdly__edge_cnn_top__DOT__conv2__DOT__engine__DOT__product_last = 0U;
            vlSelfRef.__Vdly__edge_cnn_top__DOT__conv2__DOT__engine__DOT__state = 1U;
        }
        vlSelfRef.edge_cnn_top__DOT__conv3__DOT__engine_done = 0U;
        if ((2U & (IData)(vlSelfRef.edge_cnn_top__DOT__conv3__DOT__engine__DOT__state))) {
            if ((1U & (IData)(vlSelfRef.edge_cnn_top__DOT__conv3__DOT__engine__DOT__state))) {
                vlSelfRef.edge_cnn_top__DOT__conv3__DOT__engine_done = 1U;
                vlSelfRef.__Vdly__edge_cnn_top__DOT__conv3__DOT__engine__DOT__state = 0U;
            } else {
                if ((1U & (~ (IData)(vlSelfRef.edge_cnn_top__DOT__conv3__DOT__engine__DOT__last_item)))) {
                    if ((2U != (IData)(vlSelfRef.edge_cnn_top__DOT__conv3__DOT__engine__DOT__kc))) {
                        vlSelfRef.edge_cnn_top__DOT__conv3__DOT__engine__DOT__kc 
                            = (3U & ((IData)(1U) + (IData)(vlSelfRef.edge_cnn_top__DOT__conv3__DOT__engine__DOT__kc)));
                    } else {
                        if ((2U != (IData)(vlSelfRef.edge_cnn_top__DOT__conv3__DOT__engine__DOT__kr))) {
                            vlSelfRef.edge_cnn_top__DOT__conv3__DOT__engine__DOT__kr 
                                = (3U & ((IData)(1U) 
                                         + (IData)(vlSelfRef.edge_cnn_top__DOT__conv3__DOT__engine__DOT__kr)));
                        } else {
                            vlSelfRef.__Vdly__edge_cnn_top__DOT__conv3__DOT__engine__DOT__ic 
                                = (0x0000ffffU & ((IData)(1U) 
                                                  + (IData)(vlSelfRef.edge_cnn_top__DOT__conv3__DOT__engine__DOT__ic)));
                            vlSelfRef.edge_cnn_top__DOT__conv3__DOT__engine__DOT__kr = 0U;
                        }
                        vlSelfRef.edge_cnn_top__DOT__conv3__DOT__engine__DOT__kc = 0U;
                    }
                }
                vlSelfRef.__Vdly__edge_cnn_top__DOT__conv3__DOT__engine__DOT__product_reg 
                    = (0x0000ffffU & (VL_MULS_III(16, 
                                                  (0x0000ffffU 
                                                   & VL_EXTENDS_II(16,8, (IData)(vlSelfRef.edge_cnn_top__DOT__buf_a_rdata))), 
                                                  (0x0000ffffU 
                                                   & VL_EXTENDS_II(16,8, (IData)(vlSelfRef.edge_cnn_top__DOT__conv3_weight_data)))) 
                                      & (- (IData)((IData)(vlSelfRef.edge_cnn_top__DOT__conv3__DOT__engine__DOT__valid_pixel)))));
                vlSelfRef.__Vdly__edge_cnn_top__DOT__conv3__DOT__engine__DOT__product_valid = 1U;
                vlSelfRef.__Vdly__edge_cnn_top__DOT__conv3__DOT__engine__DOT__product_last 
                    = vlSelfRef.edge_cnn_top__DOT__conv3__DOT__engine__DOT__last_item;
                vlSelfRef.__Vdly__edge_cnn_top__DOT__conv3__DOT__engine__DOT__state = 1U;
            }
        } else if ((1U & (IData)(vlSelfRef.edge_cnn_top__DOT__conv3__DOT__engine__DOT__state))) {
            if (vlSelfRef.edge_cnn_top__DOT__conv3__DOT__engine__DOT__product_valid) {
                vlSelfRef.edge_cnn_top__DOT__conv3__DOT__raw_sum 
                    = (vlSelfRef.edge_cnn_top__DOT__conv3__DOT__raw_sum 
                       + (((- (IData)((1U & ((IData)(vlSelfRef.edge_cnn_top__DOT__conv3__DOT__engine__DOT__product_reg) 
                                             >> 0x0fU)))) 
                           << 0x00000010U) | (IData)(vlSelfRef.edge_cnn_top__DOT__conv3__DOT__engine__DOT__product_reg)));
                if (vlSelfRef.edge_cnn_top__DOT__conv3__DOT__engine__DOT__product_last) {
                    vlSelfRef.__Vdly__edge_cnn_top__DOT__conv3__DOT__engine__DOT__product_valid = 0U;
                    vlSelfRef.__Vdly__edge_cnn_top__DOT__conv3__DOT__engine__DOT__state = 3U;
                } else {
                    vlSelfRef.__Vdly__edge_cnn_top__DOT__conv3__DOT__engine__DOT__state = 2U;
                }
            } else {
                vlSelfRef.__Vdly__edge_cnn_top__DOT__conv3__DOT__engine__DOT__state = 2U;
            }
        } else if (vlSelfRef.edge_cnn_top__DOT__conv3__DOT__engine_start) {
            vlSelfRef.__Vdly__edge_cnn_top__DOT__conv3__DOT__engine__DOT__ic = 0U;
            vlSelfRef.edge_cnn_top__DOT__conv3__DOT__engine__DOT__kr = 0U;
            vlSelfRef.edge_cnn_top__DOT__conv3__DOT__engine__DOT__kc = 0U;
            vlSelfRef.edge_cnn_top__DOT__conv3__DOT__raw_sum = 0U;
            vlSelfRef.__Vdly__edge_cnn_top__DOT__conv3__DOT__engine__DOT__product_valid = 0U;
            vlSelfRef.__Vdly__edge_cnn_top__DOT__conv3__DOT__engine__DOT__product_last = 0U;
            vlSelfRef.__Vdly__edge_cnn_top__DOT__conv3__DOT__engine__DOT__state = 1U;
        }
    } else {
        vlSelfRef.edge_cnn_top__DOT__conv4__DOT__biased_sum_reg = 0U;
        vlSelfRef.__Vdly__edge_cnn_top__DOT__conv4__DOT__controller__DOT__state = 0U;
        vlSelfRef.edge_cnn_top__DOT__conv4__DOT__out_row = 0U;
        vlSelfRef.edge_cnn_top__DOT__conv4__DOT__out_col = 0U;
        vlSelfRef.edge_cnn_top__DOT__conv4__DOT__out_channel = 0U;
        vlSelfRef.__Vdly__edge_cnn_top__DOT__conv4__DOT__engine_start = 0U;
        vlSelfRef.edge_cnn_top__DOT__conv4__DOT__controller_out_addr = 0U;
        vlSelfRef.edge_cnn_top__DOT__conv4__DOT__controller_busy = 0U;
        vlSelfRef.edge_cnn_top__DOT__conv4__DOT__controller_done = 0U;
        vlSelfRef.edge_cnn_top__DOT__conv1__DOT__biased_sum_reg = 0U;
        vlSelfRef.__Vdly__edge_cnn_top__DOT__conv1__DOT__controller__DOT__state = 0U;
        vlSelfRef.edge_cnn_top__DOT__conv1__DOT__out_row = 0U;
        vlSelfRef.edge_cnn_top__DOT__conv1__DOT__out_col = 0U;
        vlSelfRef.edge_cnn_top__DOT__conv1__DOT__out_channel = 0U;
        vlSelfRef.__Vdly__edge_cnn_top__DOT__conv1__DOT__engine_start = 0U;
        vlSelfRef.edge_cnn_top__DOT__conv1__DOT__controller_out_addr = 0U;
        vlSelfRef.edge_cnn_top__DOT__conv1__DOT__controller_busy = 0U;
        vlSelfRef.edge_cnn_top__DOT__conv1__DOT__controller_done = 0U;
        vlSelfRef.edge_cnn_top__DOT__conv2__DOT__biased_sum_reg = 0U;
        vlSelfRef.__Vdly__edge_cnn_top__DOT__conv2__DOT__controller__DOT__state = 0U;
        vlSelfRef.edge_cnn_top__DOT__conv2__DOT__out_row = 0U;
        vlSelfRef.edge_cnn_top__DOT__conv2__DOT__out_col = 0U;
        vlSelfRef.edge_cnn_top__DOT__conv2__DOT__out_channel = 0U;
        vlSelfRef.__Vdly__edge_cnn_top__DOT__conv2__DOT__engine_start = 0U;
        vlSelfRef.edge_cnn_top__DOT__conv2__DOT__controller_out_addr = 0U;
        vlSelfRef.edge_cnn_top__DOT__conv2__DOT__controller_busy = 0U;
        vlSelfRef.edge_cnn_top__DOT__conv2__DOT__controller_done = 0U;
        vlSelfRef.edge_cnn_top__DOT__conv3__DOT__biased_sum_reg = 0U;
        vlSelfRef.__Vdly__edge_cnn_top__DOT__conv3__DOT__controller__DOT__state = 0U;
        vlSelfRef.edge_cnn_top__DOT__conv3__DOT__out_row = 0U;
        vlSelfRef.edge_cnn_top__DOT__conv3__DOT__out_col = 0U;
        vlSelfRef.edge_cnn_top__DOT__conv3__DOT__out_channel = 0U;
        vlSelfRef.__Vdly__edge_cnn_top__DOT__conv3__DOT__engine_start = 0U;
        vlSelfRef.edge_cnn_top__DOT__conv3__DOT__controller_out_addr = 0U;
        vlSelfRef.edge_cnn_top__DOT__conv3__DOT__controller_busy = 0U;
        vlSelfRef.edge_cnn_top__DOT__conv3__DOT__controller_done = 0U;
        vlSelfRef.__Vdly__edge_cnn_top__DOT__conv4__DOT__engine__DOT__ic = 0U;
        vlSelfRef.edge_cnn_top__DOT__conv4__DOT__engine__DOT__kr = 0U;
        vlSelfRef.edge_cnn_top__DOT__conv4__DOT__engine__DOT__kc = 0U;
        vlSelfRef.edge_cnn_top__DOT__conv4__DOT__raw_sum = 0U;
        vlSelfRef.__Vdly__edge_cnn_top__DOT__conv4__DOT__engine__DOT__state = 0U;
        vlSelfRef.__Vdly__edge_cnn_top__DOT__conv4__DOT__engine__DOT__product_reg = 0U;
        vlSelfRef.__Vdly__edge_cnn_top__DOT__conv4__DOT__engine__DOT__product_valid = 0U;
        vlSelfRef.__Vdly__edge_cnn_top__DOT__conv4__DOT__engine__DOT__product_last = 0U;
        vlSelfRef.edge_cnn_top__DOT__conv4__DOT__engine_done = 0U;
        vlSelfRef.__Vdly__edge_cnn_top__DOT__conv1__DOT__engine__DOT__ic = 0U;
        vlSelfRef.edge_cnn_top__DOT__conv1__DOT__engine__DOT__kr = 0U;
        vlSelfRef.edge_cnn_top__DOT__conv1__DOT__engine__DOT__kc = 0U;
        vlSelfRef.edge_cnn_top__DOT__conv1__DOT__raw_sum = 0U;
        vlSelfRef.__Vdly__edge_cnn_top__DOT__conv1__DOT__engine__DOT__state = 0U;
        vlSelfRef.__Vdly__edge_cnn_top__DOT__conv1__DOT__engine__DOT__product_reg = 0U;
        vlSelfRef.__Vdly__edge_cnn_top__DOT__conv1__DOT__engine__DOT__product_valid = 0U;
        vlSelfRef.__Vdly__edge_cnn_top__DOT__conv1__DOT__engine__DOT__product_last = 0U;
        vlSelfRef.edge_cnn_top__DOT__conv1__DOT__engine_done = 0U;
        vlSelfRef.__Vdly__edge_cnn_top__DOT__conv2__DOT__engine__DOT__ic = 0U;
        vlSelfRef.edge_cnn_top__DOT__conv2__DOT__engine__DOT__kr = 0U;
        vlSelfRef.edge_cnn_top__DOT__conv2__DOT__engine__DOT__kc = 0U;
        vlSelfRef.edge_cnn_top__DOT__conv2__DOT__raw_sum = 0U;
        vlSelfRef.__Vdly__edge_cnn_top__DOT__conv2__DOT__engine__DOT__state = 0U;
        vlSelfRef.__Vdly__edge_cnn_top__DOT__conv2__DOT__engine__DOT__product_reg = 0U;
        vlSelfRef.__Vdly__edge_cnn_top__DOT__conv2__DOT__engine__DOT__product_valid = 0U;
        vlSelfRef.__Vdly__edge_cnn_top__DOT__conv2__DOT__engine__DOT__product_last = 0U;
        vlSelfRef.edge_cnn_top__DOT__conv2__DOT__engine_done = 0U;
        vlSelfRef.__Vdly__edge_cnn_top__DOT__conv3__DOT__engine__DOT__ic = 0U;
        vlSelfRef.edge_cnn_top__DOT__conv3__DOT__engine__DOT__kr = 0U;
        vlSelfRef.edge_cnn_top__DOT__conv3__DOT__engine__DOT__kc = 0U;
        vlSelfRef.edge_cnn_top__DOT__conv3__DOT__raw_sum = 0U;
        vlSelfRef.__Vdly__edge_cnn_top__DOT__conv3__DOT__engine__DOT__state = 0U;
        vlSelfRef.__Vdly__edge_cnn_top__DOT__conv3__DOT__engine__DOT__product_reg = 0U;
        vlSelfRef.__Vdly__edge_cnn_top__DOT__conv3__DOT__engine__DOT__product_valid = 0U;
        vlSelfRef.__Vdly__edge_cnn_top__DOT__conv3__DOT__engine__DOT__product_last = 0U;
        vlSelfRef.edge_cnn_top__DOT__conv3__DOT__engine_done = 0U;
    }
    edge_cnn_top__DOT__conv4__DOT__quant__DOT__result43 
        = (0x000007ffffffffffULL & ((1U & (IData)((vlSelfRef.edge_cnn_top__DOT__conv4__DOT__product_reg 
                                                   >> 0x0000002aU)))
                                     ? (- edge_cnn_top__DOT__conv4__DOT__quant__DOT__gen_round_shift__DOT__shifted_magnitude)
                                     : edge_cnn_top__DOT__conv4__DOT__quant__DOT__gen_round_shift__DOT__shifted_magnitude));
    vlSelfRef.conv4_start = vlSelfRef.__Vdly__conv4_start;
    vlSelfRef.edge_cnn_top__DOT__conv4__DOT__controller__DOT__state 
        = vlSelfRef.__Vdly__edge_cnn_top__DOT__conv4__DOT__controller__DOT__state;
    edge_cnn_top__DOT__conv1__DOT__quant__DOT__result43 
        = (0x000007ffffffffffULL & ((1U & (IData)((vlSelfRef.edge_cnn_top__DOT__conv1__DOT__product_reg 
                                                   >> 0x0000002aU)))
                                     ? (- edge_cnn_top__DOT__conv1__DOT__quant__DOT__gen_round_shift__DOT__shifted_magnitude)
                                     : edge_cnn_top__DOT__conv1__DOT__quant__DOT__gen_round_shift__DOT__shifted_magnitude));
    vlSelfRef.conv1_start = vlSelfRef.__Vdly__conv1_start;
    vlSelfRef.edge_cnn_top__DOT__conv1__DOT__controller__DOT__state 
        = vlSelfRef.__Vdly__edge_cnn_top__DOT__conv1__DOT__controller__DOT__state;
    edge_cnn_top__DOT__conv2__DOT__quant__DOT__result43 
        = (0x000007ffffffffffULL & ((1U & (IData)((vlSelfRef.edge_cnn_top__DOT__conv2__DOT__product_reg 
                                                   >> 0x0000002aU)))
                                     ? (- edge_cnn_top__DOT__conv2__DOT__quant__DOT__gen_round_shift__DOT__shifted_magnitude)
                                     : edge_cnn_top__DOT__conv2__DOT__quant__DOT__gen_round_shift__DOT__shifted_magnitude));
    vlSelfRef.conv2_start = vlSelfRef.__Vdly__conv2_start;
    vlSelfRef.edge_cnn_top__DOT__conv2__DOT__controller__DOT__state 
        = vlSelfRef.__Vdly__edge_cnn_top__DOT__conv2__DOT__controller__DOT__state;
    edge_cnn_top__DOT__conv3__DOT__quant__DOT__result43 
        = (0x000007ffffffffffULL & ((1U & (IData)((vlSelfRef.edge_cnn_top__DOT__conv3__DOT__product_reg 
                                                   >> 0x0000002aU)))
                                     ? (- edge_cnn_top__DOT__conv3__DOT__quant__DOT__gen_round_shift__DOT__shifted_magnitude)
                                     : edge_cnn_top__DOT__conv3__DOT__quant__DOT__gen_round_shift__DOT__shifted_magnitude));
    vlSelfRef.conv3_start = vlSelfRef.__Vdly__conv3_start;
    vlSelfRef.edge_cnn_top__DOT__conv3__DOT__controller__DOT__state 
        = vlSelfRef.__Vdly__edge_cnn_top__DOT__conv3__DOT__controller__DOT__state;
    vlSelfRef.edge_cnn_top__DOT__conv4__DOT__activated 
        = ((IData)(edge_cnn_top__DOT__conv4__DOT__quant__DOT__result43) 
           & (- (IData)((1U & (~ (IData)((edge_cnn_top__DOT__conv4__DOT__quant__DOT__result43 
                                          >> 0x0000001fU)))))));
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
    vlSelfRef.__VdfgRegularize_h6e95ff9d_0_10 = ((IData)(vlSelfRef.edge_cnn_top__DOT__conv3__DOT__out_channel) 
                                                 << 6U);
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
    vlSelfRef.edge_cnn_top__DOT__conv4__DOT__engine_start 
        = vlSelfRef.__Vdly__edge_cnn_top__DOT__conv4__DOT__engine_start;
    vlSelfRef.edge_cnn_top__DOT__conv4__DOT__engine__DOT__state 
        = vlSelfRef.__Vdly__edge_cnn_top__DOT__conv4__DOT__engine__DOT__state;
    vlSelfRef.edge_cnn_top__DOT__conv4__DOT__engine__DOT__product_reg 
        = vlSelfRef.__Vdly__edge_cnn_top__DOT__conv4__DOT__engine__DOT__product_reg;
    vlSelfRef.edge_cnn_top__DOT__conv4__DOT__engine__DOT__product_valid 
        = vlSelfRef.__Vdly__edge_cnn_top__DOT__conv4__DOT__engine__DOT__product_valid;
    vlSelfRef.edge_cnn_top__DOT__conv4__DOT__engine__DOT__product_last 
        = vlSelfRef.__Vdly__edge_cnn_top__DOT__conv4__DOT__engine__DOT__product_last;
    vlSelfRef.edge_cnn_top__DOT__conv1__DOT__engine_start 
        = vlSelfRef.__Vdly__edge_cnn_top__DOT__conv1__DOT__engine_start;
    vlSelfRef.edge_cnn_top__DOT__conv1__DOT__engine__DOT__state 
        = vlSelfRef.__Vdly__edge_cnn_top__DOT__conv1__DOT__engine__DOT__state;
    vlSelfRef.edge_cnn_top__DOT__conv1__DOT__engine__DOT__product_reg 
        = vlSelfRef.__Vdly__edge_cnn_top__DOT__conv1__DOT__engine__DOT__product_reg;
    vlSelfRef.edge_cnn_top__DOT__conv1__DOT__engine__DOT__product_valid 
        = vlSelfRef.__Vdly__edge_cnn_top__DOT__conv1__DOT__engine__DOT__product_valid;
    vlSelfRef.edge_cnn_top__DOT__conv1__DOT__engine__DOT__product_last 
        = vlSelfRef.__Vdly__edge_cnn_top__DOT__conv1__DOT__engine__DOT__product_last;
    vlSelfRef.edge_cnn_top__DOT__conv2__DOT__engine_start 
        = vlSelfRef.__Vdly__edge_cnn_top__DOT__conv2__DOT__engine_start;
    vlSelfRef.edge_cnn_top__DOT__conv2__DOT__engine__DOT__state 
        = vlSelfRef.__Vdly__edge_cnn_top__DOT__conv2__DOT__engine__DOT__state;
    vlSelfRef.edge_cnn_top__DOT__conv2__DOT__engine__DOT__product_reg 
        = vlSelfRef.__Vdly__edge_cnn_top__DOT__conv2__DOT__engine__DOT__product_reg;
    vlSelfRef.edge_cnn_top__DOT__conv2__DOT__engine__DOT__product_valid 
        = vlSelfRef.__Vdly__edge_cnn_top__DOT__conv2__DOT__engine__DOT__product_valid;
    vlSelfRef.edge_cnn_top__DOT__conv2__DOT__engine__DOT__product_last 
        = vlSelfRef.__Vdly__edge_cnn_top__DOT__conv2__DOT__engine__DOT__product_last;
    vlSelfRef.edge_cnn_top__DOT__conv3__DOT__engine_start 
        = vlSelfRef.__Vdly__edge_cnn_top__DOT__conv3__DOT__engine_start;
    vlSelfRef.edge_cnn_top__DOT__conv3__DOT__engine__DOT__state 
        = vlSelfRef.__Vdly__edge_cnn_top__DOT__conv3__DOT__engine__DOT__state;
    vlSelfRef.edge_cnn_top__DOT__conv3__DOT__engine__DOT__product_reg 
        = vlSelfRef.__Vdly__edge_cnn_top__DOT__conv3__DOT__engine__DOT__product_reg;
    vlSelfRef.edge_cnn_top__DOT__conv3__DOT__engine__DOT__product_valid 
        = vlSelfRef.__Vdly__edge_cnn_top__DOT__conv3__DOT__engine__DOT__product_valid;
    vlSelfRef.edge_cnn_top__DOT__conv3__DOT__engine__DOT__product_last 
        = vlSelfRef.__Vdly__edge_cnn_top__DOT__conv3__DOT__engine__DOT__product_last;
}

void Vedge_cnn_top___024root___nba_sequent__TOP__3(Vedge_cnn_top___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vedge_cnn_top___024root___nba_sequent__TOP__3\n"); );
    Vedge_cnn_top__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    vlSelfRef.edge_cnn_top__DOT__conv4_weight_data 
        = ((0x00047fffU >= vlSelfRef.edge_cnn_top__DOT__conv4_weight_addr)
            ? vlSelfRef.edge_cnn_top__DOT__conv4_weight_rom__DOT__rom
           [vlSelfRef.edge_cnn_top__DOT__conv4_weight_addr]
            : 0U);
    vlSelfRef.edge_cnn_top__DOT__buf_b_rdata = vlSelfRef.edge_cnn_top__DOT__buffer_b__DOT__mem
        [(0x00007fffU & ((2U == (IData)(vlSelfRef.state_dbg))
                          ? (VL_SHIFTL_III(15,32,32, (IData)(vlSelfRef.edge_cnn_top__DOT__pool1__DOT__ch), 0x0000000aU) 
                             + ((0x0000ffffU & (VL_SHIFTL_III(16,16,32, (IData)(vlSelfRef.edge_cnn_top__DOT__pool1__DOT__col), 1U) 
                                                + (1U 
                                                   & (IData)(vlSelfRef.edge_cnn_top__DOT__pool1__DOT__tap)))) 
                                + (0x001fffe0U & ((
                                                   VL_SHIFTL_III(16,16,32, (IData)(vlSelfRef.edge_cnn_top__DOT__pool1__DOT__row), 1U) 
                                                   + 
                                                   (1U 
                                                    & ((IData)(vlSelfRef.edge_cnn_top__DOT__pool1__DOT__tap) 
                                                       >> 1U))) 
                                                  << 5U))))
                          : ((4U == (IData)(vlSelfRef.state_dbg))
                              ? (VL_SHIFTL_III(15,32,32, (IData)(vlSelfRef.edge_cnn_top__DOT__pool2__DOT__ch), 8U) 
                                 + ((0x0000ffffU & 
                                     (VL_SHIFTL_III(16,16,32, (IData)(vlSelfRef.edge_cnn_top__DOT__pool2__DOT__col), 1U) 
                                      + (1U & (IData)(vlSelfRef.edge_cnn_top__DOT__pool2__DOT__tap)))) 
                                    + (0x000ffff0U 
                                       & ((VL_SHIFTL_III(16,16,32, (IData)(vlSelfRef.edge_cnn_top__DOT__pool2__DOT__row), 1U) 
                                           + (1U & 
                                              ((IData)(vlSelfRef.edge_cnn_top__DOT__pool2__DOT__tap) 
                                               >> 1U))) 
                                          << 4U))))
                              : ((6U == (IData)(vlSelfRef.state_dbg))
                                  ? (0x0007ffffU & 
                                     ((vlSelfRef.__VdfgRegularize_h6e95ff9d_0_9 
                                       + (VL_SHIFTL_III(15,32,32, (IData)(vlSelfRef.edge_cnn_top__DOT__conv4__DOT__engine__DOT__ic), 6U) 
                                          + VL_SHIFTL_III(15,32,32, vlSelfRef.__VdfgRegularize_h6e95ff9d_0_8, 3U))) 
                                      & (- (IData)((IData)(vlSelfRef.edge_cnn_top__DOT__conv4__DOT__engine__DOT__valid_pixel)))))
                                  : (0x0007ffffU & 
                                     ((IData)(vlSelfRef.edge_cnn_top__DOT__linear__DOT__feature_idx) 
                                      & (- (IData)(
                                                   (8U 
                                                    == (IData)(vlSelfRef.state_dbg))))))))))];
    vlSelfRef.edge_cnn_top__DOT__conv1_weight_data 
        = ((0x035fU >= (IData)(vlSelfRef.edge_cnn_top__DOT__conv1_weight_rom__DOT__rom_addr))
            ? vlSelfRef.edge_cnn_top__DOT__conv1_weight_rom__DOT__rom
           [vlSelfRef.edge_cnn_top__DOT__conv1_weight_rom__DOT__rom_addr]
            : 0U);
    vlSelfRef.edge_cnn_top__DOT__conv2_weight_data 
        = ((0x47ffU >= (IData)(vlSelfRef.edge_cnn_top__DOT__conv2_weight_rom__DOT__rom_addr))
            ? vlSelfRef.edge_cnn_top__DOT__conv2_weight_rom__DOT__rom
           [vlSelfRef.edge_cnn_top__DOT__conv2_weight_rom__DOT__rom_addr]
            : 0U);
    vlSelfRef.edge_cnn_top__DOT__conv3_weight_data 
        = ((0x00011fffU >= vlSelfRef.edge_cnn_top__DOT__conv3_weight_rom__DOT__rom_addr)
            ? vlSelfRef.edge_cnn_top__DOT__conv3_weight_rom__DOT__rom
           [vlSelfRef.edge_cnn_top__DOT__conv3_weight_rom__DOT__rom_addr]
            : 0U);
    vlSelfRef.edge_cnn_top__DOT__buf_a_rdata = vlSelfRef.edge_cnn_top__DOT__buffer_a__DOT__mem
        [(0x00007fffU & ((1U == (IData)(vlSelfRef.state_dbg))
                          ? (0x0007ffffU & ((vlSelfRef.__VdfgRegularize_h6e95ff9d_0_3 
                                             + (VL_SHIFTL_III(15,32,32, (IData)(vlSelfRef.edge_cnn_top__DOT__conv1__DOT__engine__DOT__ic), 0x0000000aU) 
                                                + VL_SHIFTL_III(15,32,32, vlSelfRef.__VdfgRegularize_h6e95ff9d_0_2, 5U))) 
                                            & (- (IData)((IData)(vlSelfRef.edge_cnn_top__DOT__conv1__DOT__engine__DOT__valid_pixel)))))
                          : ((3U == (IData)(vlSelfRef.state_dbg))
                              ? (0x0007ffffU & ((vlSelfRef.__VdfgRegularize_h6e95ff9d_0_5 
                                                 + 
                                                 (VL_SHIFTL_III(15,32,32, (IData)(vlSelfRef.edge_cnn_top__DOT__conv2__DOT__engine__DOT__ic), 8U) 
                                                  + 
                                                  VL_SHIFTL_III(15,32,32, vlSelfRef.__VdfgRegularize_h6e95ff9d_0_4, 4U))) 
                                                & (- (IData)((IData)(vlSelfRef.edge_cnn_top__DOT__conv2__DOT__engine__DOT__valid_pixel)))))
                              : ((5U == (IData)(vlSelfRef.state_dbg))
                                  ? (0x0007ffffU & 
                                     ((vlSelfRef.__VdfgRegularize_h6e95ff9d_0_7 
                                       + (VL_SHIFTL_III(15,32,32, (IData)(vlSelfRef.edge_cnn_top__DOT__conv3__DOT__engine__DOT__ic), 6U) 
                                          + VL_SHIFTL_III(15,32,32, vlSelfRef.__VdfgRegularize_h6e95ff9d_0_6, 3U))) 
                                      & (- (IData)((IData)(vlSelfRef.edge_cnn_top__DOT__conv3__DOT__engine__DOT__valid_pixel)))))
                                  : (0x0007ffffU & 
                                     ((VL_SHIFTL_III(15,32,32, (IData)(vlSelfRef.edge_cnn_top__DOT__gap__DOT__ch), 6U) 
                                       + (IData)(vlSelfRef.edge_cnn_top__DOT__gap__DOT__idx)) 
                                      & (- (IData)(
                                                   (7U 
                                                    == (IData)(vlSelfRef.state_dbg))))))))))];
    if (vlSelfRef.__VdlySet__edge_cnn_top__DOT__buffer_b__DOT__mem__v0) {
        vlSelfRef.edge_cnn_top__DOT__buffer_b__DOT__mem[vlSelfRef.__VdlyDim0__edge_cnn_top__DOT__buffer_b__DOT__mem__v0] 
            = vlSelfRef.__VdlyVal__edge_cnn_top__DOT__buffer_b__DOT__mem__v0;
    }
    if (vlSelfRef.__VdlySet__edge_cnn_top__DOT__buffer_a__DOT__mem__v0) {
        vlSelfRef.edge_cnn_top__DOT__buffer_a__DOT__mem[vlSelfRef.__VdlyDim0__edge_cnn_top__DOT__buffer_a__DOT__mem__v0] 
            = vlSelfRef.__VdlyVal__edge_cnn_top__DOT__buffer_a__DOT__mem__v0;
    }
}

void Vedge_cnn_top___024root___nba_sequent__TOP__4(Vedge_cnn_top___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vedge_cnn_top___024root___nba_sequent__TOP__4\n"); );
    Vedge_cnn_top__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    vlSelfRef.edge_cnn_top__DOT__pool1__DOT__ch = vlSelfRef.__Vdly__edge_cnn_top__DOT__pool1__DOT__ch;
    vlSelfRef.edge_cnn_top__DOT__pool1__DOT__col = vlSelfRef.__Vdly__edge_cnn_top__DOT__pool1__DOT__col;
    vlSelfRef.edge_cnn_top__DOT__pool1__DOT__tap = vlSelfRef.__Vdly__edge_cnn_top__DOT__pool1__DOT__tap;
    vlSelfRef.edge_cnn_top__DOT__pool1__DOT__row = vlSelfRef.__Vdly__edge_cnn_top__DOT__pool1__DOT__row;
    vlSelfRef.edge_cnn_top__DOT__pool2__DOT__ch = vlSelfRef.__Vdly__edge_cnn_top__DOT__pool2__DOT__ch;
    vlSelfRef.edge_cnn_top__DOT__pool2__DOT__col = vlSelfRef.__Vdly__edge_cnn_top__DOT__pool2__DOT__col;
    vlSelfRef.edge_cnn_top__DOT__pool2__DOT__tap = vlSelfRef.__Vdly__edge_cnn_top__DOT__pool2__DOT__tap;
    vlSelfRef.edge_cnn_top__DOT__pool2__DOT__row = vlSelfRef.__Vdly__edge_cnn_top__DOT__pool2__DOT__row;
    vlSelfRef.__VdfgRegularize_h6e95ff9d_0_9 = (((IData)(vlSelfRef.edge_cnn_top__DOT__conv4__DOT__engine__DOT__kc) 
                                                 + (IData)(vlSelfRef.edge_cnn_top__DOT__conv4__DOT__out_col)) 
                                                - (IData)(1U));
    vlSelfRef.__VdfgRegularize_h6e95ff9d_0_8 = (((IData)(vlSelfRef.edge_cnn_top__DOT__conv4__DOT__engine__DOT__kr) 
                                                 + (IData)(vlSelfRef.edge_cnn_top__DOT__conv4__DOT__out_row)) 
                                                - (IData)(1U));
    vlSelfRef.edge_cnn_top__DOT__linear__DOT__feature_idx 
        = vlSelfRef.__Vdly__edge_cnn_top__DOT__linear__DOT__feature_idx;
    vlSelfRef.edge_cnn_top__DOT__conv4__DOT__engine__DOT__ic 
        = vlSelfRef.__Vdly__edge_cnn_top__DOT__conv4__DOT__engine__DOT__ic;
    vlSelfRef.state_dbg = vlSelfRef.__Vdly__state_dbg;
    vlSelfRef.edge_cnn_top__DOT__gap__DOT__ch = vlSelfRef.__Vdly__edge_cnn_top__DOT__gap__DOT__ch;
    vlSelfRef.edge_cnn_top__DOT__gap__DOT__idx = vlSelfRef.__Vdly__edge_cnn_top__DOT__gap__DOT__idx;
    vlSelfRef.__VdfgRegularize_h6e95ff9d_0_3 = (((IData)(vlSelfRef.edge_cnn_top__DOT__conv1__DOT__engine__DOT__kc) 
                                                 + (IData)(vlSelfRef.edge_cnn_top__DOT__conv1__DOT__out_col)) 
                                                - (IData)(1U));
    vlSelfRef.__VdfgRegularize_h6e95ff9d_0_2 = (((IData)(vlSelfRef.edge_cnn_top__DOT__conv1__DOT__engine__DOT__kr) 
                                                 + (IData)(vlSelfRef.edge_cnn_top__DOT__conv1__DOT__out_row)) 
                                                - (IData)(1U));
    vlSelfRef.__VdfgRegularize_h6e95ff9d_0_5 = (((IData)(vlSelfRef.edge_cnn_top__DOT__conv2__DOT__engine__DOT__kc) 
                                                 + (IData)(vlSelfRef.edge_cnn_top__DOT__conv2__DOT__out_col)) 
                                                - (IData)(1U));
    vlSelfRef.__VdfgRegularize_h6e95ff9d_0_4 = (((IData)(vlSelfRef.edge_cnn_top__DOT__conv2__DOT__engine__DOT__kr) 
                                                 + (IData)(vlSelfRef.edge_cnn_top__DOT__conv2__DOT__out_row)) 
                                                - (IData)(1U));
    vlSelfRef.__VdfgRegularize_h6e95ff9d_0_7 = (((IData)(vlSelfRef.edge_cnn_top__DOT__conv3__DOT__engine__DOT__kc) 
                                                 + (IData)(vlSelfRef.edge_cnn_top__DOT__conv3__DOT__out_col)) 
                                                - (IData)(1U));
    vlSelfRef.__VdfgRegularize_h6e95ff9d_0_6 = (((IData)(vlSelfRef.edge_cnn_top__DOT__conv3__DOT__engine__DOT__kr) 
                                                 + (IData)(vlSelfRef.edge_cnn_top__DOT__conv3__DOT__out_row)) 
                                                - (IData)(1U));
    vlSelfRef.edge_cnn_top__DOT__conv1__DOT__engine__DOT__ic 
        = vlSelfRef.__Vdly__edge_cnn_top__DOT__conv1__DOT__engine__DOT__ic;
    vlSelfRef.edge_cnn_top__DOT__conv2__DOT__engine__DOT__ic 
        = vlSelfRef.__Vdly__edge_cnn_top__DOT__conv2__DOT__engine__DOT__ic;
    vlSelfRef.edge_cnn_top__DOT__conv3__DOT__engine__DOT__ic 
        = vlSelfRef.__Vdly__edge_cnn_top__DOT__conv3__DOT__engine__DOT__ic;
    vlSelfRef.edge_cnn_top__DOT__conv4__DOT__engine__DOT__valid_pixel 
        = (VL_LTES_III(32, 0U, vlSelfRef.__VdfgRegularize_h6e95ff9d_0_8) 
           & (VL_GTS_III(32, 8U, vlSelfRef.__VdfgRegularize_h6e95ff9d_0_8) 
              & (VL_LTES_III(32, 0U, vlSelfRef.__VdfgRegularize_h6e95ff9d_0_9) 
                 & VL_GTS_III(32, 8U, vlSelfRef.__VdfgRegularize_h6e95ff9d_0_9))));
    vlSelfRef.edge_cnn_top__DOT__linear_weight_rom__DOT__rom_addr 
        = (0x00000fffU & ((IData)(vlSelfRef.edge_cnn_top__DOT__linear__DOT__feature_idx) 
                          + ((IData)(vlSelfRef.edge_cnn_top__DOT__linear__DOT__class_idx) 
                             << 8U)));
    vlSelfRef.edge_cnn_top__DOT__conv4__DOT__engine__DOT__last_item 
        = ((0x007fU == (IData)(vlSelfRef.edge_cnn_top__DOT__conv4__DOT__engine__DOT__ic)) 
           & ((2U == (IData)(vlSelfRef.edge_cnn_top__DOT__conv4__DOT__engine__DOT__kr)) 
              & (2U == (IData)(vlSelfRef.edge_cnn_top__DOT__conv4__DOT__engine__DOT__kc))));
    vlSelfRef.edge_cnn_top__DOT__conv4_weight_addr 
        = (0x0007ffffU & ((IData)(vlSelfRef.edge_cnn_top__DOT__conv4__DOT__engine__DOT__kc) 
                          + (((IData)(9U) * VL_SHIFTL_III(32,32,32, (IData)(vlSelfRef.edge_cnn_top__DOT__conv4__DOT__out_channel), 7U)) 
                             + (((IData)(9U) * (IData)(vlSelfRef.edge_cnn_top__DOT__conv4__DOT__engine__DOT__ic)) 
                                + ((IData)(3U) * (IData)(vlSelfRef.edge_cnn_top__DOT__conv4__DOT__engine__DOT__kr))))));
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
    vlSelfRef.edge_cnn_top__DOT__conv1__DOT__engine__DOT__last_item 
        = ((2U == (IData)(vlSelfRef.edge_cnn_top__DOT__conv1__DOT__engine__DOT__ic)) 
           & ((2U == (IData)(vlSelfRef.edge_cnn_top__DOT__conv1__DOT__engine__DOT__kr)) 
              & (2U == (IData)(vlSelfRef.edge_cnn_top__DOT__conv1__DOT__engine__DOT__kc))));
    vlSelfRef.edge_cnn_top__DOT__conv1_weight_rom__DOT__rom_addr 
        = (0x000003ffU & ((IData)(vlSelfRef.edge_cnn_top__DOT__conv1__DOT__engine__DOT__kc) 
                          + (((IData)(0x0000001bU) 
                              * (IData)(vlSelfRef.edge_cnn_top__DOT__conv1__DOT__out_channel)) 
                             + (((IData)(9U) * (IData)(vlSelfRef.edge_cnn_top__DOT__conv1__DOT__engine__DOT__ic)) 
                                + ((IData)(3U) * (IData)(vlSelfRef.edge_cnn_top__DOT__conv1__DOT__engine__DOT__kr))))));
    vlSelfRef.edge_cnn_top__DOT__conv2__DOT__engine__DOT__last_item 
        = ((0x001fU == (IData)(vlSelfRef.edge_cnn_top__DOT__conv2__DOT__engine__DOT__ic)) 
           & ((2U == (IData)(vlSelfRef.edge_cnn_top__DOT__conv2__DOT__engine__DOT__kr)) 
              & (2U == (IData)(vlSelfRef.edge_cnn_top__DOT__conv2__DOT__engine__DOT__kc))));
    vlSelfRef.edge_cnn_top__DOT__conv2_weight_rom__DOT__rom_addr 
        = (0x00007fffU & ((IData)(vlSelfRef.edge_cnn_top__DOT__conv2__DOT__engine__DOT__kc) 
                          + (((IData)(9U) * VL_SHIFTL_III(32,32,32, (IData)(vlSelfRef.edge_cnn_top__DOT__conv2__DOT__out_channel), 5U)) 
                             + (((IData)(9U) * (IData)(vlSelfRef.edge_cnn_top__DOT__conv2__DOT__engine__DOT__ic)) 
                                + ((IData)(3U) * (IData)(vlSelfRef.edge_cnn_top__DOT__conv2__DOT__engine__DOT__kr))))));
    vlSelfRef.edge_cnn_top__DOT__conv3__DOT__engine__DOT__last_item 
        = ((0x003fU == (IData)(vlSelfRef.edge_cnn_top__DOT__conv3__DOT__engine__DOT__ic)) 
           & ((2U == (IData)(vlSelfRef.edge_cnn_top__DOT__conv3__DOT__engine__DOT__kr)) 
              & (2U == (IData)(vlSelfRef.edge_cnn_top__DOT__conv3__DOT__engine__DOT__kc))));
    vlSelfRef.edge_cnn_top__DOT__conv3_weight_rom__DOT__rom_addr 
        = (0x0001ffffU & ((IData)(vlSelfRef.edge_cnn_top__DOT__conv3__DOT__engine__DOT__kc) 
                          + (((IData)(9U) * ((IData)(vlSelfRef.edge_cnn_top__DOT__conv3__DOT__out_channel) 
                                             << 6U)) 
                             + (((IData)(9U) * (IData)(vlSelfRef.edge_cnn_top__DOT__conv3__DOT__engine__DOT__ic)) 
                                + ((IData)(3U) * (IData)(vlSelfRef.edge_cnn_top__DOT__conv3__DOT__engine__DOT__kr))))));
}

void Vedge_cnn_top___024root___eval_nba(Vedge_cnn_top___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vedge_cnn_top___024root___eval_nba\n"); );
    Vedge_cnn_top__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    if ((3ULL & vlSelfRef.__VnbaTriggered[0U])) {
        Vedge_cnn_top___024root___nba_sequent__TOP__0(vlSelf);
    }
    if ((1ULL & vlSelfRef.__VnbaTriggered[0U])) {
        vlSelfRef.__VdlySet__edge_cnn_top__DOT__buffer_a__DOT__mem__v0 = 0U;
        vlSelfRef.__VdlySet__edge_cnn_top__DOT__buffer_b__DOT__mem__v0 = 0U;
        if (((IData)(vlSelfRef.__VdfgRegularize_h6e95ff9d_0_0) 
             | ((IData)(vlSelfRef.edge_cnn_top__DOT__pool1_out_we) 
                | ((IData)(vlSelfRef.edge_cnn_top__DOT__pool2_out_we) 
                   | (IData)(vlSelfRef.edge_cnn_top__DOT__conv4__DOT__postprocess_valid))))) {
            if (vlSelfRef.__VdfgRegularize_h6e95ff9d_0_0) {
                vlSelfRef.__VdlyVal__edge_cnn_top__DOT__buffer_a__DOT__mem__v0 
                    = vlSelfRef.input_data;
                vlSelfRef.__VdlyDim0__edge_cnn_top__DOT__buffer_a__DOT__mem__v0 
                    = (0x00007fffU & vlSelfRef.input_addr);
            } else if (vlSelfRef.edge_cnn_top__DOT__pool1_out_we) {
                vlSelfRef.__VdlyVal__edge_cnn_top__DOT__buffer_a__DOT__mem__v0 
                    = vlSelfRef.edge_cnn_top__DOT__pool1_out_data;
                vlSelfRef.__VdlyDim0__edge_cnn_top__DOT__buffer_a__DOT__mem__v0 
                    = (0x00007fffU & vlSelfRef.edge_cnn_top__DOT__pool1_out_addr);
            } else if (vlSelfRef.edge_cnn_top__DOT__pool2_out_we) {
                vlSelfRef.__VdlyVal__edge_cnn_top__DOT__buffer_a__DOT__mem__v0 
                    = vlSelfRef.edge_cnn_top__DOT__pool2_out_data;
                vlSelfRef.__VdlyDim0__edge_cnn_top__DOT__buffer_a__DOT__mem__v0 
                    = (0x00007fffU & vlSelfRef.edge_cnn_top__DOT__pool2_out_addr);
            } else {
                vlSelfRef.__VdlyVal__edge_cnn_top__DOT__buffer_a__DOT__mem__v0 
                    = vlSelfRef.edge_cnn_top__DOT__conv4__DOT__out_data_reg;
                vlSelfRef.__VdlyDim0__edge_cnn_top__DOT__buffer_a__DOT__mem__v0 
                    = (0x00007fffU & vlSelfRef.edge_cnn_top__DOT__conv4__DOT__out_addr_reg);
            }
            vlSelfRef.__VdlySet__edge_cnn_top__DOT__buffer_a__DOT__mem__v0 = 1U;
        }
        if (((IData)(vlSelfRef.edge_cnn_top__DOT__conv1__DOT__postprocess_valid) 
             | ((IData)(vlSelfRef.edge_cnn_top__DOT__conv2__DOT__postprocess_valid) 
                | ((IData)(vlSelfRef.edge_cnn_top__DOT__gap_out_we) 
                   | (IData)(vlSelfRef.edge_cnn_top__DOT__conv3__DOT__postprocess_valid))))) {
            if (vlSelfRef.edge_cnn_top__DOT__conv1__DOT__postprocess_valid) {
                vlSelfRef.__VdlyVal__edge_cnn_top__DOT__buffer_b__DOT__mem__v0 
                    = vlSelfRef.edge_cnn_top__DOT__conv1__DOT__out_data_reg;
                vlSelfRef.__VdlyDim0__edge_cnn_top__DOT__buffer_b__DOT__mem__v0 
                    = (0x00007fffU & vlSelfRef.edge_cnn_top__DOT__conv1__DOT__out_addr_reg);
            } else if (vlSelfRef.edge_cnn_top__DOT__conv2__DOT__postprocess_valid) {
                vlSelfRef.__VdlyVal__edge_cnn_top__DOT__buffer_b__DOT__mem__v0 
                    = vlSelfRef.edge_cnn_top__DOT__conv2__DOT__out_data_reg;
                vlSelfRef.__VdlyDim0__edge_cnn_top__DOT__buffer_b__DOT__mem__v0 
                    = (0x00007fffU & vlSelfRef.edge_cnn_top__DOT__conv2__DOT__out_addr_reg);
            } else if (vlSelfRef.edge_cnn_top__DOT__conv3__DOT__postprocess_valid) {
                vlSelfRef.__VdlyVal__edge_cnn_top__DOT__buffer_b__DOT__mem__v0 
                    = vlSelfRef.edge_cnn_top__DOT__conv3__DOT__out_data_reg;
                vlSelfRef.__VdlyDim0__edge_cnn_top__DOT__buffer_b__DOT__mem__v0 
                    = (0x00007fffU & vlSelfRef.edge_cnn_top__DOT__conv3__DOT__out_addr_reg);
            } else {
                vlSelfRef.__VdlyVal__edge_cnn_top__DOT__buffer_b__DOT__mem__v0 
                    = vlSelfRef.edge_cnn_top__DOT__gap_out_data;
                vlSelfRef.__VdlyDim0__edge_cnn_top__DOT__buffer_b__DOT__mem__v0 
                    = (0x00007fffU & vlSelfRef.edge_cnn_top__DOT__gap_out_addr);
            }
            vlSelfRef.__VdlySet__edge_cnn_top__DOT__buffer_b__DOT__mem__v0 = 1U;
        }
        vlSelfRef.edge_cnn_top__DOT__linear_weight_data 
            = ((0x09ffU >= (IData)(vlSelfRef.edge_cnn_top__DOT__linear_weight_rom__DOT__rom_addr))
                ? vlSelfRef.edge_cnn_top__DOT__linear_weight_rom__DOT__rom
               [vlSelfRef.edge_cnn_top__DOT__linear_weight_rom__DOT__rom_addr]
                : 0U);
    }
    if ((3ULL & vlSelfRef.__VnbaTriggered[0U])) {
        Vedge_cnn_top___024root___nba_sequent__TOP__2(vlSelf);
    }
    if ((1ULL & vlSelfRef.__VnbaTriggered[0U])) {
        Vedge_cnn_top___024root___nba_sequent__TOP__3(vlSelf);
    }
    if ((3ULL & vlSelfRef.__VnbaTriggered[0U])) {
        Vedge_cnn_top___024root___nba_sequent__TOP__4(vlSelf);
    }
}

void Vedge_cnn_top___024root___trigger_orInto__act_vec_vec(VlUnpacked<QData/*63:0*/, 1> &out, const VlUnpacked<QData/*63:0*/, 1> &in) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vedge_cnn_top___024root___trigger_orInto__act_vec_vec\n"); );
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
VL_ATTR_COLD void Vedge_cnn_top___024root___dump_triggers__act(const VlUnpacked<QData/*63:0*/, 1> &triggers, const std::string &tag);
#endif  // VL_DEBUG

bool Vedge_cnn_top___024root___eval_phase__act(Vedge_cnn_top___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vedge_cnn_top___024root___eval_phase__act\n"); );
    Vedge_cnn_top__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    Vedge_cnn_top___024root___eval_triggers_vec__act(vlSelf);
#ifdef VL_DEBUG
    if (VL_UNLIKELY(vlSymsp->_vm_contextp__->debug())) {
        Vedge_cnn_top___024root___dump_triggers__act(vlSelfRef.__VactTriggered, "act"s);
    }
#endif
    Vedge_cnn_top___024root___trigger_orInto__act_vec_vec(vlSelfRef.__VnbaTriggered, vlSelfRef.__VactTriggered);
    return (0U);
}

void Vedge_cnn_top___024root___trigger_clear__act(VlUnpacked<QData/*63:0*/, 1> &out) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vedge_cnn_top___024root___trigger_clear__act\n"); );
    // Locals
    IData/*31:0*/ n;
    // Body
    n = 0U;
    do {
        out[n] = 0ULL;
        n = ((IData)(1U) + n);
    } while ((1U > n));
}

bool Vedge_cnn_top___024root___eval_phase__nba(Vedge_cnn_top___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vedge_cnn_top___024root___eval_phase__nba\n"); );
    Vedge_cnn_top__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Locals
    CData/*0:0*/ __VnbaExecute;
    // Body
    __VnbaExecute = Vedge_cnn_top___024root___trigger_anySet__act(vlSelfRef.__VnbaTriggered);
    if (__VnbaExecute) {
        Vedge_cnn_top___024root___eval_nba(vlSelf);
        Vedge_cnn_top___024root___trigger_clear__act(vlSelfRef.__VnbaTriggered);
    }
    return (__VnbaExecute);
}

void Vedge_cnn_top___024root___eval(Vedge_cnn_top___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vedge_cnn_top___024root___eval\n"); );
    Vedge_cnn_top__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
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
            Vedge_cnn_top___024root___dump_triggers__ico(vlSelfRef.__VicoTriggered, "ico"s);
#endif
            VL_FATAL_MT("src/edge_cnn_top.v", 20, "", "DIDNOTCONVERGE: Input combinational region did not converge after '--converge-limit' of 10000 tries");
        }
        __VicoIterCount = ((IData)(1U) + __VicoIterCount);
        vlSelfRef.__VicoPhaseResult = Vedge_cnn_top___024root___eval_phase__ico(vlSelf);
        vlSelfRef.__VicoFirstIteration = 0U;
    } while (vlSelfRef.__VicoPhaseResult);
    __VnbaIterCount = 0U;
    do {
        if (VL_UNLIKELY(((0x00002710U < __VnbaIterCount)))) {
#ifdef VL_DEBUG
            Vedge_cnn_top___024root___dump_triggers__act(vlSelfRef.__VnbaTriggered, "nba"s);
#endif
            VL_FATAL_MT("src/edge_cnn_top.v", 20, "", "DIDNOTCONVERGE: NBA region did not converge after '--converge-limit' of 10000 tries");
        }
        __VnbaIterCount = ((IData)(1U) + __VnbaIterCount);
        vlSelfRef.__VactIterCount = 0U;
        do {
            if (VL_UNLIKELY(((0x00002710U < vlSelfRef.__VactIterCount)))) {
#ifdef VL_DEBUG
                Vedge_cnn_top___024root___dump_triggers__act(vlSelfRef.__VactTriggered, "act"s);
#endif
                VL_FATAL_MT("src/edge_cnn_top.v", 20, "", "DIDNOTCONVERGE: Active region did not converge after '--converge-limit' of 10000 tries");
            }
            vlSelfRef.__VactIterCount = ((IData)(1U) 
                                         + vlSelfRef.__VactIterCount);
            vlSelfRef.__VactPhaseResult = Vedge_cnn_top___024root___eval_phase__act(vlSelf);
        } while (vlSelfRef.__VactPhaseResult);
        vlSelfRef.__VnbaPhaseResult = Vedge_cnn_top___024root___eval_phase__nba(vlSelf);
    } while (vlSelfRef.__VnbaPhaseResult);
}

#ifdef VL_DEBUG
void Vedge_cnn_top___024root___eval_debug_assertions(Vedge_cnn_top___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vedge_cnn_top___024root___eval_debug_assertions\n"); );
    Vedge_cnn_top__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
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
    if (VL_UNLIKELY(((vlSelfRef.input_addr & 0xfff80000U)))) {
        Verilated::overWidthError("input_addr");
    }
}
#endif  // VL_DEBUG
