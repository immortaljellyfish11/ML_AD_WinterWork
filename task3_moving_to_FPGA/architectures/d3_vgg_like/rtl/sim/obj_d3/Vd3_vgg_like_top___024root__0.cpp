// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design implementation internals
// See Vd3_vgg_like_top.h for the primary calling header

#include "Vd3_vgg_like_top__pch.h"

void Vd3_vgg_like_top___024root___eval_triggers_vec__ico(Vd3_vgg_like_top___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vd3_vgg_like_top___024root___eval_triggers_vec__ico\n"); );
    Vd3_vgg_like_top__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    vlSelfRef.__VicoTriggered[0U] = ((0xfffffffffffffffeULL 
                                      & vlSelfRef.__VicoTriggered[0U]) 
                                     | (IData)((IData)(vlSelfRef.__VicoFirstIteration)));
}

bool Vd3_vgg_like_top___024root___trigger_anySet__ico(const VlUnpacked<QData/*63:0*/, 1> &in) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vd3_vgg_like_top___024root___trigger_anySet__ico\n"); );
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

void Vd3_vgg_like_top___024root___ico_sequent__TOP__0(Vd3_vgg_like_top___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vd3_vgg_like_top___024root___ico_sequent__TOP__0\n"); );
    Vd3_vgg_like_top__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
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
                                                                                | ((IData)(vlSelfRef.d3_vgg_like_top__DOT__conv5__DOT__postprocess_pending) 
                                                                                | ((IData)(vlSelfRef.d3_vgg_like_top__DOT__conv5__DOT__product_valid) 
                                                                                | ((IData)(vlSelfRef.d3_vgg_like_top__DOT__conv5__DOT__postprocess_valid) 
                                                                                | ((IData)(vlSelfRef.d3_vgg_like_top__DOT__conv1__DOT__controller_busy) 
                                                                                | ((IData)(vlSelfRef.d3_vgg_like_top__DOT__conv2__DOT__controller_busy) 
                                                                                | ((IData)(vlSelfRef.d3_vgg_like_top__DOT__conv3__DOT__controller_busy) 
                                                                                | ((IData)(vlSelfRef.d3_vgg_like_top__DOT__conv4__DOT__controller_busy) 
                                                                                | ((IData)(vlSelfRef.d3_vgg_like_top__DOT__conv5__DOT__controller_busy) 
                                                                                | ((IData)(vlSelfRef.d3_vgg_like_top__DOT__p1_busy) 
                                                                                | ((IData)(vlSelfRef.d3_vgg_like_top__DOT__p2_busy) 
                                                                                | ((IData)(vlSelfRef.d3_vgg_like_top__DOT__gap_busy) 
                                                                                | (IData)(vlSelfRef.d3_vgg_like_top__DOT__fc_busy))))))))))))))))))))))))) 
                                                   & (IData)(vlSelfRef.input_we)));
}

void Vd3_vgg_like_top___024root___eval_ico(Vd3_vgg_like_top___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vd3_vgg_like_top___024root___eval_ico\n"); );
    Vd3_vgg_like_top__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    if ((1ULL & vlSelfRef.__VicoTriggered[0U])) {
        vlSelfRef.__VdfgRegularize_h6e95ff9d_0_0 = 
            ((~ (IData)(vlSelfRef.busy)) & ((~ ((IData)(vlSelfRef.d3_vgg_like_top__DOT__conv1__DOT__postprocess_pending) 
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
                                                                                | ((IData)(vlSelfRef.d3_vgg_like_top__DOT__conv5__DOT__postprocess_pending) 
                                                                                | ((IData)(vlSelfRef.d3_vgg_like_top__DOT__conv5__DOT__product_valid) 
                                                                                | ((IData)(vlSelfRef.d3_vgg_like_top__DOT__conv5__DOT__postprocess_valid) 
                                                                                | ((IData)(vlSelfRef.d3_vgg_like_top__DOT__conv1__DOT__controller_busy) 
                                                                                | ((IData)(vlSelfRef.d3_vgg_like_top__DOT__conv2__DOT__controller_busy) 
                                                                                | ((IData)(vlSelfRef.d3_vgg_like_top__DOT__conv3__DOT__controller_busy) 
                                                                                | ((IData)(vlSelfRef.d3_vgg_like_top__DOT__conv4__DOT__controller_busy) 
                                                                                | ((IData)(vlSelfRef.d3_vgg_like_top__DOT__conv5__DOT__controller_busy) 
                                                                                | ((IData)(vlSelfRef.d3_vgg_like_top__DOT__p1_busy) 
                                                                                | ((IData)(vlSelfRef.d3_vgg_like_top__DOT__p2_busy) 
                                                                                | ((IData)(vlSelfRef.d3_vgg_like_top__DOT__gap_busy) 
                                                                                | (IData)(vlSelfRef.d3_vgg_like_top__DOT__fc_busy))))))))))))))))))))))))) 
                                            & (IData)(vlSelfRef.input_we)));
    }
}

#ifdef VL_DEBUG
VL_ATTR_COLD void Vd3_vgg_like_top___024root___dump_triggers__ico(const VlUnpacked<QData/*63:0*/, 1> &triggers, const std::string &tag);
#endif  // VL_DEBUG

bool Vd3_vgg_like_top___024root___eval_phase__ico(Vd3_vgg_like_top___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vd3_vgg_like_top___024root___eval_phase__ico\n"); );
    Vd3_vgg_like_top__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Locals
    CData/*0:0*/ __VicoExecute;
    // Body
    Vd3_vgg_like_top___024root___eval_triggers_vec__ico(vlSelf);
#ifdef VL_DEBUG
    if (VL_UNLIKELY(vlSymsp->_vm_contextp__->debug())) {
        Vd3_vgg_like_top___024root___dump_triggers__ico(vlSelfRef.__VicoTriggered, "ico"s);
    }
#endif
    __VicoExecute = Vd3_vgg_like_top___024root___trigger_anySet__ico(vlSelfRef.__VicoTriggered);
    if (__VicoExecute) {
        Vd3_vgg_like_top___024root___eval_ico(vlSelf);
    }
    return (__VicoExecute);
}

void Vd3_vgg_like_top___024root___eval_triggers_vec__act(Vd3_vgg_like_top___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vd3_vgg_like_top___024root___eval_triggers_vec__act\n"); );
    Vd3_vgg_like_top__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
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

bool Vd3_vgg_like_top___024root___trigger_anySet__act(const VlUnpacked<QData/*63:0*/, 1> &in) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vd3_vgg_like_top___024root___trigger_anySet__act\n"); );
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

void Vd3_vgg_like_top___024root___nba_sequent__TOP__0(Vd3_vgg_like_top___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vd3_vgg_like_top___024root___nba_sequent__TOP__0\n"); );
    Vd3_vgg_like_top__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Locals
    CData/*0:0*/ __Vdly__d3_vgg_like_top__DOT__fc_start;
    __Vdly__d3_vgg_like_top__DOT__fc_start = 0;
    CData/*3:0*/ __Vdly__d3_vgg_like_top__DOT__fsm__DOT__state;
    __Vdly__d3_vgg_like_top__DOT__fsm__DOT__state = 0;
    CData/*2:0*/ __Vdly__d3_vgg_like_top__DOT__linear__DOT__state;
    __Vdly__d3_vgg_like_top__DOT__linear__DOT__state = 0;
    IData/*31:0*/ __Vdly__d3_vgg_like_top__DOT__linear__DOT__acc;
    __Vdly__d3_vgg_like_top__DOT__linear__DOT__acc = 0;
    // Body
    vlSelfRef.__Vdly__d3_vgg_like_top__DOT__gap__DOT__state 
        = vlSelfRef.d3_vgg_like_top__DOT__gap__DOT__state;
    vlSelfRef.__Vdly__d3_vgg_like_top__DOT__gap__DOT__idx 
        = vlSelfRef.d3_vgg_like_top__DOT__gap__DOT__idx;
    vlSelfRef.__Vdly__d3_vgg_like_top__DOT__gap__DOT__ch 
        = vlSelfRef.d3_vgg_like_top__DOT__gap__DOT__ch;
    vlSelfRef.__Vdly__d3_vgg_like_top__DOT__gap__DOT__sum 
        = vlSelfRef.d3_vgg_like_top__DOT__gap__DOT__sum;
    vlSelfRef.__Vdly__d3_vgg_like_top__DOT__pool2__DOT__state 
        = vlSelfRef.d3_vgg_like_top__DOT__pool2__DOT__state;
    vlSelfRef.__Vdly__d3_vgg_like_top__DOT__pool2__DOT__tap 
        = vlSelfRef.d3_vgg_like_top__DOT__pool2__DOT__tap;
    vlSelfRef.__Vdly__d3_vgg_like_top__DOT__pool2__DOT__col 
        = vlSelfRef.d3_vgg_like_top__DOT__pool2__DOT__col;
    vlSelfRef.__Vdly__d3_vgg_like_top__DOT__pool2__DOT__row 
        = vlSelfRef.d3_vgg_like_top__DOT__pool2__DOT__row;
    vlSelfRef.__Vdly__d3_vgg_like_top__DOT__pool2__DOT__ch 
        = vlSelfRef.d3_vgg_like_top__DOT__pool2__DOT__ch;
    vlSelfRef.__Vdly__d3_vgg_like_top__DOT__pool2__DOT__current_max 
        = vlSelfRef.d3_vgg_like_top__DOT__pool2__DOT__current_max;
    __Vdly__d3_vgg_like_top__DOT__linear__DOT__state 
        = vlSelfRef.d3_vgg_like_top__DOT__linear__DOT__state;
    __Vdly__d3_vgg_like_top__DOT__linear__DOT__acc 
        = vlSelfRef.d3_vgg_like_top__DOT__linear__DOT__acc;
    vlSelfRef.__Vdly__d3_vgg_like_top__DOT__linear__DOT__feature_idx 
        = vlSelfRef.d3_vgg_like_top__DOT__linear__DOT__feature_idx;
    vlSelfRef.__Vdly__d3_vgg_like_top__DOT__pool1__DOT__state 
        = vlSelfRef.d3_vgg_like_top__DOT__pool1__DOT__state;
    vlSelfRef.__Vdly__d3_vgg_like_top__DOT__pool1__DOT__current_max 
        = vlSelfRef.d3_vgg_like_top__DOT__pool1__DOT__current_max;
    vlSelfRef.__Vdly__d3_vgg_like_top__DOT__pool1__DOT__tap 
        = vlSelfRef.d3_vgg_like_top__DOT__pool1__DOT__tap;
    vlSelfRef.__Vdly__d3_vgg_like_top__DOT__conv2__DOT__engine__DOT__state 
        = vlSelfRef.d3_vgg_like_top__DOT__conv2__DOT__engine__DOT__state;
    vlSelfRef.__Vdly__d3_vgg_like_top__DOT__conv2__DOT__engine__DOT__product_reg 
        = vlSelfRef.d3_vgg_like_top__DOT__conv2__DOT__engine__DOT__product_reg;
    vlSelfRef.__Vdly__d3_vgg_like_top__DOT__conv2__DOT__engine__DOT__product_valid 
        = vlSelfRef.d3_vgg_like_top__DOT__conv2__DOT__engine__DOT__product_valid;
    vlSelfRef.__Vdly__d3_vgg_like_top__DOT__conv2__DOT__engine__DOT__product_last 
        = vlSelfRef.d3_vgg_like_top__DOT__conv2__DOT__engine__DOT__product_last;
    vlSelfRef.__Vdly__d3_vgg_like_top__DOT__conv3__DOT__engine__DOT__state 
        = vlSelfRef.d3_vgg_like_top__DOT__conv3__DOT__engine__DOT__state;
    vlSelfRef.__Vdly__d3_vgg_like_top__DOT__conv3__DOT__engine__DOT__product_reg 
        = vlSelfRef.d3_vgg_like_top__DOT__conv3__DOT__engine__DOT__product_reg;
    vlSelfRef.__Vdly__d3_vgg_like_top__DOT__conv3__DOT__engine__DOT__product_valid 
        = vlSelfRef.d3_vgg_like_top__DOT__conv3__DOT__engine__DOT__product_valid;
    vlSelfRef.__Vdly__d3_vgg_like_top__DOT__conv3__DOT__engine__DOT__product_last 
        = vlSelfRef.d3_vgg_like_top__DOT__conv3__DOT__engine__DOT__product_last;
    vlSelfRef.__Vdly__d3_vgg_like_top__DOT__conv2__DOT__engine__DOT__ic 
        = vlSelfRef.d3_vgg_like_top__DOT__conv2__DOT__engine__DOT__ic;
    vlSelfRef.__Vdly__d3_vgg_like_top__DOT__conv3__DOT__engine__DOT__ic 
        = vlSelfRef.d3_vgg_like_top__DOT__conv3__DOT__engine__DOT__ic;
    vlSelfRef.__Vdly__d3_vgg_like_top__DOT__conv1__DOT__engine__DOT__state 
        = vlSelfRef.d3_vgg_like_top__DOT__conv1__DOT__engine__DOT__state;
    vlSelfRef.__Vdly__d3_vgg_like_top__DOT__conv1__DOT__engine__DOT__product_reg 
        = vlSelfRef.d3_vgg_like_top__DOT__conv1__DOT__engine__DOT__product_reg;
    vlSelfRef.__Vdly__d3_vgg_like_top__DOT__conv1__DOT__engine__DOT__product_valid 
        = vlSelfRef.d3_vgg_like_top__DOT__conv1__DOT__engine__DOT__product_valid;
    vlSelfRef.__Vdly__d3_vgg_like_top__DOT__conv1__DOT__engine__DOT__product_last 
        = vlSelfRef.d3_vgg_like_top__DOT__conv1__DOT__engine__DOT__product_last;
    vlSelfRef.__Vdly__d3_vgg_like_top__DOT__conv4__DOT__engine__DOT__state 
        = vlSelfRef.d3_vgg_like_top__DOT__conv4__DOT__engine__DOT__state;
    vlSelfRef.__Vdly__d3_vgg_like_top__DOT__conv4__DOT__engine__DOT__product_reg 
        = vlSelfRef.d3_vgg_like_top__DOT__conv4__DOT__engine__DOT__product_reg;
    vlSelfRef.__Vdly__d3_vgg_like_top__DOT__conv4__DOT__engine__DOT__product_valid 
        = vlSelfRef.d3_vgg_like_top__DOT__conv4__DOT__engine__DOT__product_valid;
    vlSelfRef.__Vdly__d3_vgg_like_top__DOT__conv4__DOT__engine__DOT__product_last 
        = vlSelfRef.d3_vgg_like_top__DOT__conv4__DOT__engine__DOT__product_last;
    vlSelfRef.__Vdly__d3_vgg_like_top__DOT__conv5__DOT__engine__DOT__state 
        = vlSelfRef.d3_vgg_like_top__DOT__conv5__DOT__engine__DOT__state;
    vlSelfRef.__Vdly__d3_vgg_like_top__DOT__conv5__DOT__engine__DOT__product_reg 
        = vlSelfRef.d3_vgg_like_top__DOT__conv5__DOT__engine__DOT__product_reg;
    vlSelfRef.__Vdly__d3_vgg_like_top__DOT__conv5__DOT__engine__DOT__product_valid 
        = vlSelfRef.d3_vgg_like_top__DOT__conv5__DOT__engine__DOT__product_valid;
    vlSelfRef.__Vdly__d3_vgg_like_top__DOT__conv5__DOT__engine__DOT__product_last 
        = vlSelfRef.d3_vgg_like_top__DOT__conv5__DOT__engine__DOT__product_last;
    vlSelfRef.__Vdly__d3_vgg_like_top__DOT__conv1__DOT__engine__DOT__ic 
        = vlSelfRef.d3_vgg_like_top__DOT__conv1__DOT__engine__DOT__ic;
    vlSelfRef.__Vdly__d3_vgg_like_top__DOT__conv4__DOT__engine__DOT__ic 
        = vlSelfRef.d3_vgg_like_top__DOT__conv4__DOT__engine__DOT__ic;
    vlSelfRef.__Vdly__d3_vgg_like_top__DOT__conv5__DOT__engine__DOT__ic 
        = vlSelfRef.d3_vgg_like_top__DOT__conv5__DOT__engine__DOT__ic;
    vlSelfRef.__Vdly__d3_vgg_like_top__DOT__conv2__DOT__engine_start 
        = vlSelfRef.d3_vgg_like_top__DOT__conv2__DOT__engine_start;
    vlSelfRef.__Vdly__d3_vgg_like_top__DOT__conv2__DOT__controller__DOT__state 
        = vlSelfRef.d3_vgg_like_top__DOT__conv2__DOT__controller__DOT__state;
    vlSelfRef.__Vdly__d3_vgg_like_top__DOT__conv3__DOT__engine_start 
        = vlSelfRef.d3_vgg_like_top__DOT__conv3__DOT__engine_start;
    vlSelfRef.__Vdly__d3_vgg_like_top__DOT__conv3__DOT__controller__DOT__state 
        = vlSelfRef.d3_vgg_like_top__DOT__conv3__DOT__controller__DOT__state;
    vlSelfRef.__Vdly__d3_vgg_like_top__DOT__conv1__DOT__engine_start 
        = vlSelfRef.d3_vgg_like_top__DOT__conv1__DOT__engine_start;
    vlSelfRef.__Vdly__d3_vgg_like_top__DOT__conv1__DOT__controller__DOT__state 
        = vlSelfRef.d3_vgg_like_top__DOT__conv1__DOT__controller__DOT__state;
    vlSelfRef.__Vdly__d3_vgg_like_top__DOT__conv4__DOT__engine_start 
        = vlSelfRef.d3_vgg_like_top__DOT__conv4__DOT__engine_start;
    vlSelfRef.__Vdly__d3_vgg_like_top__DOT__conv4__DOT__controller__DOT__state 
        = vlSelfRef.d3_vgg_like_top__DOT__conv4__DOT__controller__DOT__state;
    vlSelfRef.__Vdly__d3_vgg_like_top__DOT__conv5__DOT__engine_start 
        = vlSelfRef.d3_vgg_like_top__DOT__conv5__DOT__engine_start;
    vlSelfRef.__Vdly__d3_vgg_like_top__DOT__conv5__DOT__controller__DOT__state 
        = vlSelfRef.d3_vgg_like_top__DOT__conv5__DOT__controller__DOT__state;
    vlSelfRef.__Vdly__d3_vgg_like_top__DOT__c1_start 
        = vlSelfRef.d3_vgg_like_top__DOT__c1_start;
    vlSelfRef.__Vdly__d3_vgg_like_top__DOT__c2_start 
        = vlSelfRef.d3_vgg_like_top__DOT__c2_start;
    vlSelfRef.__Vdly__d3_vgg_like_top__DOT__p1_start 
        = vlSelfRef.d3_vgg_like_top__DOT__p1_start;
    vlSelfRef.__Vdly__d3_vgg_like_top__DOT__c3_start 
        = vlSelfRef.d3_vgg_like_top__DOT__c3_start;
    vlSelfRef.__Vdly__d3_vgg_like_top__DOT__c4_start 
        = vlSelfRef.d3_vgg_like_top__DOT__c4_start;
    vlSelfRef.__Vdly__d3_vgg_like_top__DOT__p2_start 
        = vlSelfRef.d3_vgg_like_top__DOT__p2_start;
    vlSelfRef.__Vdly__d3_vgg_like_top__DOT__c5_start 
        = vlSelfRef.d3_vgg_like_top__DOT__c5_start;
    vlSelfRef.__Vdly__d3_vgg_like_top__DOT__gap_start 
        = vlSelfRef.d3_vgg_like_top__DOT__gap_start;
    __Vdly__d3_vgg_like_top__DOT__fc_start = vlSelfRef.d3_vgg_like_top__DOT__fc_start;
    vlSelfRef.__Vdly__state_dbg = vlSelfRef.state_dbg;
    __Vdly__d3_vgg_like_top__DOT__fsm__DOT__state = vlSelfRef.d3_vgg_like_top__DOT__fsm__DOT__state;
    if (vlSelfRef.rst_n) {
        if (vlSelfRef.d3_vgg_like_top__DOT__logit_valid) {
            if (((((((((0U == (IData)(vlSelfRef.d3_vgg_like_top__DOT__logit_index)) 
                       | (1U == (IData)(vlSelfRef.d3_vgg_like_top__DOT__logit_index))) 
                      | (2U == (IData)(vlSelfRef.d3_vgg_like_top__DOT__logit_index))) 
                     | (3U == (IData)(vlSelfRef.d3_vgg_like_top__DOT__logit_index))) 
                    | (4U == (IData)(vlSelfRef.d3_vgg_like_top__DOT__logit_index))) 
                   | (5U == (IData)(vlSelfRef.d3_vgg_like_top__DOT__logit_index))) 
                  | (6U == (IData)(vlSelfRef.d3_vgg_like_top__DOT__logit_index))) 
                 | (7U == (IData)(vlSelfRef.d3_vgg_like_top__DOT__logit_index)))) {
                if ((0U != (IData)(vlSelfRef.d3_vgg_like_top__DOT__logit_index))) {
                    if ((1U != (IData)(vlSelfRef.d3_vgg_like_top__DOT__logit_index))) {
                        if ((2U == (IData)(vlSelfRef.d3_vgg_like_top__DOT__logit_index))) {
                            vlSelfRef.d3_vgg_like_top__DOT__l2 
                                = vlSelfRef.d3_vgg_like_top__DOT__logit_data;
                        }
                        if ((2U != (IData)(vlSelfRef.d3_vgg_like_top__DOT__logit_index))) {
                            if ((3U != (IData)(vlSelfRef.d3_vgg_like_top__DOT__logit_index))) {
                                if ((4U == (IData)(vlSelfRef.d3_vgg_like_top__DOT__logit_index))) {
                                    vlSelfRef.d3_vgg_like_top__DOT__l4 
                                        = vlSelfRef.d3_vgg_like_top__DOT__logit_data;
                                }
                                if ((4U != (IData)(vlSelfRef.d3_vgg_like_top__DOT__logit_index))) {
                                    if ((5U != (IData)(vlSelfRef.d3_vgg_like_top__DOT__logit_index))) {
                                        if ((6U != (IData)(vlSelfRef.d3_vgg_like_top__DOT__logit_index))) {
                                            vlSelfRef.d3_vgg_like_top__DOT__l7 
                                                = vlSelfRef.d3_vgg_like_top__DOT__logit_data;
                                        }
                                        if ((6U == (IData)(vlSelfRef.d3_vgg_like_top__DOT__logit_index))) {
                                            vlSelfRef.d3_vgg_like_top__DOT__l6 
                                                = vlSelfRef.d3_vgg_like_top__DOT__logit_data;
                                        }
                                    }
                                    if ((5U == (IData)(vlSelfRef.d3_vgg_like_top__DOT__logit_index))) {
                                        vlSelfRef.d3_vgg_like_top__DOT__l5 
                                            = vlSelfRef.d3_vgg_like_top__DOT__logit_data;
                                    }
                                }
                            }
                            if ((3U == (IData)(vlSelfRef.d3_vgg_like_top__DOT__logit_index))) {
                                vlSelfRef.d3_vgg_like_top__DOT__l3 
                                    = vlSelfRef.d3_vgg_like_top__DOT__logit_data;
                            }
                        }
                    }
                    if ((1U == (IData)(vlSelfRef.d3_vgg_like_top__DOT__logit_index))) {
                        vlSelfRef.d3_vgg_like_top__DOT__l1 
                            = vlSelfRef.d3_vgg_like_top__DOT__logit_data;
                    }
                }
                if ((0U == (IData)(vlSelfRef.d3_vgg_like_top__DOT__logit_index))) {
                    vlSelfRef.d3_vgg_like_top__DOT__l0 
                        = vlSelfRef.d3_vgg_like_top__DOT__logit_data;
                }
            }
            if ((1U & (~ ((((((((0U == (IData)(vlSelfRef.d3_vgg_like_top__DOT__logit_index)) 
                                | (1U == (IData)(vlSelfRef.d3_vgg_like_top__DOT__logit_index))) 
                               | (2U == (IData)(vlSelfRef.d3_vgg_like_top__DOT__logit_index))) 
                              | (3U == (IData)(vlSelfRef.d3_vgg_like_top__DOT__logit_index))) 
                             | (4U == (IData)(vlSelfRef.d3_vgg_like_top__DOT__logit_index))) 
                            | (5U == (IData)(vlSelfRef.d3_vgg_like_top__DOT__logit_index))) 
                           | (6U == (IData)(vlSelfRef.d3_vgg_like_top__DOT__logit_index))) 
                          | (7U == (IData)(vlSelfRef.d3_vgg_like_top__DOT__logit_index)))))) {
                if ((8U == (IData)(vlSelfRef.d3_vgg_like_top__DOT__logit_index))) {
                    vlSelfRef.d3_vgg_like_top__DOT__l8 
                        = vlSelfRef.d3_vgg_like_top__DOT__logit_data;
                }
                if ((8U != (IData)(vlSelfRef.d3_vgg_like_top__DOT__logit_index))) {
                    if ((9U == (IData)(vlSelfRef.d3_vgg_like_top__DOT__logit_index))) {
                        vlSelfRef.d3_vgg_like_top__DOT__l9 
                            = vlSelfRef.d3_vgg_like_top__DOT__logit_data;
                    }
                }
            }
        }
        vlSelfRef.__Vdly__d3_vgg_like_top__DOT__c1_start = 0U;
        vlSelfRef.__Vdly__d3_vgg_like_top__DOT__c2_start = 0U;
        vlSelfRef.__Vdly__d3_vgg_like_top__DOT__p1_start = 0U;
        vlSelfRef.__Vdly__d3_vgg_like_top__DOT__c3_start = 0U;
        vlSelfRef.__Vdly__d3_vgg_like_top__DOT__c4_start = 0U;
        vlSelfRef.__Vdly__d3_vgg_like_top__DOT__p2_start = 0U;
        vlSelfRef.__Vdly__d3_vgg_like_top__DOT__c5_start = 0U;
        vlSelfRef.__Vdly__d3_vgg_like_top__DOT__gap_start = 0U;
        __Vdly__d3_vgg_like_top__DOT__fc_start = 0U;
        vlSelfRef.argmax_valid = 0U;
        vlSelfRef.done = 0U;
        vlSelfRef.__Vdly__state_dbg = vlSelfRef.d3_vgg_like_top__DOT__fsm__DOT__state;
        if (((((((((0U == (IData)(vlSelfRef.d3_vgg_like_top__DOT__fsm__DOT__state)) 
                   | (1U == (IData)(vlSelfRef.d3_vgg_like_top__DOT__fsm__DOT__state))) 
                  | (2U == (IData)(vlSelfRef.d3_vgg_like_top__DOT__fsm__DOT__state))) 
                 | (3U == (IData)(vlSelfRef.d3_vgg_like_top__DOT__fsm__DOT__state))) 
                | (4U == (IData)(vlSelfRef.d3_vgg_like_top__DOT__fsm__DOT__state))) 
               | (5U == (IData)(vlSelfRef.d3_vgg_like_top__DOT__fsm__DOT__state))) 
              | (6U == (IData)(vlSelfRef.d3_vgg_like_top__DOT__fsm__DOT__state))) 
             | (7U == (IData)(vlSelfRef.d3_vgg_like_top__DOT__fsm__DOT__state)))) {
            if ((0U == (IData)(vlSelfRef.d3_vgg_like_top__DOT__fsm__DOT__state))) {
                vlSelfRef.busy = 0U;
                if (vlSelfRef.start) {
                    vlSelfRef.busy = 1U;
                    vlSelfRef.__Vdly__d3_vgg_like_top__DOT__c1_start = 1U;
                    __Vdly__d3_vgg_like_top__DOT__fsm__DOT__state = 1U;
                }
            } else if ((1U == (IData)(vlSelfRef.d3_vgg_like_top__DOT__fsm__DOT__state))) {
                if (vlSelfRef.d3_vgg_like_top__DOT__conv1__DOT__controller_done_d2) {
                    vlSelfRef.__Vdly__d3_vgg_like_top__DOT__c2_start = 1U;
                    __Vdly__d3_vgg_like_top__DOT__fsm__DOT__state = 2U;
                }
            } else if ((2U == (IData)(vlSelfRef.d3_vgg_like_top__DOT__fsm__DOT__state))) {
                if (vlSelfRef.d3_vgg_like_top__DOT__conv2__DOT__controller_done_d2) {
                    vlSelfRef.__Vdly__d3_vgg_like_top__DOT__p1_start = 1U;
                    __Vdly__d3_vgg_like_top__DOT__fsm__DOT__state = 3U;
                }
            } else if ((3U == (IData)(vlSelfRef.d3_vgg_like_top__DOT__fsm__DOT__state))) {
                if (vlSelfRef.d3_vgg_like_top__DOT__p1_done) {
                    vlSelfRef.__Vdly__d3_vgg_like_top__DOT__c3_start = 1U;
                    __Vdly__d3_vgg_like_top__DOT__fsm__DOT__state = 4U;
                }
            } else if ((4U == (IData)(vlSelfRef.d3_vgg_like_top__DOT__fsm__DOT__state))) {
                if (vlSelfRef.d3_vgg_like_top__DOT__conv3__DOT__controller_done_d2) {
                    vlSelfRef.__Vdly__d3_vgg_like_top__DOT__c4_start = 1U;
                    __Vdly__d3_vgg_like_top__DOT__fsm__DOT__state = 5U;
                }
            } else if ((5U == (IData)(vlSelfRef.d3_vgg_like_top__DOT__fsm__DOT__state))) {
                if (vlSelfRef.d3_vgg_like_top__DOT__conv4__DOT__controller_done_d2) {
                    vlSelfRef.__Vdly__d3_vgg_like_top__DOT__p2_start = 1U;
                    __Vdly__d3_vgg_like_top__DOT__fsm__DOT__state = 6U;
                }
            } else if ((6U == (IData)(vlSelfRef.d3_vgg_like_top__DOT__fsm__DOT__state))) {
                if (vlSelfRef.d3_vgg_like_top__DOT__p2_done) {
                    vlSelfRef.__Vdly__d3_vgg_like_top__DOT__c5_start = 1U;
                    __Vdly__d3_vgg_like_top__DOT__fsm__DOT__state = 7U;
                }
            } else if (vlSelfRef.d3_vgg_like_top__DOT__conv5__DOT__controller_done_d2) {
                vlSelfRef.__Vdly__d3_vgg_like_top__DOT__gap_start = 1U;
                __Vdly__d3_vgg_like_top__DOT__fsm__DOT__state = 8U;
            }
        } else if ((8U == (IData)(vlSelfRef.d3_vgg_like_top__DOT__fsm__DOT__state))) {
            if (vlSelfRef.d3_vgg_like_top__DOT__gap_done) {
                __Vdly__d3_vgg_like_top__DOT__fc_start = 1U;
                __Vdly__d3_vgg_like_top__DOT__fsm__DOT__state = 9U;
            }
        } else if ((9U == (IData)(vlSelfRef.d3_vgg_like_top__DOT__fsm__DOT__state))) {
            if (vlSelfRef.d3_vgg_like_top__DOT__fc_done) {
                vlSelfRef.argmax_valid = 1U;
                __Vdly__d3_vgg_like_top__DOT__fsm__DOT__state = 0x0aU;
            }
        } else if ((0x0aU == (IData)(vlSelfRef.d3_vgg_like_top__DOT__fsm__DOT__state))) {
            __Vdly__d3_vgg_like_top__DOT__fsm__DOT__state = 0x0bU;
        } else if ((0x0bU == (IData)(vlSelfRef.d3_vgg_like_top__DOT__fsm__DOT__state))) {
            vlSelfRef.busy = 0U;
            vlSelfRef.done = 1U;
            __Vdly__d3_vgg_like_top__DOT__fsm__DOT__state = 0U;
        } else {
            __Vdly__d3_vgg_like_top__DOT__fsm__DOT__state = 0U;
        }
        vlSelfRef.d3_vgg_like_top__DOT__logit_valid = 0U;
        vlSelfRef.d3_vgg_like_top__DOT__fc_done = 0U;
        if ((4U & (IData)(vlSelfRef.d3_vgg_like_top__DOT__linear__DOT__state))) {
            if ((2U & (IData)(vlSelfRef.d3_vgg_like_top__DOT__linear__DOT__state))) {
                __Vdly__d3_vgg_like_top__DOT__linear__DOT__state = 0U;
            } else if ((1U & (IData)(vlSelfRef.d3_vgg_like_top__DOT__linear__DOT__state))) {
                __Vdly__d3_vgg_like_top__DOT__linear__DOT__state = 0U;
            } else {
                vlSelfRef.d3_vgg_like_top__DOT__fc_busy = 0U;
                vlSelfRef.d3_vgg_like_top__DOT__fc_done = 1U;
                __Vdly__d3_vgg_like_top__DOT__linear__DOT__state = 0U;
            }
        } else if ((2U & (IData)(vlSelfRef.d3_vgg_like_top__DOT__linear__DOT__state))) {
            if ((1U & (IData)(vlSelfRef.d3_vgg_like_top__DOT__linear__DOT__state))) {
                vlSelfRef.d3_vgg_like_top__DOT__logit_index 
                    = vlSelfRef.d3_vgg_like_top__DOT__linear__DOT__class_idx;
                vlSelfRef.d3_vgg_like_top__DOT__logit_valid = 1U;
                vlSelfRef.d3_vgg_like_top__DOT__logit_data 
                    = (vlSelfRef.d3_vgg_like_top__DOT__linear__DOT__acc 
                       + (vlSelfRef.d3_vgg_like_top__DOT__rbf__DOT__rom
                          [vlSelfRef.d3_vgg_like_top__DOT__linear__DOT__class_idx] 
                          & (- (IData)((9U >= (IData)(vlSelfRef.d3_vgg_like_top__DOT__linear__DOT__class_idx))))));
                if ((9U == (IData)(vlSelfRef.d3_vgg_like_top__DOT__linear__DOT__class_idx))) {
                    __Vdly__d3_vgg_like_top__DOT__linear__DOT__state = 4U;
                } else {
                    vlSelfRef.d3_vgg_like_top__DOT__linear__DOT__class_idx 
                        = (0x0000000fU & ((IData)(1U) 
                                          + (IData)(vlSelfRef.d3_vgg_like_top__DOT__linear__DOT__class_idx)));
                    vlSelfRef.__Vdly__d3_vgg_like_top__DOT__linear__DOT__feature_idx = 0U;
                    __Vdly__d3_vgg_like_top__DOT__linear__DOT__acc = 0U;
                    __Vdly__d3_vgg_like_top__DOT__linear__DOT__state = 1U;
                }
            } else {
                __Vdly__d3_vgg_like_top__DOT__linear__DOT__acc 
                    = (vlSelfRef.d3_vgg_like_top__DOT__linear__DOT__acc 
                       + VL_MULS_III(32, VL_EXTENDS_II(32,8, (IData)(vlSelfRef.d3_vgg_like_top__DOT__a_rd)), 
                                     VL_EXTENDS_II(32,8, (IData)(vlSelfRef.d3_vgg_like_top__DOT__fc_wd))));
                if ((0x0000007fU == (IData)(vlSelfRef.d3_vgg_like_top__DOT__linear__DOT__feature_idx))) {
                    __Vdly__d3_vgg_like_top__DOT__linear__DOT__state = 3U;
                } else {
                    vlSelfRef.__Vdly__d3_vgg_like_top__DOT__linear__DOT__feature_idx 
                        = (0x0000007fU & ((IData)(1U) 
                                          + (IData)(vlSelfRef.d3_vgg_like_top__DOT__linear__DOT__feature_idx)));
                    __Vdly__d3_vgg_like_top__DOT__linear__DOT__state = 1U;
                }
            }
        } else if ((1U & (IData)(vlSelfRef.d3_vgg_like_top__DOT__linear__DOT__state))) {
            __Vdly__d3_vgg_like_top__DOT__linear__DOT__state = 2U;
        } else {
            vlSelfRef.d3_vgg_like_top__DOT__fc_busy = 0U;
            if (vlSelfRef.d3_vgg_like_top__DOT__fc_start) {
                vlSelfRef.d3_vgg_like_top__DOT__linear__DOT__class_idx = 0U;
                vlSelfRef.__Vdly__d3_vgg_like_top__DOT__linear__DOT__feature_idx = 0U;
                __Vdly__d3_vgg_like_top__DOT__linear__DOT__acc = 0U;
                vlSelfRef.d3_vgg_like_top__DOT__fc_busy = 1U;
                __Vdly__d3_vgg_like_top__DOT__linear__DOT__state = 1U;
            }
        }
    } else {
        vlSelfRef.d3_vgg_like_top__DOT__l2 = 0U;
        vlSelfRef.d3_vgg_like_top__DOT__l1 = 0U;
        vlSelfRef.d3_vgg_like_top__DOT__l4 = 0U;
        vlSelfRef.d3_vgg_like_top__DOT__l3 = 0U;
        vlSelfRef.d3_vgg_like_top__DOT__l0 = 0U;
        vlSelfRef.d3_vgg_like_top__DOT__l7 = 0U;
        vlSelfRef.d3_vgg_like_top__DOT__l5 = 0U;
        vlSelfRef.d3_vgg_like_top__DOT__l6 = 0U;
        vlSelfRef.d3_vgg_like_top__DOT__l8 = 0U;
        vlSelfRef.d3_vgg_like_top__DOT__l9 = 0U;
        __Vdly__d3_vgg_like_top__DOT__fsm__DOT__state = 0U;
        vlSelfRef.__Vdly__d3_vgg_like_top__DOT__c1_start = 0U;
        vlSelfRef.__Vdly__d3_vgg_like_top__DOT__c2_start = 0U;
        vlSelfRef.__Vdly__d3_vgg_like_top__DOT__p1_start = 0U;
        vlSelfRef.__Vdly__d3_vgg_like_top__DOT__c3_start = 0U;
        vlSelfRef.__Vdly__d3_vgg_like_top__DOT__c4_start = 0U;
        vlSelfRef.__Vdly__d3_vgg_like_top__DOT__p2_start = 0U;
        vlSelfRef.__Vdly__d3_vgg_like_top__DOT__c5_start = 0U;
        vlSelfRef.__Vdly__d3_vgg_like_top__DOT__gap_start = 0U;
        __Vdly__d3_vgg_like_top__DOT__fc_start = 0U;
        vlSelfRef.argmax_valid = 0U;
        vlSelfRef.busy = 0U;
        vlSelfRef.done = 0U;
        vlSelfRef.__Vdly__state_dbg = 0U;
        vlSelfRef.d3_vgg_like_top__DOT__linear__DOT__class_idx = 0U;
        __Vdly__d3_vgg_like_top__DOT__linear__DOT__state = 0U;
        vlSelfRef.__Vdly__d3_vgg_like_top__DOT__linear__DOT__feature_idx = 0U;
        __Vdly__d3_vgg_like_top__DOT__linear__DOT__acc = 0U;
        vlSelfRef.d3_vgg_like_top__DOT__logit_index = 0U;
        vlSelfRef.d3_vgg_like_top__DOT__logit_data = 0U;
        vlSelfRef.d3_vgg_like_top__DOT__logit_valid = 0U;
        vlSelfRef.d3_vgg_like_top__DOT__fc_busy = 0U;
        vlSelfRef.d3_vgg_like_top__DOT__fc_done = 0U;
    }
    vlSelfRef.d3_vgg_like_top__DOT__fsm__DOT__state 
        = __Vdly__d3_vgg_like_top__DOT__fsm__DOT__state;
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
    vlSelfRef.d3_vgg_like_top__DOT__conv2__DOT__controller_done_d2 
        = ((IData)(vlSelfRef.rst_n) && (IData)(vlSelfRef.d3_vgg_like_top__DOT__conv2__DOT__controller_done_d1));
    vlSelfRef.d3_vgg_like_top__DOT__conv3__DOT__controller_done_d2 
        = ((IData)(vlSelfRef.rst_n) && (IData)(vlSelfRef.d3_vgg_like_top__DOT__conv3__DOT__controller_done_d1));
    vlSelfRef.d3_vgg_like_top__DOT__conv1__DOT__controller_done_d2 
        = ((IData)(vlSelfRef.rst_n) && (IData)(vlSelfRef.d3_vgg_like_top__DOT__conv1__DOT__controller_done_d1));
    vlSelfRef.d3_vgg_like_top__DOT__conv4__DOT__controller_done_d2 
        = ((IData)(vlSelfRef.rst_n) && (IData)(vlSelfRef.d3_vgg_like_top__DOT__conv4__DOT__controller_done_d1));
    vlSelfRef.d3_vgg_like_top__DOT__conv5__DOT__controller_done_d2 
        = ((IData)(vlSelfRef.rst_n) && (IData)(vlSelfRef.d3_vgg_like_top__DOT__conv5__DOT__controller_done_d1));
    vlSelfRef.d3_vgg_like_top__DOT__fc_start = __Vdly__d3_vgg_like_top__DOT__fc_start;
    vlSelfRef.d3_vgg_like_top__DOT__linear__DOT__state 
        = __Vdly__d3_vgg_like_top__DOT__linear__DOT__state;
    vlSelfRef.d3_vgg_like_top__DOT__linear__DOT__acc 
        = __Vdly__d3_vgg_like_top__DOT__linear__DOT__acc;
    vlSelfRef.d3_vgg_like_top__DOT__conv2__DOT__controller_done_d1 
        = ((IData)(vlSelfRef.rst_n) && (IData)(vlSelfRef.d3_vgg_like_top__DOT__conv2__DOT__controller_done));
    vlSelfRef.d3_vgg_like_top__DOT__conv3__DOT__controller_done_d1 
        = ((IData)(vlSelfRef.rst_n) && (IData)(vlSelfRef.d3_vgg_like_top__DOT__conv3__DOT__controller_done));
    vlSelfRef.d3_vgg_like_top__DOT__conv1__DOT__controller_done_d1 
        = ((IData)(vlSelfRef.rst_n) && (IData)(vlSelfRef.d3_vgg_like_top__DOT__conv1__DOT__controller_done));
    vlSelfRef.d3_vgg_like_top__DOT__conv4__DOT__controller_done_d1 
        = ((IData)(vlSelfRef.rst_n) && (IData)(vlSelfRef.d3_vgg_like_top__DOT__conv4__DOT__controller_done));
    vlSelfRef.d3_vgg_like_top__DOT__conv5__DOT__controller_done_d1 
        = ((IData)(vlSelfRef.rst_n) && (IData)(vlSelfRef.d3_vgg_like_top__DOT__conv5__DOT__controller_done));
}

void Vd3_vgg_like_top___024root___nba_sequent__TOP__1(Vd3_vgg_like_top___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vd3_vgg_like_top___024root___nba_sequent__TOP__1\n"); );
    Vd3_vgg_like_top__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    vlSelfRef.__VdlySet__d3_vgg_like_top__DOT__buffer_a__DOT__mem__v0 = 0U;
    vlSelfRef.__VdlySet__d3_vgg_like_top__DOT__buffer_b__DOT__mem__v0 = 0U;
    if (((IData)(vlSelfRef.__VdfgRegularize_h6e95ff9d_0_0) 
         | ((IData)(vlSelfRef.d3_vgg_like_top__DOT__conv2__DOT__postprocess_valid) 
            | ((IData)(vlSelfRef.d3_vgg_like_top__DOT__conv3__DOT__postprocess_valid) 
               | ((IData)(vlSelfRef.d3_vgg_like_top__DOT__p2_owe) 
                  | (IData)(vlSelfRef.d3_vgg_like_top__DOT__gap_owe)))))) {
        if (vlSelfRef.__VdfgRegularize_h6e95ff9d_0_0) {
            vlSelfRef.__VdlyVal__d3_vgg_like_top__DOT__buffer_a__DOT__mem__v0 
                = vlSelfRef.input_data;
            vlSelfRef.__VdlyDim0__d3_vgg_like_top__DOT__buffer_a__DOT__mem__v0 
                = (0x00007fffU & vlSelfRef.input_addr);
        } else if (vlSelfRef.d3_vgg_like_top__DOT__conv2__DOT__postprocess_valid) {
            vlSelfRef.__VdlyVal__d3_vgg_like_top__DOT__buffer_a__DOT__mem__v0 
                = vlSelfRef.d3_vgg_like_top__DOT__conv2__DOT__out_data_reg;
            vlSelfRef.__VdlyDim0__d3_vgg_like_top__DOT__buffer_a__DOT__mem__v0 
                = (0x00007fffU & vlSelfRef.d3_vgg_like_top__DOT__conv2__DOT__out_addr_reg);
        } else if (vlSelfRef.d3_vgg_like_top__DOT__conv3__DOT__postprocess_valid) {
            vlSelfRef.__VdlyVal__d3_vgg_like_top__DOT__buffer_a__DOT__mem__v0 
                = vlSelfRef.d3_vgg_like_top__DOT__conv3__DOT__out_data_reg;
            vlSelfRef.__VdlyDim0__d3_vgg_like_top__DOT__buffer_a__DOT__mem__v0 
                = (0x00007fffU & vlSelfRef.d3_vgg_like_top__DOT__conv3__DOT__out_addr_reg);
        } else if (vlSelfRef.d3_vgg_like_top__DOT__p2_owe) {
            vlSelfRef.__VdlyVal__d3_vgg_like_top__DOT__buffer_a__DOT__mem__v0 
                = vlSelfRef.d3_vgg_like_top__DOT__p2_od;
            vlSelfRef.__VdlyDim0__d3_vgg_like_top__DOT__buffer_a__DOT__mem__v0 
                = (0x00007fffU & vlSelfRef.d3_vgg_like_top__DOT__p2_oa);
        } else {
            vlSelfRef.__VdlyVal__d3_vgg_like_top__DOT__buffer_a__DOT__mem__v0 
                = vlSelfRef.d3_vgg_like_top__DOT__gap_od;
            vlSelfRef.__VdlyDim0__d3_vgg_like_top__DOT__buffer_a__DOT__mem__v0 
                = (0x00007fffU & vlSelfRef.d3_vgg_like_top__DOT__gap_oa);
        }
        vlSelfRef.__VdlySet__d3_vgg_like_top__DOT__buffer_a__DOT__mem__v0 = 1U;
    }
    if (((IData)(vlSelfRef.d3_vgg_like_top__DOT__conv1__DOT__postprocess_valid) 
         | ((IData)(vlSelfRef.d3_vgg_like_top__DOT__p1_owe) 
            | ((IData)(vlSelfRef.d3_vgg_like_top__DOT__conv4__DOT__postprocess_valid) 
               | (IData)(vlSelfRef.d3_vgg_like_top__DOT__conv5__DOT__postprocess_valid))))) {
        if (vlSelfRef.d3_vgg_like_top__DOT__conv1__DOT__postprocess_valid) {
            vlSelfRef.__VdlyVal__d3_vgg_like_top__DOT__buffer_b__DOT__mem__v0 
                = vlSelfRef.d3_vgg_like_top__DOT__conv1__DOT__out_data_reg;
            vlSelfRef.__VdlyDim0__d3_vgg_like_top__DOT__buffer_b__DOT__mem__v0 
                = (0x00007fffU & vlSelfRef.d3_vgg_like_top__DOT__conv1__DOT__out_addr_reg);
        } else if (vlSelfRef.d3_vgg_like_top__DOT__p1_owe) {
            vlSelfRef.__VdlyVal__d3_vgg_like_top__DOT__buffer_b__DOT__mem__v0 
                = vlSelfRef.d3_vgg_like_top__DOT__p1_od;
            vlSelfRef.__VdlyDim0__d3_vgg_like_top__DOT__buffer_b__DOT__mem__v0 
                = (0x00007fffU & vlSelfRef.d3_vgg_like_top__DOT__p1_oa);
        } else if (vlSelfRef.d3_vgg_like_top__DOT__conv4__DOT__postprocess_valid) {
            vlSelfRef.__VdlyVal__d3_vgg_like_top__DOT__buffer_b__DOT__mem__v0 
                = vlSelfRef.d3_vgg_like_top__DOT__conv4__DOT__out_data_reg;
            vlSelfRef.__VdlyDim0__d3_vgg_like_top__DOT__buffer_b__DOT__mem__v0 
                = (0x00007fffU & vlSelfRef.d3_vgg_like_top__DOT__conv4__DOT__out_addr_reg);
        } else {
            vlSelfRef.__VdlyVal__d3_vgg_like_top__DOT__buffer_b__DOT__mem__v0 
                = vlSelfRef.d3_vgg_like_top__DOT__conv5__DOT__out_data_reg;
            vlSelfRef.__VdlyDim0__d3_vgg_like_top__DOT__buffer_b__DOT__mem__v0 
                = (0x00007fffU & vlSelfRef.d3_vgg_like_top__DOT__conv5__DOT__out_addr_reg);
        }
        vlSelfRef.__VdlySet__d3_vgg_like_top__DOT__buffer_b__DOT__mem__v0 = 1U;
    }
    vlSelfRef.d3_vgg_like_top__DOT__fc_wd = ((0x04ffU 
                                              >= (IData)(vlSelfRef.d3_vgg_like_top__DOT__rwf__DOT__rom_addr))
                                              ? vlSelfRef.d3_vgg_like_top__DOT__rwf__DOT__rom
                                             [vlSelfRef.d3_vgg_like_top__DOT__rwf__DOT__rom_addr]
                                              : 0U);
}

void Vd3_vgg_like_top___024root___nba_sequent__TOP__2(Vd3_vgg_like_top___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vd3_vgg_like_top___024root___nba_sequent__TOP__2\n"); );
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
    QData/*42:0*/ d3_vgg_like_top__DOT__conv5__DOT__quant__DOT__result43;
    d3_vgg_like_top__DOT__conv5__DOT__quant__DOT__result43 = 0;
    QData/*42:0*/ d3_vgg_like_top__DOT__conv5__DOT__quant__DOT__gen_round_shift__DOT__shifted_magnitude;
    d3_vgg_like_top__DOT__conv5__DOT__quant__DOT__gen_round_shift__DOT__shifted_magnitude = 0;
    // Body
    if (vlSelfRef.rst_n) {
        vlSelfRef.d3_vgg_like_top__DOT__gap_owe = 0U;
        vlSelfRef.d3_vgg_like_top__DOT__gap_done = 0U;
        if ((4U & (IData)(vlSelfRef.d3_vgg_like_top__DOT__gap__DOT__state))) {
            if ((2U & (IData)(vlSelfRef.d3_vgg_like_top__DOT__gap__DOT__state))) {
                vlSelfRef.__Vdly__d3_vgg_like_top__DOT__gap__DOT__state = 0U;
            } else if ((1U & (IData)(vlSelfRef.d3_vgg_like_top__DOT__gap__DOT__state))) {
                vlSelfRef.__Vdly__d3_vgg_like_top__DOT__gap__DOT__state = 0U;
            } else {
                vlSelfRef.d3_vgg_like_top__DOT__gap_busy = 0U;
                vlSelfRef.d3_vgg_like_top__DOT__gap_done = 1U;
                vlSelfRef.__Vdly__d3_vgg_like_top__DOT__gap__DOT__state = 0U;
            }
        } else if ((2U & (IData)(vlSelfRef.d3_vgg_like_top__DOT__gap__DOT__state))) {
            if ((1U & (IData)(vlSelfRef.d3_vgg_like_top__DOT__gap__DOT__state))) {
                vlSelfRef.d3_vgg_like_top__DOT__gap_oa 
                    = (0x0001ffffU & (IData)(vlSelfRef.d3_vgg_like_top__DOT__gap__DOT__ch));
                vlSelfRef.d3_vgg_like_top__DOT__gap_od 
                    = (0x000000ffU & (vlSelfRef.d3_vgg_like_top__DOT__gap__DOT__sum 
                                      >> 6U));
                vlSelfRef.d3_vgg_like_top__DOT__gap_owe = 1U;
                vlSelfRef.__Vdly__d3_vgg_like_top__DOT__gap__DOT__idx = 0U;
                if ((0x007fU == (IData)(vlSelfRef.d3_vgg_like_top__DOT__gap__DOT__ch))) {
                    vlSelfRef.__Vdly__d3_vgg_like_top__DOT__gap__DOT__state = 4U;
                } else {
                    vlSelfRef.__Vdly__d3_vgg_like_top__DOT__gap__DOT__ch 
                        = (0x0000ffffU & ((IData)(1U) 
                                          + (IData)(vlSelfRef.d3_vgg_like_top__DOT__gap__DOT__ch)));
                    vlSelfRef.__Vdly__d3_vgg_like_top__DOT__gap__DOT__state = 1U;
                }
                vlSelfRef.__Vdly__d3_vgg_like_top__DOT__gap__DOT__sum = 0U;
            } else {
                vlSelfRef.__Vdly__d3_vgg_like_top__DOT__gap__DOT__sum 
                    = (vlSelfRef.d3_vgg_like_top__DOT__gap__DOT__sum 
                       + (((- (IData)((1U & ((IData)(vlSelfRef.d3_vgg_like_top__DOT__b_rd) 
                                             >> 7U)))) 
                           << 8U) | (IData)(vlSelfRef.d3_vgg_like_top__DOT__b_rd)));
                if ((0x003fU == (IData)(vlSelfRef.d3_vgg_like_top__DOT__gap__DOT__idx))) {
                    vlSelfRef.__Vdly__d3_vgg_like_top__DOT__gap__DOT__state = 3U;
                } else {
                    vlSelfRef.__Vdly__d3_vgg_like_top__DOT__gap__DOT__idx 
                        = (0x0000ffffU & ((IData)(1U) 
                                          + (IData)(vlSelfRef.d3_vgg_like_top__DOT__gap__DOT__idx)));
                    vlSelfRef.__Vdly__d3_vgg_like_top__DOT__gap__DOT__state = 1U;
                }
            }
        } else if ((1U & (IData)(vlSelfRef.d3_vgg_like_top__DOT__gap__DOT__state))) {
            vlSelfRef.__Vdly__d3_vgg_like_top__DOT__gap__DOT__state = 2U;
        } else {
            vlSelfRef.d3_vgg_like_top__DOT__gap_busy = 0U;
            if (vlSelfRef.d3_vgg_like_top__DOT__gap_start) {
                vlSelfRef.__Vdly__d3_vgg_like_top__DOT__gap__DOT__ch = 0U;
                vlSelfRef.__Vdly__d3_vgg_like_top__DOT__gap__DOT__idx = 0U;
                vlSelfRef.__Vdly__d3_vgg_like_top__DOT__gap__DOT__sum = 0U;
                vlSelfRef.d3_vgg_like_top__DOT__gap_busy = 1U;
                vlSelfRef.__Vdly__d3_vgg_like_top__DOT__gap__DOT__state = 1U;
            }
        }
        vlSelfRef.d3_vgg_like_top__DOT__gap_start = vlSelfRef.__Vdly__d3_vgg_like_top__DOT__gap_start;
        vlSelfRef.d3_vgg_like_top__DOT__gap__DOT__state 
            = vlSelfRef.__Vdly__d3_vgg_like_top__DOT__gap__DOT__state;
        vlSelfRef.d3_vgg_like_top__DOT__gap__DOT__sum 
            = vlSelfRef.__Vdly__d3_vgg_like_top__DOT__gap__DOT__sum;
        vlSelfRef.d3_vgg_like_top__DOT__p2_owe = 0U;
        vlSelfRef.d3_vgg_like_top__DOT__p2_done = 0U;
        if ((4U & (IData)(vlSelfRef.d3_vgg_like_top__DOT__pool2__DOT__state))) {
            if ((2U & (IData)(vlSelfRef.d3_vgg_like_top__DOT__pool2__DOT__state))) {
                vlSelfRef.__Vdly__d3_vgg_like_top__DOT__pool2__DOT__state = 0U;
            } else if ((1U & (IData)(vlSelfRef.d3_vgg_like_top__DOT__pool2__DOT__state))) {
                vlSelfRef.__Vdly__d3_vgg_like_top__DOT__pool2__DOT__state = 0U;
            } else {
                vlSelfRef.d3_vgg_like_top__DOT__p2_busy = 0U;
                vlSelfRef.d3_vgg_like_top__DOT__p2_done = 1U;
                vlSelfRef.__Vdly__d3_vgg_like_top__DOT__pool2__DOT__state = 0U;
            }
        } else if ((2U & (IData)(vlSelfRef.d3_vgg_like_top__DOT__pool2__DOT__state))) {
            if ((1U & (IData)(vlSelfRef.d3_vgg_like_top__DOT__pool2__DOT__state))) {
                vlSelfRef.d3_vgg_like_top__DOT__p2_oa 
                    = (0x0001ffffU & (VL_SHIFTL_III(17,32,32, (IData)(vlSelfRef.d3_vgg_like_top__DOT__pool2__DOT__ch), 6U) 
                                      + ((IData)(vlSelfRef.d3_vgg_like_top__DOT__pool2__DOT__col) 
                                         + ((IData)(vlSelfRef.d3_vgg_like_top__DOT__pool2__DOT__row) 
                                            << 3U))));
                vlSelfRef.d3_vgg_like_top__DOT__p2_od 
                    = vlSelfRef.d3_vgg_like_top__DOT__pool2__DOT__current_max;
                vlSelfRef.d3_vgg_like_top__DOT__p2_owe = 1U;
                vlSelfRef.__Vdly__d3_vgg_like_top__DOT__pool2__DOT__tap = 0U;
                if (((0x003fU == (IData)(vlSelfRef.d3_vgg_like_top__DOT__pool2__DOT__ch)) 
                     & ((7U == (IData)(vlSelfRef.d3_vgg_like_top__DOT__pool2__DOT__row)) 
                        & (7U == (IData)(vlSelfRef.d3_vgg_like_top__DOT__pool2__DOT__col))))) {
                    vlSelfRef.__Vdly__d3_vgg_like_top__DOT__pool2__DOT__state = 4U;
                } else {
                    if ((7U != (IData)(vlSelfRef.d3_vgg_like_top__DOT__pool2__DOT__col))) {
                        vlSelfRef.__Vdly__d3_vgg_like_top__DOT__pool2__DOT__col 
                            = (0x0000ffffU & ((IData)(1U) 
                                              + (IData)(vlSelfRef.d3_vgg_like_top__DOT__pool2__DOT__col)));
                    } else {
                        if ((7U != (IData)(vlSelfRef.d3_vgg_like_top__DOT__pool2__DOT__row))) {
                            vlSelfRef.__Vdly__d3_vgg_like_top__DOT__pool2__DOT__row 
                                = (0x0000ffffU & ((IData)(1U) 
                                                  + (IData)(vlSelfRef.d3_vgg_like_top__DOT__pool2__DOT__row)));
                        } else {
                            vlSelfRef.__Vdly__d3_vgg_like_top__DOT__pool2__DOT__ch 
                                = (0x0000ffffU & ((IData)(1U) 
                                                  + (IData)(vlSelfRef.d3_vgg_like_top__DOT__pool2__DOT__ch)));
                            vlSelfRef.__Vdly__d3_vgg_like_top__DOT__pool2__DOT__row = 0U;
                        }
                        vlSelfRef.__Vdly__d3_vgg_like_top__DOT__pool2__DOT__col = 0U;
                    }
                    vlSelfRef.__Vdly__d3_vgg_like_top__DOT__pool2__DOT__state = 1U;
                }
                vlSelfRef.__Vdly__d3_vgg_like_top__DOT__pool2__DOT__current_max = 0x80U;
            } else {
                if (((0U == (IData)(vlSelfRef.d3_vgg_like_top__DOT__pool2__DOT__tap)) 
                     | VL_GTS_III(8, (IData)(vlSelfRef.d3_vgg_like_top__DOT__b_rd), (IData)(vlSelfRef.d3_vgg_like_top__DOT__pool2__DOT__current_max)))) {
                    vlSelfRef.__Vdly__d3_vgg_like_top__DOT__pool2__DOT__current_max 
                        = vlSelfRef.d3_vgg_like_top__DOT__b_rd;
                }
                if ((3U == (IData)(vlSelfRef.d3_vgg_like_top__DOT__pool2__DOT__tap))) {
                    vlSelfRef.__Vdly__d3_vgg_like_top__DOT__pool2__DOT__state = 3U;
                } else {
                    vlSelfRef.__Vdly__d3_vgg_like_top__DOT__pool2__DOT__tap 
                        = (3U & ((IData)(1U) + (IData)(vlSelfRef.d3_vgg_like_top__DOT__pool2__DOT__tap)));
                    vlSelfRef.__Vdly__d3_vgg_like_top__DOT__pool2__DOT__state = 1U;
                }
            }
        } else if ((1U & (IData)(vlSelfRef.d3_vgg_like_top__DOT__pool2__DOT__state))) {
            vlSelfRef.__Vdly__d3_vgg_like_top__DOT__pool2__DOT__state = 2U;
        } else {
            vlSelfRef.d3_vgg_like_top__DOT__p2_busy = 0U;
            if (vlSelfRef.d3_vgg_like_top__DOT__p2_start) {
                vlSelfRef.__Vdly__d3_vgg_like_top__DOT__pool2__DOT__ch = 0U;
                vlSelfRef.__Vdly__d3_vgg_like_top__DOT__pool2__DOT__row = 0U;
                vlSelfRef.__Vdly__d3_vgg_like_top__DOT__pool2__DOT__col = 0U;
                vlSelfRef.__Vdly__d3_vgg_like_top__DOT__pool2__DOT__tap = 0U;
                vlSelfRef.__Vdly__d3_vgg_like_top__DOT__pool2__DOT__current_max = 0x80U;
                vlSelfRef.d3_vgg_like_top__DOT__p2_busy = 1U;
                vlSelfRef.__Vdly__d3_vgg_like_top__DOT__pool2__DOT__state = 1U;
            }
        }
        vlSelfRef.d3_vgg_like_top__DOT__p2_start = vlSelfRef.__Vdly__d3_vgg_like_top__DOT__p2_start;
        vlSelfRef.d3_vgg_like_top__DOT__pool2__DOT__state 
            = vlSelfRef.__Vdly__d3_vgg_like_top__DOT__pool2__DOT__state;
        vlSelfRef.d3_vgg_like_top__DOT__pool2__DOT__current_max 
            = vlSelfRef.__Vdly__d3_vgg_like_top__DOT__pool2__DOT__current_max;
        if (vlSelfRef.d3_vgg_like_top__DOT__conv2__DOT__product_valid) {
            vlSelfRef.d3_vgg_like_top__DOT__conv2__DOT__postprocess_valid = 1U;
            vlSelfRef.d3_vgg_like_top__DOT__conv2__DOT__out_data_reg 
                = (VL_LTS_III(32, 0x0000007fU, vlSelfRef.d3_vgg_like_top__DOT__conv2__DOT__activated)
                    ? 0x0000007fU : (VL_GTS_III(32, 0xffffff80U, vlSelfRef.d3_vgg_like_top__DOT__conv2__DOT__activated)
                                      ? 0x00000080U
                                      : (0x000000ffU 
                                         & vlSelfRef.d3_vgg_like_top__DOT__conv2__DOT__activated)));
            vlSelfRef.d3_vgg_like_top__DOT__conv2__DOT__out_addr_reg 
                = vlSelfRef.d3_vgg_like_top__DOT__conv2__DOT__product_addr_reg;
        } else {
            vlSelfRef.d3_vgg_like_top__DOT__conv2__DOT__postprocess_valid = 0U;
        }
        if (vlSelfRef.d3_vgg_like_top__DOT__conv3__DOT__product_valid) {
            vlSelfRef.d3_vgg_like_top__DOT__conv3__DOT__postprocess_valid = 1U;
            vlSelfRef.d3_vgg_like_top__DOT__conv3__DOT__out_data_reg 
                = (VL_LTS_III(32, 0x0000007fU, vlSelfRef.d3_vgg_like_top__DOT__conv3__DOT__activated)
                    ? 0x0000007fU : (VL_GTS_III(32, 0xffffff80U, vlSelfRef.d3_vgg_like_top__DOT__conv3__DOT__activated)
                                      ? 0x00000080U
                                      : (0x000000ffU 
                                         & vlSelfRef.d3_vgg_like_top__DOT__conv3__DOT__activated)));
            vlSelfRef.d3_vgg_like_top__DOT__conv3__DOT__out_addr_reg 
                = vlSelfRef.d3_vgg_like_top__DOT__conv3__DOT__product_addr_reg;
        } else {
            vlSelfRef.d3_vgg_like_top__DOT__conv3__DOT__postprocess_valid = 0U;
        }
        vlSelfRef.d3_vgg_like_top__DOT__p1_owe = 0U;
        vlSelfRef.d3_vgg_like_top__DOT__p1_done = 0U;
        if ((4U & (IData)(vlSelfRef.d3_vgg_like_top__DOT__pool1__DOT__state))) {
            if ((2U & (IData)(vlSelfRef.d3_vgg_like_top__DOT__pool1__DOT__state))) {
                vlSelfRef.__Vdly__d3_vgg_like_top__DOT__pool1__DOT__state = 0U;
            } else if ((1U & (IData)(vlSelfRef.d3_vgg_like_top__DOT__pool1__DOT__state))) {
                vlSelfRef.__Vdly__d3_vgg_like_top__DOT__pool1__DOT__state = 0U;
            } else {
                vlSelfRef.d3_vgg_like_top__DOT__p1_busy = 0U;
                vlSelfRef.d3_vgg_like_top__DOT__p1_done = 1U;
                vlSelfRef.__Vdly__d3_vgg_like_top__DOT__pool1__DOT__state = 0U;
            }
        } else if ((2U & (IData)(vlSelfRef.d3_vgg_like_top__DOT__pool1__DOT__state))) {
            if ((1U & (IData)(vlSelfRef.d3_vgg_like_top__DOT__pool1__DOT__state))) {
                vlSelfRef.d3_vgg_like_top__DOT__p1_oa 
                    = (0x0001ffffU & (VL_SHIFTL_III(17,32,32, (IData)(vlSelfRef.d3_vgg_like_top__DOT__pool1__DOT__ch), 8U) 
                                      + ((IData)(vlSelfRef.d3_vgg_like_top__DOT__pool1__DOT__col) 
                                         + ((IData)(vlSelfRef.d3_vgg_like_top__DOT__pool1__DOT__row) 
                                            << 4U))));
                vlSelfRef.d3_vgg_like_top__DOT__p1_od 
                    = vlSelfRef.d3_vgg_like_top__DOT__pool1__DOT__current_max;
                vlSelfRef.d3_vgg_like_top__DOT__p1_owe = 1U;
                vlSelfRef.__Vdly__d3_vgg_like_top__DOT__pool1__DOT__tap = 0U;
                if (((0x001fU == (IData)(vlSelfRef.d3_vgg_like_top__DOT__pool1__DOT__ch)) 
                     & ((0x000fU == (IData)(vlSelfRef.d3_vgg_like_top__DOT__pool1__DOT__row)) 
                        & (0x000fU == (IData)(vlSelfRef.d3_vgg_like_top__DOT__pool1__DOT__col))))) {
                    vlSelfRef.__Vdly__d3_vgg_like_top__DOT__pool1__DOT__state = 4U;
                } else {
                    if ((0x000fU != (IData)(vlSelfRef.d3_vgg_like_top__DOT__pool1__DOT__col))) {
                        vlSelfRef.d3_vgg_like_top__DOT__pool1__DOT__col 
                            = (0x0000ffffU & ((IData)(1U) 
                                              + (IData)(vlSelfRef.d3_vgg_like_top__DOT__pool1__DOT__col)));
                    } else {
                        if ((0x000fU != (IData)(vlSelfRef.d3_vgg_like_top__DOT__pool1__DOT__row))) {
                            vlSelfRef.d3_vgg_like_top__DOT__pool1__DOT__row 
                                = (0x0000ffffU & ((IData)(1U) 
                                                  + (IData)(vlSelfRef.d3_vgg_like_top__DOT__pool1__DOT__row)));
                        } else {
                            vlSelfRef.d3_vgg_like_top__DOT__pool1__DOT__ch 
                                = (0x0000ffffU & ((IData)(1U) 
                                                  + (IData)(vlSelfRef.d3_vgg_like_top__DOT__pool1__DOT__ch)));
                            vlSelfRef.d3_vgg_like_top__DOT__pool1__DOT__row = 0U;
                        }
                        vlSelfRef.d3_vgg_like_top__DOT__pool1__DOT__col = 0U;
                    }
                    vlSelfRef.__Vdly__d3_vgg_like_top__DOT__pool1__DOT__state = 1U;
                }
                vlSelfRef.__Vdly__d3_vgg_like_top__DOT__pool1__DOT__current_max = 0x80U;
            } else {
                if (((0U == (IData)(vlSelfRef.d3_vgg_like_top__DOT__pool1__DOT__tap)) 
                     | VL_GTS_III(8, (IData)(vlSelfRef.d3_vgg_like_top__DOT__a_rd), (IData)(vlSelfRef.d3_vgg_like_top__DOT__pool1__DOT__current_max)))) {
                    vlSelfRef.__Vdly__d3_vgg_like_top__DOT__pool1__DOT__current_max 
                        = vlSelfRef.d3_vgg_like_top__DOT__a_rd;
                }
                if ((3U == (IData)(vlSelfRef.d3_vgg_like_top__DOT__pool1__DOT__tap))) {
                    vlSelfRef.__Vdly__d3_vgg_like_top__DOT__pool1__DOT__state = 3U;
                } else {
                    vlSelfRef.__Vdly__d3_vgg_like_top__DOT__pool1__DOT__tap 
                        = (3U & ((IData)(1U) + (IData)(vlSelfRef.d3_vgg_like_top__DOT__pool1__DOT__tap)));
                    vlSelfRef.__Vdly__d3_vgg_like_top__DOT__pool1__DOT__state = 1U;
                }
            }
        } else if ((1U & (IData)(vlSelfRef.d3_vgg_like_top__DOT__pool1__DOT__state))) {
            vlSelfRef.__Vdly__d3_vgg_like_top__DOT__pool1__DOT__state = 2U;
        } else {
            vlSelfRef.d3_vgg_like_top__DOT__p1_busy = 0U;
            if (vlSelfRef.d3_vgg_like_top__DOT__p1_start) {
                vlSelfRef.d3_vgg_like_top__DOT__pool1__DOT__ch = 0U;
                vlSelfRef.d3_vgg_like_top__DOT__pool1__DOT__row = 0U;
                vlSelfRef.d3_vgg_like_top__DOT__pool1__DOT__col = 0U;
                vlSelfRef.__Vdly__d3_vgg_like_top__DOT__pool1__DOT__tap = 0U;
                vlSelfRef.__Vdly__d3_vgg_like_top__DOT__pool1__DOT__current_max = 0x80U;
                vlSelfRef.d3_vgg_like_top__DOT__p1_busy = 1U;
                vlSelfRef.__Vdly__d3_vgg_like_top__DOT__pool1__DOT__state = 1U;
            }
        }
        vlSelfRef.d3_vgg_like_top__DOT__p1_start = vlSelfRef.__Vdly__d3_vgg_like_top__DOT__p1_start;
        vlSelfRef.d3_vgg_like_top__DOT__pool1__DOT__state 
            = vlSelfRef.__Vdly__d3_vgg_like_top__DOT__pool1__DOT__state;
        vlSelfRef.d3_vgg_like_top__DOT__pool1__DOT__current_max 
            = vlSelfRef.__Vdly__d3_vgg_like_top__DOT__pool1__DOT__current_max;
        vlSelfRef.d3_vgg_like_top__DOT__pool1__DOT__tap 
            = vlSelfRef.__Vdly__d3_vgg_like_top__DOT__pool1__DOT__tap;
        if (vlSelfRef.d3_vgg_like_top__DOT__conv1__DOT__product_valid) {
            vlSelfRef.d3_vgg_like_top__DOT__conv1__DOT__postprocess_valid = 1U;
            vlSelfRef.d3_vgg_like_top__DOT__conv1__DOT__out_data_reg 
                = (VL_LTS_III(32, 0x0000007fU, vlSelfRef.d3_vgg_like_top__DOT__conv1__DOT__activated)
                    ? 0x0000007fU : (VL_GTS_III(32, 0xffffff80U, vlSelfRef.d3_vgg_like_top__DOT__conv1__DOT__activated)
                                      ? 0x00000080U
                                      : (0x000000ffU 
                                         & vlSelfRef.d3_vgg_like_top__DOT__conv1__DOT__activated)));
            vlSelfRef.d3_vgg_like_top__DOT__conv1__DOT__out_addr_reg 
                = vlSelfRef.d3_vgg_like_top__DOT__conv1__DOT__product_addr_reg;
        } else {
            vlSelfRef.d3_vgg_like_top__DOT__conv1__DOT__postprocess_valid = 0U;
        }
        if (vlSelfRef.d3_vgg_like_top__DOT__conv4__DOT__product_valid) {
            vlSelfRef.d3_vgg_like_top__DOT__conv4__DOT__postprocess_valid = 1U;
            vlSelfRef.d3_vgg_like_top__DOT__conv4__DOT__out_data_reg 
                = (VL_LTS_III(32, 0x0000007fU, vlSelfRef.d3_vgg_like_top__DOT__conv4__DOT__activated)
                    ? 0x0000007fU : (VL_GTS_III(32, 0xffffff80U, vlSelfRef.d3_vgg_like_top__DOT__conv4__DOT__activated)
                                      ? 0x00000080U
                                      : (0x000000ffU 
                                         & vlSelfRef.d3_vgg_like_top__DOT__conv4__DOT__activated)));
            vlSelfRef.d3_vgg_like_top__DOT__conv4__DOT__out_addr_reg 
                = vlSelfRef.d3_vgg_like_top__DOT__conv4__DOT__product_addr_reg;
        } else {
            vlSelfRef.d3_vgg_like_top__DOT__conv4__DOT__postprocess_valid = 0U;
        }
        if (vlSelfRef.d3_vgg_like_top__DOT__conv5__DOT__product_valid) {
            vlSelfRef.d3_vgg_like_top__DOT__conv5__DOT__postprocess_valid = 1U;
            vlSelfRef.d3_vgg_like_top__DOT__conv5__DOT__out_data_reg 
                = (VL_LTS_III(32, 0x0000007fU, vlSelfRef.d3_vgg_like_top__DOT__conv5__DOT__activated)
                    ? 0x0000007fU : (VL_GTS_III(32, 0xffffff80U, vlSelfRef.d3_vgg_like_top__DOT__conv5__DOT__activated)
                                      ? 0x00000080U
                                      : (0x000000ffU 
                                         & vlSelfRef.d3_vgg_like_top__DOT__conv5__DOT__activated)));
            vlSelfRef.d3_vgg_like_top__DOT__conv5__DOT__out_addr_reg 
                = vlSelfRef.d3_vgg_like_top__DOT__conv5__DOT__product_addr_reg;
        } else {
            vlSelfRef.d3_vgg_like_top__DOT__conv5__DOT__postprocess_valid = 0U;
        }
        if (vlSelfRef.d3_vgg_like_top__DOT__conv2__DOT__postprocess_pending) {
            vlSelfRef.d3_vgg_like_top__DOT__conv2__DOT__product_valid = 1U;
            vlSelfRef.d3_vgg_like_top__DOT__conv2__DOT__product_reg 
                = (0x000007ffffffffffULL & VL_MULS_QQQ(43, 0x00000000000095e1ULL, 
                                                       (0x000007ffffffffffULL 
                                                        & VL_EXTENDS_QI(43,25, 
                                                                        (0x01ffffffU 
                                                                         & vlSelfRef.d3_vgg_like_top__DOT__conv2__DOT__biased_sum_reg)))));
            vlSelfRef.d3_vgg_like_top__DOT__conv2__DOT__product_addr_reg 
                = vlSelfRef.d3_vgg_like_top__DOT__conv2__DOT__controller_out_addr;
        } else {
            vlSelfRef.d3_vgg_like_top__DOT__conv2__DOT__product_valid = 0U;
        }
    } else {
        vlSelfRef.__Vdly__d3_vgg_like_top__DOT__gap__DOT__ch = 0U;
        vlSelfRef.__Vdly__d3_vgg_like_top__DOT__gap__DOT__state = 0U;
        vlSelfRef.__Vdly__d3_vgg_like_top__DOT__gap__DOT__idx = 0U;
        vlSelfRef.__Vdly__d3_vgg_like_top__DOT__gap__DOT__sum = 0U;
        vlSelfRef.d3_vgg_like_top__DOT__gap_owe = 0U;
        vlSelfRef.d3_vgg_like_top__DOT__gap_oa = 0U;
        vlSelfRef.d3_vgg_like_top__DOT__gap_od = 0U;
        vlSelfRef.d3_vgg_like_top__DOT__gap_busy = 0U;
        vlSelfRef.d3_vgg_like_top__DOT__gap_done = 0U;
        vlSelfRef.d3_vgg_like_top__DOT__gap_start = vlSelfRef.__Vdly__d3_vgg_like_top__DOT__gap_start;
        vlSelfRef.d3_vgg_like_top__DOT__gap__DOT__state 
            = vlSelfRef.__Vdly__d3_vgg_like_top__DOT__gap__DOT__state;
        vlSelfRef.d3_vgg_like_top__DOT__gap__DOT__sum 
            = vlSelfRef.__Vdly__d3_vgg_like_top__DOT__gap__DOT__sum;
        vlSelfRef.__Vdly__d3_vgg_like_top__DOT__pool2__DOT__ch = 0U;
        vlSelfRef.__Vdly__d3_vgg_like_top__DOT__pool2__DOT__row = 0U;
        vlSelfRef.__Vdly__d3_vgg_like_top__DOT__pool2__DOT__col = 0U;
        vlSelfRef.__Vdly__d3_vgg_like_top__DOT__pool2__DOT__state = 0U;
        vlSelfRef.__Vdly__d3_vgg_like_top__DOT__pool2__DOT__tap = 0U;
        vlSelfRef.__Vdly__d3_vgg_like_top__DOT__pool2__DOT__current_max = 0x80U;
        vlSelfRef.d3_vgg_like_top__DOT__p2_od = 0U;
        vlSelfRef.d3_vgg_like_top__DOT__p2_oa = 0U;
        vlSelfRef.d3_vgg_like_top__DOT__p2_owe = 0U;
        vlSelfRef.d3_vgg_like_top__DOT__p2_busy = 0U;
        vlSelfRef.d3_vgg_like_top__DOT__p2_done = 0U;
        vlSelfRef.d3_vgg_like_top__DOT__p2_start = vlSelfRef.__Vdly__d3_vgg_like_top__DOT__p2_start;
        vlSelfRef.d3_vgg_like_top__DOT__pool2__DOT__state 
            = vlSelfRef.__Vdly__d3_vgg_like_top__DOT__pool2__DOT__state;
        vlSelfRef.d3_vgg_like_top__DOT__pool2__DOT__current_max 
            = vlSelfRef.__Vdly__d3_vgg_like_top__DOT__pool2__DOT__current_max;
        vlSelfRef.d3_vgg_like_top__DOT__conv2__DOT__postprocess_valid = 0U;
        vlSelfRef.d3_vgg_like_top__DOT__conv2__DOT__out_data_reg = 0U;
        vlSelfRef.d3_vgg_like_top__DOT__conv2__DOT__out_addr_reg = 0U;
        vlSelfRef.d3_vgg_like_top__DOT__conv3__DOT__postprocess_valid = 0U;
        vlSelfRef.d3_vgg_like_top__DOT__conv3__DOT__out_data_reg = 0U;
        vlSelfRef.d3_vgg_like_top__DOT__conv3__DOT__out_addr_reg = 0U;
        vlSelfRef.d3_vgg_like_top__DOT__pool1__DOT__ch = 0U;
        vlSelfRef.d3_vgg_like_top__DOT__pool1__DOT__row = 0U;
        vlSelfRef.d3_vgg_like_top__DOT__pool1__DOT__col = 0U;
        vlSelfRef.__Vdly__d3_vgg_like_top__DOT__pool1__DOT__state = 0U;
        vlSelfRef.__Vdly__d3_vgg_like_top__DOT__pool1__DOT__tap = 0U;
        vlSelfRef.__Vdly__d3_vgg_like_top__DOT__pool1__DOT__current_max = 0x80U;
        vlSelfRef.d3_vgg_like_top__DOT__p1_od = 0U;
        vlSelfRef.d3_vgg_like_top__DOT__p1_oa = 0U;
        vlSelfRef.d3_vgg_like_top__DOT__p1_owe = 0U;
        vlSelfRef.d3_vgg_like_top__DOT__p1_busy = 0U;
        vlSelfRef.d3_vgg_like_top__DOT__p1_done = 0U;
        vlSelfRef.d3_vgg_like_top__DOT__p1_start = vlSelfRef.__Vdly__d3_vgg_like_top__DOT__p1_start;
        vlSelfRef.d3_vgg_like_top__DOT__pool1__DOT__state 
            = vlSelfRef.__Vdly__d3_vgg_like_top__DOT__pool1__DOT__state;
        vlSelfRef.d3_vgg_like_top__DOT__pool1__DOT__current_max 
            = vlSelfRef.__Vdly__d3_vgg_like_top__DOT__pool1__DOT__current_max;
        vlSelfRef.d3_vgg_like_top__DOT__pool1__DOT__tap 
            = vlSelfRef.__Vdly__d3_vgg_like_top__DOT__pool1__DOT__tap;
        vlSelfRef.d3_vgg_like_top__DOT__conv1__DOT__postprocess_valid = 0U;
        vlSelfRef.d3_vgg_like_top__DOT__conv1__DOT__out_data_reg = 0U;
        vlSelfRef.d3_vgg_like_top__DOT__conv1__DOT__out_addr_reg = 0U;
        vlSelfRef.d3_vgg_like_top__DOT__conv4__DOT__postprocess_valid = 0U;
        vlSelfRef.d3_vgg_like_top__DOT__conv4__DOT__out_data_reg = 0U;
        vlSelfRef.d3_vgg_like_top__DOT__conv4__DOT__out_addr_reg = 0U;
        vlSelfRef.d3_vgg_like_top__DOT__conv5__DOT__postprocess_valid = 0U;
        vlSelfRef.d3_vgg_like_top__DOT__conv5__DOT__out_data_reg = 0U;
        vlSelfRef.d3_vgg_like_top__DOT__conv5__DOT__out_addr_reg = 0U;
        vlSelfRef.d3_vgg_like_top__DOT__conv2__DOT__product_reg = 0ULL;
        vlSelfRef.d3_vgg_like_top__DOT__conv2__DOT__product_valid = 0U;
        vlSelfRef.d3_vgg_like_top__DOT__conv2__DOT__product_addr_reg = 0U;
    }
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
    vlSelfRef.d3_vgg_like_top__DOT__conv2__DOT__postprocess_pending 
        = ((IData)(vlSelfRef.rst_n) && (IData)(vlSelfRef.d3_vgg_like_top__DOT__conv2__DOT__engine_done));
    if (vlSelfRef.rst_n) {
        if (vlSelfRef.d3_vgg_like_top__DOT__conv3__DOT__postprocess_pending) {
            vlSelfRef.d3_vgg_like_top__DOT__conv3__DOT__product_valid = 1U;
            vlSelfRef.d3_vgg_like_top__DOT__conv3__DOT__product_reg 
                = (0x000007ffffffffffULL & VL_MULS_QQQ(43, 0x000000000000e141ULL, 
                                                       (0x000007ffffffffffULL 
                                                        & VL_EXTENDS_QI(43,25, 
                                                                        (0x01ffffffU 
                                                                         & vlSelfRef.d3_vgg_like_top__DOT__conv3__DOT__biased_sum_reg)))));
            vlSelfRef.d3_vgg_like_top__DOT__conv3__DOT__product_addr_reg 
                = vlSelfRef.d3_vgg_like_top__DOT__conv3__DOT__controller_out_addr;
        } else {
            vlSelfRef.d3_vgg_like_top__DOT__conv3__DOT__product_valid = 0U;
        }
    } else {
        vlSelfRef.d3_vgg_like_top__DOT__conv3__DOT__product_reg = 0ULL;
        vlSelfRef.d3_vgg_like_top__DOT__conv3__DOT__product_valid = 0U;
        vlSelfRef.d3_vgg_like_top__DOT__conv3__DOT__product_addr_reg = 0U;
    }
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
    vlSelfRef.d3_vgg_like_top__DOT__conv3__DOT__postprocess_pending 
        = ((IData)(vlSelfRef.rst_n) && (IData)(vlSelfRef.d3_vgg_like_top__DOT__conv3__DOT__engine_done));
    if (vlSelfRef.rst_n) {
        if (vlSelfRef.d3_vgg_like_top__DOT__conv1__DOT__postprocess_pending) {
            vlSelfRef.d3_vgg_like_top__DOT__conv1__DOT__product_valid = 1U;
            vlSelfRef.d3_vgg_like_top__DOT__conv1__DOT__product_reg 
                = (0x000007ffffffffffULL & VL_MULS_QQQ(43, 0x000000000000bb3dULL, 
                                                       (0x000007ffffffffffULL 
                                                        & VL_EXTENDS_QI(43,25, 
                                                                        (0x01ffffffU 
                                                                         & vlSelfRef.d3_vgg_like_top__DOT__conv1__DOT__biased_sum_reg)))));
            vlSelfRef.d3_vgg_like_top__DOT__conv1__DOT__product_addr_reg 
                = vlSelfRef.d3_vgg_like_top__DOT__conv1__DOT__controller_out_addr;
        } else {
            vlSelfRef.d3_vgg_like_top__DOT__conv1__DOT__product_valid = 0U;
        }
    } else {
        vlSelfRef.d3_vgg_like_top__DOT__conv1__DOT__product_reg = 0ULL;
        vlSelfRef.d3_vgg_like_top__DOT__conv1__DOT__product_valid = 0U;
        vlSelfRef.d3_vgg_like_top__DOT__conv1__DOT__product_addr_reg = 0U;
    }
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
    vlSelfRef.d3_vgg_like_top__DOT__conv1__DOT__postprocess_pending 
        = ((IData)(vlSelfRef.rst_n) && (IData)(vlSelfRef.d3_vgg_like_top__DOT__conv1__DOT__engine_done));
    if (vlSelfRef.rst_n) {
        if (vlSelfRef.d3_vgg_like_top__DOT__conv4__DOT__postprocess_pending) {
            vlSelfRef.d3_vgg_like_top__DOT__conv4__DOT__product_valid = 1U;
            vlSelfRef.d3_vgg_like_top__DOT__conv4__DOT__product_reg 
                = (0x000007ffffffffffULL & VL_MULS_QQQ(43, 0x0000000000007834ULL, 
                                                       (0x000007ffffffffffULL 
                                                        & VL_EXTENDS_QI(43,25, 
                                                                        (0x01ffffffU 
                                                                         & vlSelfRef.d3_vgg_like_top__DOT__conv4__DOT__biased_sum_reg)))));
            vlSelfRef.d3_vgg_like_top__DOT__conv4__DOT__product_addr_reg 
                = vlSelfRef.d3_vgg_like_top__DOT__conv4__DOT__controller_out_addr;
        } else {
            vlSelfRef.d3_vgg_like_top__DOT__conv4__DOT__product_valid = 0U;
        }
    } else {
        vlSelfRef.d3_vgg_like_top__DOT__conv4__DOT__product_reg = 0ULL;
        vlSelfRef.d3_vgg_like_top__DOT__conv4__DOT__product_valid = 0U;
        vlSelfRef.d3_vgg_like_top__DOT__conv4__DOT__product_addr_reg = 0U;
    }
    d3_vgg_like_top__DOT__conv4__DOT__quant__DOT__gen_round_shift__DOT__shifted_magnitude 
        = (0x000007ffffffffffULL & VL_SHIFTRS_QQI(43,43,32, 
                                                  (0x000007ffffffffffULL 
                                                   & ((1U 
                                                       & (IData)(
                                                                 (vlSelfRef.d3_vgg_like_top__DOT__conv4__DOT__product_reg 
                                                                  >> 0x0000002aU)))
                                                       ? 
                                                      (0x0000000000800000ULL 
                                                       + 
                                                       (- vlSelfRef.d3_vgg_like_top__DOT__conv4__DOT__product_reg))
                                                       : 
                                                      (0x0000000000800000ULL 
                                                       + vlSelfRef.d3_vgg_like_top__DOT__conv4__DOT__product_reg))), 0x00000018U));
    vlSelfRef.d3_vgg_like_top__DOT__conv4__DOT__postprocess_pending 
        = ((IData)(vlSelfRef.rst_n) && (IData)(vlSelfRef.d3_vgg_like_top__DOT__conv4__DOT__engine_done));
    if (vlSelfRef.rst_n) {
        if (vlSelfRef.d3_vgg_like_top__DOT__conv5__DOT__postprocess_pending) {
            vlSelfRef.d3_vgg_like_top__DOT__conv5__DOT__product_valid = 1U;
            vlSelfRef.d3_vgg_like_top__DOT__conv5__DOT__product_reg 
                = (0x000007ffffffffffULL & VL_MULS_QQQ(43, 0x000000000000f191ULL, 
                                                       (0x000007ffffffffffULL 
                                                        & VL_EXTENDS_QI(43,25, 
                                                                        (0x01ffffffU 
                                                                         & vlSelfRef.d3_vgg_like_top__DOT__conv5__DOT__biased_sum_reg)))));
            vlSelfRef.d3_vgg_like_top__DOT__conv5__DOT__product_addr_reg 
                = vlSelfRef.d3_vgg_like_top__DOT__conv5__DOT__controller_out_addr;
        } else {
            vlSelfRef.d3_vgg_like_top__DOT__conv5__DOT__product_valid = 0U;
        }
    } else {
        vlSelfRef.d3_vgg_like_top__DOT__conv5__DOT__product_reg = 0ULL;
        vlSelfRef.d3_vgg_like_top__DOT__conv5__DOT__product_valid = 0U;
        vlSelfRef.d3_vgg_like_top__DOT__conv5__DOT__product_addr_reg = 0U;
    }
    d3_vgg_like_top__DOT__conv5__DOT__quant__DOT__gen_round_shift__DOT__shifted_magnitude 
        = (0x000007ffffffffffULL & VL_SHIFTRS_QQI(43,43,32, 
                                                  (0x000007ffffffffffULL 
                                                   & ((1U 
                                                       & (IData)(
                                                                 (vlSelfRef.d3_vgg_like_top__DOT__conv5__DOT__product_reg 
                                                                  >> 0x0000002aU)))
                                                       ? 
                                                      (0x0000000000800000ULL 
                                                       + 
                                                       (- vlSelfRef.d3_vgg_like_top__DOT__conv5__DOT__product_reg))
                                                       : 
                                                      (0x0000000000800000ULL 
                                                       + vlSelfRef.d3_vgg_like_top__DOT__conv5__DOT__product_reg))), 0x00000018U));
    vlSelfRef.d3_vgg_like_top__DOT__conv5__DOT__postprocess_pending 
        = ((IData)(vlSelfRef.rst_n) && (IData)(vlSelfRef.d3_vgg_like_top__DOT__conv5__DOT__engine_done));
    if (vlSelfRef.rst_n) {
        if (vlSelfRef.d3_vgg_like_top__DOT__conv2__DOT__engine_done) {
            vlSelfRef.d3_vgg_like_top__DOT__conv2__DOT__biased_sum_reg 
                = (vlSelfRef.d3_vgg_like_top__DOT__conv2__DOT__raw_sum 
                   + vlSelfRef.d3_vgg_like_top__DOT__rb2__DOT__rom
                   [(0x0000001fU & (IData)(vlSelfRef.d3_vgg_like_top__DOT__conv2__DOT__out_channel))]);
        }
        vlSelfRef.__Vdly__d3_vgg_like_top__DOT__conv2__DOT__engine_start = 0U;
        vlSelfRef.d3_vgg_like_top__DOT__conv2__DOT__controller_done = 0U;
        if ((2U & (IData)(vlSelfRef.d3_vgg_like_top__DOT__conv2__DOT__controller__DOT__state))) {
            if ((1U & (IData)(vlSelfRef.d3_vgg_like_top__DOT__conv2__DOT__controller__DOT__state))) {
                vlSelfRef.d3_vgg_like_top__DOT__conv2__DOT__controller_busy = 0U;
                vlSelfRef.d3_vgg_like_top__DOT__conv2__DOT__controller_done = 1U;
                vlSelfRef.__Vdly__d3_vgg_like_top__DOT__conv2__DOT__controller__DOT__state = 0U;
            } else if (vlSelfRef.d3_vgg_like_top__DOT__conv2__DOT__engine_done) {
                vlSelfRef.d3_vgg_like_top__DOT__conv2__DOT__controller_out_addr 
                    = (0x0001ffffU & (VL_SHIFTL_III(17,32,32, (IData)(vlSelfRef.d3_vgg_like_top__DOT__conv2__DOT__out_channel), 0x0000000aU) 
                                      + (VL_SHIFTL_III(17,32,32, (IData)(vlSelfRef.d3_vgg_like_top__DOT__conv2__DOT__out_row), 5U) 
                                         + (IData)(vlSelfRef.d3_vgg_like_top__DOT__conv2__DOT__out_col))));
                if (((0x001fU == (IData)(vlSelfRef.d3_vgg_like_top__DOT__conv2__DOT__out_channel)) 
                     & ((0x001fU == (IData)(vlSelfRef.d3_vgg_like_top__DOT__conv2__DOT__out_row)) 
                        & (0x001fU == (IData)(vlSelfRef.d3_vgg_like_top__DOT__conv2__DOT__out_col))))) {
                    vlSelfRef.__Vdly__d3_vgg_like_top__DOT__conv2__DOT__controller__DOT__state = 3U;
                } else {
                    if ((0x001fU != (IData)(vlSelfRef.d3_vgg_like_top__DOT__conv2__DOT__out_col))) {
                        vlSelfRef.d3_vgg_like_top__DOT__conv2__DOT__out_col 
                            = (0x0000ffffU & ((IData)(1U) 
                                              + (IData)(vlSelfRef.d3_vgg_like_top__DOT__conv2__DOT__out_col)));
                    } else {
                        if ((0x001fU != (IData)(vlSelfRef.d3_vgg_like_top__DOT__conv2__DOT__out_row))) {
                            vlSelfRef.d3_vgg_like_top__DOT__conv2__DOT__out_row 
                                = (0x0000ffffU & ((IData)(1U) 
                                                  + (IData)(vlSelfRef.d3_vgg_like_top__DOT__conv2__DOT__out_row)));
                        } else {
                            vlSelfRef.d3_vgg_like_top__DOT__conv2__DOT__out_channel 
                                = (0x0000ffffU & ((IData)(1U) 
                                                  + (IData)(vlSelfRef.d3_vgg_like_top__DOT__conv2__DOT__out_channel)));
                            vlSelfRef.d3_vgg_like_top__DOT__conv2__DOT__out_row = 0U;
                        }
                        vlSelfRef.d3_vgg_like_top__DOT__conv2__DOT__out_col = 0U;
                    }
                    vlSelfRef.__Vdly__d3_vgg_like_top__DOT__conv2__DOT__controller__DOT__state = 1U;
                }
            }
        } else if ((1U & (IData)(vlSelfRef.d3_vgg_like_top__DOT__conv2__DOT__controller__DOT__state))) {
            vlSelfRef.__Vdly__d3_vgg_like_top__DOT__conv2__DOT__engine_start = 1U;
            vlSelfRef.__Vdly__d3_vgg_like_top__DOT__conv2__DOT__controller__DOT__state = 2U;
        } else {
            vlSelfRef.d3_vgg_like_top__DOT__conv2__DOT__controller_busy = 0U;
            if (vlSelfRef.d3_vgg_like_top__DOT__c2_start) {
                vlSelfRef.d3_vgg_like_top__DOT__conv2__DOT__out_row = 0U;
                vlSelfRef.d3_vgg_like_top__DOT__conv2__DOT__out_col = 0U;
                vlSelfRef.d3_vgg_like_top__DOT__conv2__DOT__out_channel = 0U;
                vlSelfRef.d3_vgg_like_top__DOT__conv2__DOT__controller_busy = 1U;
                vlSelfRef.__Vdly__d3_vgg_like_top__DOT__conv2__DOT__controller__DOT__state = 1U;
            }
        }
        if (vlSelfRef.d3_vgg_like_top__DOT__conv3__DOT__engine_done) {
            vlSelfRef.d3_vgg_like_top__DOT__conv3__DOT__biased_sum_reg 
                = (vlSelfRef.d3_vgg_like_top__DOT__conv3__DOT__raw_sum 
                   + vlSelfRef.d3_vgg_like_top__DOT__rb3__DOT__rom
                   [(0x0000003fU & (IData)(vlSelfRef.d3_vgg_like_top__DOT__conv3__DOT__out_channel))]);
        }
        vlSelfRef.__Vdly__d3_vgg_like_top__DOT__conv3__DOT__engine_start = 0U;
        vlSelfRef.d3_vgg_like_top__DOT__conv3__DOT__controller_done = 0U;
        if ((2U & (IData)(vlSelfRef.d3_vgg_like_top__DOT__conv3__DOT__controller__DOT__state))) {
            if ((1U & (IData)(vlSelfRef.d3_vgg_like_top__DOT__conv3__DOT__controller__DOT__state))) {
                vlSelfRef.d3_vgg_like_top__DOT__conv3__DOT__controller_busy = 0U;
                vlSelfRef.d3_vgg_like_top__DOT__conv3__DOT__controller_done = 1U;
                vlSelfRef.__Vdly__d3_vgg_like_top__DOT__conv3__DOT__controller__DOT__state = 0U;
            } else if (vlSelfRef.d3_vgg_like_top__DOT__conv3__DOT__engine_done) {
                vlSelfRef.d3_vgg_like_top__DOT__conv3__DOT__controller_out_addr 
                    = (0x0001ffffU & (VL_SHIFTL_III(17,32,32, (IData)(vlSelfRef.d3_vgg_like_top__DOT__conv3__DOT__out_channel), 8U) 
                                      + (VL_SHIFTL_III(17,32,32, (IData)(vlSelfRef.d3_vgg_like_top__DOT__conv3__DOT__out_row), 4U) 
                                         + (IData)(vlSelfRef.d3_vgg_like_top__DOT__conv3__DOT__out_col))));
                if (((0x003fU == (IData)(vlSelfRef.d3_vgg_like_top__DOT__conv3__DOT__out_channel)) 
                     & ((0x000fU == (IData)(vlSelfRef.d3_vgg_like_top__DOT__conv3__DOT__out_row)) 
                        & (0x000fU == (IData)(vlSelfRef.d3_vgg_like_top__DOT__conv3__DOT__out_col))))) {
                    vlSelfRef.__Vdly__d3_vgg_like_top__DOT__conv3__DOT__controller__DOT__state = 3U;
                } else {
                    if ((0x000fU != (IData)(vlSelfRef.d3_vgg_like_top__DOT__conv3__DOT__out_col))) {
                        vlSelfRef.d3_vgg_like_top__DOT__conv3__DOT__out_col 
                            = (0x0000ffffU & ((IData)(1U) 
                                              + (IData)(vlSelfRef.d3_vgg_like_top__DOT__conv3__DOT__out_col)));
                    } else {
                        if ((0x000fU != (IData)(vlSelfRef.d3_vgg_like_top__DOT__conv3__DOT__out_row))) {
                            vlSelfRef.d3_vgg_like_top__DOT__conv3__DOT__out_row 
                                = (0x0000ffffU & ((IData)(1U) 
                                                  + (IData)(vlSelfRef.d3_vgg_like_top__DOT__conv3__DOT__out_row)));
                        } else {
                            vlSelfRef.d3_vgg_like_top__DOT__conv3__DOT__out_channel 
                                = (0x0000ffffU & ((IData)(1U) 
                                                  + (IData)(vlSelfRef.d3_vgg_like_top__DOT__conv3__DOT__out_channel)));
                            vlSelfRef.d3_vgg_like_top__DOT__conv3__DOT__out_row = 0U;
                        }
                        vlSelfRef.d3_vgg_like_top__DOT__conv3__DOT__out_col = 0U;
                    }
                    vlSelfRef.__Vdly__d3_vgg_like_top__DOT__conv3__DOT__controller__DOT__state = 1U;
                }
            }
        } else if ((1U & (IData)(vlSelfRef.d3_vgg_like_top__DOT__conv3__DOT__controller__DOT__state))) {
            vlSelfRef.__Vdly__d3_vgg_like_top__DOT__conv3__DOT__engine_start = 1U;
            vlSelfRef.__Vdly__d3_vgg_like_top__DOT__conv3__DOT__controller__DOT__state = 2U;
        } else {
            vlSelfRef.d3_vgg_like_top__DOT__conv3__DOT__controller_busy = 0U;
            if (vlSelfRef.d3_vgg_like_top__DOT__c3_start) {
                vlSelfRef.d3_vgg_like_top__DOT__conv3__DOT__out_row = 0U;
                vlSelfRef.d3_vgg_like_top__DOT__conv3__DOT__out_col = 0U;
                vlSelfRef.d3_vgg_like_top__DOT__conv3__DOT__out_channel = 0U;
                vlSelfRef.d3_vgg_like_top__DOT__conv3__DOT__controller_busy = 1U;
                vlSelfRef.__Vdly__d3_vgg_like_top__DOT__conv3__DOT__controller__DOT__state = 1U;
            }
        }
        if (vlSelfRef.d3_vgg_like_top__DOT__conv1__DOT__engine_done) {
            vlSelfRef.d3_vgg_like_top__DOT__conv1__DOT__biased_sum_reg 
                = (vlSelfRef.d3_vgg_like_top__DOT__conv1__DOT__raw_sum 
                   + vlSelfRef.d3_vgg_like_top__DOT__rb1__DOT__rom
                   [(0x0000001fU & (IData)(vlSelfRef.d3_vgg_like_top__DOT__conv1__DOT__out_channel))]);
        }
        vlSelfRef.__Vdly__d3_vgg_like_top__DOT__conv1__DOT__engine_start = 0U;
        vlSelfRef.d3_vgg_like_top__DOT__conv1__DOT__controller_done = 0U;
        if ((2U & (IData)(vlSelfRef.d3_vgg_like_top__DOT__conv1__DOT__controller__DOT__state))) {
            if ((1U & (IData)(vlSelfRef.d3_vgg_like_top__DOT__conv1__DOT__controller__DOT__state))) {
                vlSelfRef.d3_vgg_like_top__DOT__conv1__DOT__controller_busy = 0U;
                vlSelfRef.d3_vgg_like_top__DOT__conv1__DOT__controller_done = 1U;
                vlSelfRef.__Vdly__d3_vgg_like_top__DOT__conv1__DOT__controller__DOT__state = 0U;
            } else if (vlSelfRef.d3_vgg_like_top__DOT__conv1__DOT__engine_done) {
                vlSelfRef.d3_vgg_like_top__DOT__conv1__DOT__controller_out_addr 
                    = (0x0001ffffU & (VL_SHIFTL_III(17,32,32, (IData)(vlSelfRef.d3_vgg_like_top__DOT__conv1__DOT__out_channel), 0x0000000aU) 
                                      + (VL_SHIFTL_III(17,32,32, (IData)(vlSelfRef.d3_vgg_like_top__DOT__conv1__DOT__out_row), 5U) 
                                         + (IData)(vlSelfRef.d3_vgg_like_top__DOT__conv1__DOT__out_col))));
                if (((0x001fU == (IData)(vlSelfRef.d3_vgg_like_top__DOT__conv1__DOT__out_channel)) 
                     & ((0x001fU == (IData)(vlSelfRef.d3_vgg_like_top__DOT__conv1__DOT__out_row)) 
                        & (0x001fU == (IData)(vlSelfRef.d3_vgg_like_top__DOT__conv1__DOT__out_col))))) {
                    vlSelfRef.__Vdly__d3_vgg_like_top__DOT__conv1__DOT__controller__DOT__state = 3U;
                } else {
                    if ((0x001fU != (IData)(vlSelfRef.d3_vgg_like_top__DOT__conv1__DOT__out_col))) {
                        vlSelfRef.d3_vgg_like_top__DOT__conv1__DOT__out_col 
                            = (0x0000ffffU & ((IData)(1U) 
                                              + (IData)(vlSelfRef.d3_vgg_like_top__DOT__conv1__DOT__out_col)));
                    } else {
                        if ((0x001fU != (IData)(vlSelfRef.d3_vgg_like_top__DOT__conv1__DOT__out_row))) {
                            vlSelfRef.d3_vgg_like_top__DOT__conv1__DOT__out_row 
                                = (0x0000ffffU & ((IData)(1U) 
                                                  + (IData)(vlSelfRef.d3_vgg_like_top__DOT__conv1__DOT__out_row)));
                        } else {
                            vlSelfRef.d3_vgg_like_top__DOT__conv1__DOT__out_channel 
                                = (0x0000ffffU & ((IData)(1U) 
                                                  + (IData)(vlSelfRef.d3_vgg_like_top__DOT__conv1__DOT__out_channel)));
                            vlSelfRef.d3_vgg_like_top__DOT__conv1__DOT__out_row = 0U;
                        }
                        vlSelfRef.d3_vgg_like_top__DOT__conv1__DOT__out_col = 0U;
                    }
                    vlSelfRef.__Vdly__d3_vgg_like_top__DOT__conv1__DOT__controller__DOT__state = 1U;
                }
            }
        } else if ((1U & (IData)(vlSelfRef.d3_vgg_like_top__DOT__conv1__DOT__controller__DOT__state))) {
            vlSelfRef.__Vdly__d3_vgg_like_top__DOT__conv1__DOT__engine_start = 1U;
            vlSelfRef.__Vdly__d3_vgg_like_top__DOT__conv1__DOT__controller__DOT__state = 2U;
        } else {
            vlSelfRef.d3_vgg_like_top__DOT__conv1__DOT__controller_busy = 0U;
            if (vlSelfRef.d3_vgg_like_top__DOT__c1_start) {
                vlSelfRef.d3_vgg_like_top__DOT__conv1__DOT__out_row = 0U;
                vlSelfRef.d3_vgg_like_top__DOT__conv1__DOT__out_col = 0U;
                vlSelfRef.d3_vgg_like_top__DOT__conv1__DOT__out_channel = 0U;
                vlSelfRef.d3_vgg_like_top__DOT__conv1__DOT__controller_busy = 1U;
                vlSelfRef.__Vdly__d3_vgg_like_top__DOT__conv1__DOT__controller__DOT__state = 1U;
            }
        }
        if (vlSelfRef.d3_vgg_like_top__DOT__conv4__DOT__engine_done) {
            vlSelfRef.d3_vgg_like_top__DOT__conv4__DOT__biased_sum_reg 
                = (vlSelfRef.d3_vgg_like_top__DOT__conv4__DOT__raw_sum 
                   + vlSelfRef.d3_vgg_like_top__DOT__rb4__DOT__rom
                   [(0x0000003fU & (IData)(vlSelfRef.d3_vgg_like_top__DOT__conv4__DOT__out_channel))]);
        }
        vlSelfRef.__Vdly__d3_vgg_like_top__DOT__conv4__DOT__engine_start = 0U;
        vlSelfRef.d3_vgg_like_top__DOT__conv4__DOT__controller_done = 0U;
        if ((2U & (IData)(vlSelfRef.d3_vgg_like_top__DOT__conv4__DOT__controller__DOT__state))) {
            if ((1U & (IData)(vlSelfRef.d3_vgg_like_top__DOT__conv4__DOT__controller__DOT__state))) {
                vlSelfRef.d3_vgg_like_top__DOT__conv4__DOT__controller_busy = 0U;
                vlSelfRef.d3_vgg_like_top__DOT__conv4__DOT__controller_done = 1U;
                vlSelfRef.__Vdly__d3_vgg_like_top__DOT__conv4__DOT__controller__DOT__state = 0U;
            } else if (vlSelfRef.d3_vgg_like_top__DOT__conv4__DOT__engine_done) {
                vlSelfRef.d3_vgg_like_top__DOT__conv4__DOT__controller_out_addr 
                    = (0x0001ffffU & (VL_SHIFTL_III(17,32,32, (IData)(vlSelfRef.d3_vgg_like_top__DOT__conv4__DOT__out_channel), 8U) 
                                      + (VL_SHIFTL_III(17,32,32, (IData)(vlSelfRef.d3_vgg_like_top__DOT__conv4__DOT__out_row), 4U) 
                                         + (IData)(vlSelfRef.d3_vgg_like_top__DOT__conv4__DOT__out_col))));
                if (((0x003fU == (IData)(vlSelfRef.d3_vgg_like_top__DOT__conv4__DOT__out_channel)) 
                     & ((0x000fU == (IData)(vlSelfRef.d3_vgg_like_top__DOT__conv4__DOT__out_row)) 
                        & (0x000fU == (IData)(vlSelfRef.d3_vgg_like_top__DOT__conv4__DOT__out_col))))) {
                    vlSelfRef.__Vdly__d3_vgg_like_top__DOT__conv4__DOT__controller__DOT__state = 3U;
                } else {
                    if ((0x000fU != (IData)(vlSelfRef.d3_vgg_like_top__DOT__conv4__DOT__out_col))) {
                        vlSelfRef.d3_vgg_like_top__DOT__conv4__DOT__out_col 
                            = (0x0000ffffU & ((IData)(1U) 
                                              + (IData)(vlSelfRef.d3_vgg_like_top__DOT__conv4__DOT__out_col)));
                    } else {
                        if ((0x000fU != (IData)(vlSelfRef.d3_vgg_like_top__DOT__conv4__DOT__out_row))) {
                            vlSelfRef.d3_vgg_like_top__DOT__conv4__DOT__out_row 
                                = (0x0000ffffU & ((IData)(1U) 
                                                  + (IData)(vlSelfRef.d3_vgg_like_top__DOT__conv4__DOT__out_row)));
                        } else {
                            vlSelfRef.d3_vgg_like_top__DOT__conv4__DOT__out_channel 
                                = (0x0000ffffU & ((IData)(1U) 
                                                  + (IData)(vlSelfRef.d3_vgg_like_top__DOT__conv4__DOT__out_channel)));
                            vlSelfRef.d3_vgg_like_top__DOT__conv4__DOT__out_row = 0U;
                        }
                        vlSelfRef.d3_vgg_like_top__DOT__conv4__DOT__out_col = 0U;
                    }
                    vlSelfRef.__Vdly__d3_vgg_like_top__DOT__conv4__DOT__controller__DOT__state = 1U;
                }
            }
        } else if ((1U & (IData)(vlSelfRef.d3_vgg_like_top__DOT__conv4__DOT__controller__DOT__state))) {
            vlSelfRef.__Vdly__d3_vgg_like_top__DOT__conv4__DOT__engine_start = 1U;
            vlSelfRef.__Vdly__d3_vgg_like_top__DOT__conv4__DOT__controller__DOT__state = 2U;
        } else {
            vlSelfRef.d3_vgg_like_top__DOT__conv4__DOT__controller_busy = 0U;
            if (vlSelfRef.d3_vgg_like_top__DOT__c4_start) {
                vlSelfRef.d3_vgg_like_top__DOT__conv4__DOT__out_row = 0U;
                vlSelfRef.d3_vgg_like_top__DOT__conv4__DOT__out_col = 0U;
                vlSelfRef.d3_vgg_like_top__DOT__conv4__DOT__out_channel = 0U;
                vlSelfRef.d3_vgg_like_top__DOT__conv4__DOT__controller_busy = 1U;
                vlSelfRef.__Vdly__d3_vgg_like_top__DOT__conv4__DOT__controller__DOT__state = 1U;
            }
        }
        if (vlSelfRef.d3_vgg_like_top__DOT__conv5__DOT__engine_done) {
            vlSelfRef.d3_vgg_like_top__DOT__conv5__DOT__biased_sum_reg 
                = (vlSelfRef.d3_vgg_like_top__DOT__conv5__DOT__raw_sum 
                   + vlSelfRef.d3_vgg_like_top__DOT__rb5__DOT__rom
                   [(0x0000007fU & (IData)(vlSelfRef.d3_vgg_like_top__DOT__conv5__DOT__out_channel))]);
        }
        vlSelfRef.__Vdly__d3_vgg_like_top__DOT__conv5__DOT__engine_start = 0U;
        vlSelfRef.d3_vgg_like_top__DOT__conv5__DOT__controller_done = 0U;
        if ((2U & (IData)(vlSelfRef.d3_vgg_like_top__DOT__conv5__DOT__controller__DOT__state))) {
            if ((1U & (IData)(vlSelfRef.d3_vgg_like_top__DOT__conv5__DOT__controller__DOT__state))) {
                vlSelfRef.d3_vgg_like_top__DOT__conv5__DOT__controller_busy = 0U;
                vlSelfRef.d3_vgg_like_top__DOT__conv5__DOT__controller_done = 1U;
                vlSelfRef.__Vdly__d3_vgg_like_top__DOT__conv5__DOT__controller__DOT__state = 0U;
            } else if (vlSelfRef.d3_vgg_like_top__DOT__conv5__DOT__engine_done) {
                vlSelfRef.d3_vgg_like_top__DOT__conv5__DOT__controller_out_addr 
                    = (0x0001ffffU & (vlSelfRef.__VdfgRegularize_h6e95ff9d_0_13 
                                      + (VL_SHIFTL_III(17,32,32, (IData)(vlSelfRef.d3_vgg_like_top__DOT__conv5__DOT__out_row), 3U) 
                                         + (IData)(vlSelfRef.d3_vgg_like_top__DOT__conv5__DOT__out_col))));
                if (((0x007fU == (IData)(vlSelfRef.d3_vgg_like_top__DOT__conv5__DOT__out_channel)) 
                     & ((7U == (IData)(vlSelfRef.d3_vgg_like_top__DOT__conv5__DOT__out_row)) 
                        & (7U == (IData)(vlSelfRef.d3_vgg_like_top__DOT__conv5__DOT__out_col))))) {
                    vlSelfRef.__Vdly__d3_vgg_like_top__DOT__conv5__DOT__controller__DOT__state = 3U;
                } else {
                    if ((7U != (IData)(vlSelfRef.d3_vgg_like_top__DOT__conv5__DOT__out_col))) {
                        vlSelfRef.d3_vgg_like_top__DOT__conv5__DOT__out_col 
                            = (0x0000ffffU & ((IData)(1U) 
                                              + (IData)(vlSelfRef.d3_vgg_like_top__DOT__conv5__DOT__out_col)));
                    } else {
                        if ((7U != (IData)(vlSelfRef.d3_vgg_like_top__DOT__conv5__DOT__out_row))) {
                            vlSelfRef.d3_vgg_like_top__DOT__conv5__DOT__out_row 
                                = (0x0000ffffU & ((IData)(1U) 
                                                  + (IData)(vlSelfRef.d3_vgg_like_top__DOT__conv5__DOT__out_row)));
                        } else {
                            vlSelfRef.d3_vgg_like_top__DOT__conv5__DOT__out_channel 
                                = (0x0000ffffU & ((IData)(1U) 
                                                  + (IData)(vlSelfRef.d3_vgg_like_top__DOT__conv5__DOT__out_channel)));
                            vlSelfRef.d3_vgg_like_top__DOT__conv5__DOT__out_row = 0U;
                        }
                        vlSelfRef.d3_vgg_like_top__DOT__conv5__DOT__out_col = 0U;
                    }
                    vlSelfRef.__Vdly__d3_vgg_like_top__DOT__conv5__DOT__controller__DOT__state = 1U;
                }
            }
        } else if ((1U & (IData)(vlSelfRef.d3_vgg_like_top__DOT__conv5__DOT__controller__DOT__state))) {
            vlSelfRef.__Vdly__d3_vgg_like_top__DOT__conv5__DOT__engine_start = 1U;
            vlSelfRef.__Vdly__d3_vgg_like_top__DOT__conv5__DOT__controller__DOT__state = 2U;
        } else {
            vlSelfRef.d3_vgg_like_top__DOT__conv5__DOT__controller_busy = 0U;
            if (vlSelfRef.d3_vgg_like_top__DOT__c5_start) {
                vlSelfRef.d3_vgg_like_top__DOT__conv5__DOT__out_row = 0U;
                vlSelfRef.d3_vgg_like_top__DOT__conv5__DOT__out_col = 0U;
                vlSelfRef.d3_vgg_like_top__DOT__conv5__DOT__out_channel = 0U;
                vlSelfRef.d3_vgg_like_top__DOT__conv5__DOT__controller_busy = 1U;
                vlSelfRef.__Vdly__d3_vgg_like_top__DOT__conv5__DOT__controller__DOT__state = 1U;
            }
        }
        vlSelfRef.d3_vgg_like_top__DOT__conv2__DOT__engine_done = 0U;
        if ((2U & (IData)(vlSelfRef.d3_vgg_like_top__DOT__conv2__DOT__engine__DOT__state))) {
            if ((1U & (IData)(vlSelfRef.d3_vgg_like_top__DOT__conv2__DOT__engine__DOT__state))) {
                vlSelfRef.d3_vgg_like_top__DOT__conv2__DOT__engine_done = 1U;
                vlSelfRef.__Vdly__d3_vgg_like_top__DOT__conv2__DOT__engine__DOT__state = 0U;
            } else {
                if ((1U & (~ (IData)(vlSelfRef.d3_vgg_like_top__DOT__conv2__DOT__engine__DOT__last_item)))) {
                    if ((2U != (IData)(vlSelfRef.d3_vgg_like_top__DOT__conv2__DOT__engine__DOT__kc))) {
                        vlSelfRef.d3_vgg_like_top__DOT__conv2__DOT__engine__DOT__kc 
                            = (3U & ((IData)(1U) + (IData)(vlSelfRef.d3_vgg_like_top__DOT__conv2__DOT__engine__DOT__kc)));
                    } else {
                        if ((2U != (IData)(vlSelfRef.d3_vgg_like_top__DOT__conv2__DOT__engine__DOT__kr))) {
                            vlSelfRef.d3_vgg_like_top__DOT__conv2__DOT__engine__DOT__kr 
                                = (3U & ((IData)(1U) 
                                         + (IData)(vlSelfRef.d3_vgg_like_top__DOT__conv2__DOT__engine__DOT__kr)));
                        } else {
                            vlSelfRef.__Vdly__d3_vgg_like_top__DOT__conv2__DOT__engine__DOT__ic 
                                = (0x0000ffffU & ((IData)(1U) 
                                                  + (IData)(vlSelfRef.d3_vgg_like_top__DOT__conv2__DOT__engine__DOT__ic)));
                            vlSelfRef.d3_vgg_like_top__DOT__conv2__DOT__engine__DOT__kr = 0U;
                        }
                        vlSelfRef.d3_vgg_like_top__DOT__conv2__DOT__engine__DOT__kc = 0U;
                    }
                }
                vlSelfRef.__Vdly__d3_vgg_like_top__DOT__conv2__DOT__engine__DOT__product_reg 
                    = (0x0000ffffU & (VL_MULS_III(16, 
                                                  (0x0000ffffU 
                                                   & VL_EXTENDS_II(16,8, (IData)(vlSelfRef.d3_vgg_like_top__DOT__b_rd))), 
                                                  (0x0000ffffU 
                                                   & VL_EXTENDS_II(16,8, (IData)(vlSelfRef.d3_vgg_like_top__DOT__c2_wd)))) 
                                      & (- (IData)((IData)(vlSelfRef.d3_vgg_like_top__DOT__conv2__DOT__engine__DOT__valid_pixel)))));
                vlSelfRef.__Vdly__d3_vgg_like_top__DOT__conv2__DOT__engine__DOT__product_valid = 1U;
                vlSelfRef.__Vdly__d3_vgg_like_top__DOT__conv2__DOT__engine__DOT__product_last 
                    = vlSelfRef.d3_vgg_like_top__DOT__conv2__DOT__engine__DOT__last_item;
                vlSelfRef.__Vdly__d3_vgg_like_top__DOT__conv2__DOT__engine__DOT__state = 1U;
            }
        } else if ((1U & (IData)(vlSelfRef.d3_vgg_like_top__DOT__conv2__DOT__engine__DOT__state))) {
            if (vlSelfRef.d3_vgg_like_top__DOT__conv2__DOT__engine__DOT__product_valid) {
                vlSelfRef.d3_vgg_like_top__DOT__conv2__DOT__raw_sum 
                    = (vlSelfRef.d3_vgg_like_top__DOT__conv2__DOT__raw_sum 
                       + (((- (IData)((1U & ((IData)(vlSelfRef.d3_vgg_like_top__DOT__conv2__DOT__engine__DOT__product_reg) 
                                             >> 0x0fU)))) 
                           << 0x00000010U) | (IData)(vlSelfRef.d3_vgg_like_top__DOT__conv2__DOT__engine__DOT__product_reg)));
                if (vlSelfRef.d3_vgg_like_top__DOT__conv2__DOT__engine__DOT__product_last) {
                    vlSelfRef.__Vdly__d3_vgg_like_top__DOT__conv2__DOT__engine__DOT__product_valid = 0U;
                    vlSelfRef.__Vdly__d3_vgg_like_top__DOT__conv2__DOT__engine__DOT__state = 3U;
                } else {
                    vlSelfRef.__Vdly__d3_vgg_like_top__DOT__conv2__DOT__engine__DOT__state = 2U;
                }
            } else {
                vlSelfRef.__Vdly__d3_vgg_like_top__DOT__conv2__DOT__engine__DOT__state = 2U;
            }
        } else if (vlSelfRef.d3_vgg_like_top__DOT__conv2__DOT__engine_start) {
            vlSelfRef.__Vdly__d3_vgg_like_top__DOT__conv2__DOT__engine__DOT__ic = 0U;
            vlSelfRef.d3_vgg_like_top__DOT__conv2__DOT__engine__DOT__kr = 0U;
            vlSelfRef.d3_vgg_like_top__DOT__conv2__DOT__engine__DOT__kc = 0U;
            vlSelfRef.d3_vgg_like_top__DOT__conv2__DOT__raw_sum = 0U;
            vlSelfRef.__Vdly__d3_vgg_like_top__DOT__conv2__DOT__engine__DOT__product_valid = 0U;
            vlSelfRef.__Vdly__d3_vgg_like_top__DOT__conv2__DOT__engine__DOT__product_last = 0U;
            vlSelfRef.__Vdly__d3_vgg_like_top__DOT__conv2__DOT__engine__DOT__state = 1U;
        }
        vlSelfRef.d3_vgg_like_top__DOT__conv3__DOT__engine_done = 0U;
        if ((2U & (IData)(vlSelfRef.d3_vgg_like_top__DOT__conv3__DOT__engine__DOT__state))) {
            if ((1U & (IData)(vlSelfRef.d3_vgg_like_top__DOT__conv3__DOT__engine__DOT__state))) {
                vlSelfRef.d3_vgg_like_top__DOT__conv3__DOT__engine_done = 1U;
                vlSelfRef.__Vdly__d3_vgg_like_top__DOT__conv3__DOT__engine__DOT__state = 0U;
            } else {
                if ((1U & (~ (IData)(vlSelfRef.d3_vgg_like_top__DOT__conv3__DOT__engine__DOT__last_item)))) {
                    if ((2U != (IData)(vlSelfRef.d3_vgg_like_top__DOT__conv3__DOT__engine__DOT__kc))) {
                        vlSelfRef.d3_vgg_like_top__DOT__conv3__DOT__engine__DOT__kc 
                            = (3U & ((IData)(1U) + (IData)(vlSelfRef.d3_vgg_like_top__DOT__conv3__DOT__engine__DOT__kc)));
                    } else {
                        if ((2U != (IData)(vlSelfRef.d3_vgg_like_top__DOT__conv3__DOT__engine__DOT__kr))) {
                            vlSelfRef.d3_vgg_like_top__DOT__conv3__DOT__engine__DOT__kr 
                                = (3U & ((IData)(1U) 
                                         + (IData)(vlSelfRef.d3_vgg_like_top__DOT__conv3__DOT__engine__DOT__kr)));
                        } else {
                            vlSelfRef.__Vdly__d3_vgg_like_top__DOT__conv3__DOT__engine__DOT__ic 
                                = (0x0000ffffU & ((IData)(1U) 
                                                  + (IData)(vlSelfRef.d3_vgg_like_top__DOT__conv3__DOT__engine__DOT__ic)));
                            vlSelfRef.d3_vgg_like_top__DOT__conv3__DOT__engine__DOT__kr = 0U;
                        }
                        vlSelfRef.d3_vgg_like_top__DOT__conv3__DOT__engine__DOT__kc = 0U;
                    }
                }
                vlSelfRef.__Vdly__d3_vgg_like_top__DOT__conv3__DOT__engine__DOT__product_reg 
                    = (0x0000ffffU & (VL_MULS_III(16, 
                                                  (0x0000ffffU 
                                                   & VL_EXTENDS_II(16,8, (IData)(vlSelfRef.d3_vgg_like_top__DOT__b_rd))), 
                                                  (0x0000ffffU 
                                                   & VL_EXTENDS_II(16,8, (IData)(vlSelfRef.d3_vgg_like_top__DOT__c3_wd)))) 
                                      & (- (IData)((IData)(vlSelfRef.d3_vgg_like_top__DOT__conv3__DOT__engine__DOT__valid_pixel)))));
                vlSelfRef.__Vdly__d3_vgg_like_top__DOT__conv3__DOT__engine__DOT__product_valid = 1U;
                vlSelfRef.__Vdly__d3_vgg_like_top__DOT__conv3__DOT__engine__DOT__product_last 
                    = vlSelfRef.d3_vgg_like_top__DOT__conv3__DOT__engine__DOT__last_item;
                vlSelfRef.__Vdly__d3_vgg_like_top__DOT__conv3__DOT__engine__DOT__state = 1U;
            }
        } else if ((1U & (IData)(vlSelfRef.d3_vgg_like_top__DOT__conv3__DOT__engine__DOT__state))) {
            if (vlSelfRef.d3_vgg_like_top__DOT__conv3__DOT__engine__DOT__product_valid) {
                vlSelfRef.d3_vgg_like_top__DOT__conv3__DOT__raw_sum 
                    = (vlSelfRef.d3_vgg_like_top__DOT__conv3__DOT__raw_sum 
                       + (((- (IData)((1U & ((IData)(vlSelfRef.d3_vgg_like_top__DOT__conv3__DOT__engine__DOT__product_reg) 
                                             >> 0x0fU)))) 
                           << 0x00000010U) | (IData)(vlSelfRef.d3_vgg_like_top__DOT__conv3__DOT__engine__DOT__product_reg)));
                if (vlSelfRef.d3_vgg_like_top__DOT__conv3__DOT__engine__DOT__product_last) {
                    vlSelfRef.__Vdly__d3_vgg_like_top__DOT__conv3__DOT__engine__DOT__product_valid = 0U;
                    vlSelfRef.__Vdly__d3_vgg_like_top__DOT__conv3__DOT__engine__DOT__state = 3U;
                } else {
                    vlSelfRef.__Vdly__d3_vgg_like_top__DOT__conv3__DOT__engine__DOT__state = 2U;
                }
            } else {
                vlSelfRef.__Vdly__d3_vgg_like_top__DOT__conv3__DOT__engine__DOT__state = 2U;
            }
        } else if (vlSelfRef.d3_vgg_like_top__DOT__conv3__DOT__engine_start) {
            vlSelfRef.__Vdly__d3_vgg_like_top__DOT__conv3__DOT__engine__DOT__ic = 0U;
            vlSelfRef.d3_vgg_like_top__DOT__conv3__DOT__engine__DOT__kr = 0U;
            vlSelfRef.d3_vgg_like_top__DOT__conv3__DOT__engine__DOT__kc = 0U;
            vlSelfRef.d3_vgg_like_top__DOT__conv3__DOT__raw_sum = 0U;
            vlSelfRef.__Vdly__d3_vgg_like_top__DOT__conv3__DOT__engine__DOT__product_valid = 0U;
            vlSelfRef.__Vdly__d3_vgg_like_top__DOT__conv3__DOT__engine__DOT__product_last = 0U;
            vlSelfRef.__Vdly__d3_vgg_like_top__DOT__conv3__DOT__engine__DOT__state = 1U;
        }
        vlSelfRef.d3_vgg_like_top__DOT__conv1__DOT__engine_done = 0U;
        if ((2U & (IData)(vlSelfRef.d3_vgg_like_top__DOT__conv1__DOT__engine__DOT__state))) {
            if ((1U & (IData)(vlSelfRef.d3_vgg_like_top__DOT__conv1__DOT__engine__DOT__state))) {
                vlSelfRef.d3_vgg_like_top__DOT__conv1__DOT__engine_done = 1U;
                vlSelfRef.__Vdly__d3_vgg_like_top__DOT__conv1__DOT__engine__DOT__state = 0U;
            } else {
                if ((1U & (~ (IData)(vlSelfRef.d3_vgg_like_top__DOT__conv1__DOT__engine__DOT__last_item)))) {
                    if ((2U != (IData)(vlSelfRef.d3_vgg_like_top__DOT__conv1__DOT__engine__DOT__kc))) {
                        vlSelfRef.d3_vgg_like_top__DOT__conv1__DOT__engine__DOT__kc 
                            = (3U & ((IData)(1U) + (IData)(vlSelfRef.d3_vgg_like_top__DOT__conv1__DOT__engine__DOT__kc)));
                    } else {
                        if ((2U != (IData)(vlSelfRef.d3_vgg_like_top__DOT__conv1__DOT__engine__DOT__kr))) {
                            vlSelfRef.d3_vgg_like_top__DOT__conv1__DOT__engine__DOT__kr 
                                = (3U & ((IData)(1U) 
                                         + (IData)(vlSelfRef.d3_vgg_like_top__DOT__conv1__DOT__engine__DOT__kr)));
                        } else {
                            vlSelfRef.__Vdly__d3_vgg_like_top__DOT__conv1__DOT__engine__DOT__ic 
                                = (0x0000ffffU & ((IData)(1U) 
                                                  + (IData)(vlSelfRef.d3_vgg_like_top__DOT__conv1__DOT__engine__DOT__ic)));
                            vlSelfRef.d3_vgg_like_top__DOT__conv1__DOT__engine__DOT__kr = 0U;
                        }
                        vlSelfRef.d3_vgg_like_top__DOT__conv1__DOT__engine__DOT__kc = 0U;
                    }
                }
                vlSelfRef.__Vdly__d3_vgg_like_top__DOT__conv1__DOT__engine__DOT__product_reg 
                    = (0x0000ffffU & (VL_MULS_III(16, 
                                                  (0x0000ffffU 
                                                   & VL_EXTENDS_II(16,8, (IData)(vlSelfRef.d3_vgg_like_top__DOT__a_rd))), 
                                                  (0x0000ffffU 
                                                   & VL_EXTENDS_II(16,8, (IData)(vlSelfRef.d3_vgg_like_top__DOT__c1_wd)))) 
                                      & (- (IData)((IData)(vlSelfRef.d3_vgg_like_top__DOT__conv1__DOT__engine__DOT__valid_pixel)))));
                vlSelfRef.__Vdly__d3_vgg_like_top__DOT__conv1__DOT__engine__DOT__product_valid = 1U;
                vlSelfRef.__Vdly__d3_vgg_like_top__DOT__conv1__DOT__engine__DOT__product_last 
                    = vlSelfRef.d3_vgg_like_top__DOT__conv1__DOT__engine__DOT__last_item;
                vlSelfRef.__Vdly__d3_vgg_like_top__DOT__conv1__DOT__engine__DOT__state = 1U;
            }
        } else if ((1U & (IData)(vlSelfRef.d3_vgg_like_top__DOT__conv1__DOT__engine__DOT__state))) {
            if (vlSelfRef.d3_vgg_like_top__DOT__conv1__DOT__engine__DOT__product_valid) {
                vlSelfRef.d3_vgg_like_top__DOT__conv1__DOT__raw_sum 
                    = (vlSelfRef.d3_vgg_like_top__DOT__conv1__DOT__raw_sum 
                       + (((- (IData)((1U & ((IData)(vlSelfRef.d3_vgg_like_top__DOT__conv1__DOT__engine__DOT__product_reg) 
                                             >> 0x0fU)))) 
                           << 0x00000010U) | (IData)(vlSelfRef.d3_vgg_like_top__DOT__conv1__DOT__engine__DOT__product_reg)));
                if (vlSelfRef.d3_vgg_like_top__DOT__conv1__DOT__engine__DOT__product_last) {
                    vlSelfRef.__Vdly__d3_vgg_like_top__DOT__conv1__DOT__engine__DOT__product_valid = 0U;
                    vlSelfRef.__Vdly__d3_vgg_like_top__DOT__conv1__DOT__engine__DOT__state = 3U;
                } else {
                    vlSelfRef.__Vdly__d3_vgg_like_top__DOT__conv1__DOT__engine__DOT__state = 2U;
                }
            } else {
                vlSelfRef.__Vdly__d3_vgg_like_top__DOT__conv1__DOT__engine__DOT__state = 2U;
            }
        } else if (vlSelfRef.d3_vgg_like_top__DOT__conv1__DOT__engine_start) {
            vlSelfRef.__Vdly__d3_vgg_like_top__DOT__conv1__DOT__engine__DOT__ic = 0U;
            vlSelfRef.d3_vgg_like_top__DOT__conv1__DOT__engine__DOT__kr = 0U;
            vlSelfRef.d3_vgg_like_top__DOT__conv1__DOT__engine__DOT__kc = 0U;
            vlSelfRef.d3_vgg_like_top__DOT__conv1__DOT__raw_sum = 0U;
            vlSelfRef.__Vdly__d3_vgg_like_top__DOT__conv1__DOT__engine__DOT__product_valid = 0U;
            vlSelfRef.__Vdly__d3_vgg_like_top__DOT__conv1__DOT__engine__DOT__product_last = 0U;
            vlSelfRef.__Vdly__d3_vgg_like_top__DOT__conv1__DOT__engine__DOT__state = 1U;
        }
        vlSelfRef.d3_vgg_like_top__DOT__conv4__DOT__engine_done = 0U;
        if ((2U & (IData)(vlSelfRef.d3_vgg_like_top__DOT__conv4__DOT__engine__DOT__state))) {
            if ((1U & (IData)(vlSelfRef.d3_vgg_like_top__DOT__conv4__DOT__engine__DOT__state))) {
                vlSelfRef.d3_vgg_like_top__DOT__conv4__DOT__engine_done = 1U;
                vlSelfRef.__Vdly__d3_vgg_like_top__DOT__conv4__DOT__engine__DOT__state = 0U;
            } else {
                if ((1U & (~ (IData)(vlSelfRef.d3_vgg_like_top__DOT__conv4__DOT__engine__DOT__last_item)))) {
                    if ((2U != (IData)(vlSelfRef.d3_vgg_like_top__DOT__conv4__DOT__engine__DOT__kc))) {
                        vlSelfRef.d3_vgg_like_top__DOT__conv4__DOT__engine__DOT__kc 
                            = (3U & ((IData)(1U) + (IData)(vlSelfRef.d3_vgg_like_top__DOT__conv4__DOT__engine__DOT__kc)));
                    } else {
                        if ((2U != (IData)(vlSelfRef.d3_vgg_like_top__DOT__conv4__DOT__engine__DOT__kr))) {
                            vlSelfRef.d3_vgg_like_top__DOT__conv4__DOT__engine__DOT__kr 
                                = (3U & ((IData)(1U) 
                                         + (IData)(vlSelfRef.d3_vgg_like_top__DOT__conv4__DOT__engine__DOT__kr)));
                        } else {
                            vlSelfRef.__Vdly__d3_vgg_like_top__DOT__conv4__DOT__engine__DOT__ic 
                                = (0x0000ffffU & ((IData)(1U) 
                                                  + (IData)(vlSelfRef.d3_vgg_like_top__DOT__conv4__DOT__engine__DOT__ic)));
                            vlSelfRef.d3_vgg_like_top__DOT__conv4__DOT__engine__DOT__kr = 0U;
                        }
                        vlSelfRef.d3_vgg_like_top__DOT__conv4__DOT__engine__DOT__kc = 0U;
                    }
                }
                vlSelfRef.__Vdly__d3_vgg_like_top__DOT__conv4__DOT__engine__DOT__product_reg 
                    = (0x0000ffffU & (VL_MULS_III(16, 
                                                  (0x0000ffffU 
                                                   & VL_EXTENDS_II(16,8, (IData)(vlSelfRef.d3_vgg_like_top__DOT__a_rd))), 
                                                  (0x0000ffffU 
                                                   & VL_EXTENDS_II(16,8, (IData)(vlSelfRef.d3_vgg_like_top__DOT__c4_wd)))) 
                                      & (- (IData)((IData)(vlSelfRef.d3_vgg_like_top__DOT__conv4__DOT__engine__DOT__valid_pixel)))));
                vlSelfRef.__Vdly__d3_vgg_like_top__DOT__conv4__DOT__engine__DOT__product_valid = 1U;
                vlSelfRef.__Vdly__d3_vgg_like_top__DOT__conv4__DOT__engine__DOT__product_last 
                    = vlSelfRef.d3_vgg_like_top__DOT__conv4__DOT__engine__DOT__last_item;
                vlSelfRef.__Vdly__d3_vgg_like_top__DOT__conv4__DOT__engine__DOT__state = 1U;
            }
        } else if ((1U & (IData)(vlSelfRef.d3_vgg_like_top__DOT__conv4__DOT__engine__DOT__state))) {
            if (vlSelfRef.d3_vgg_like_top__DOT__conv4__DOT__engine__DOT__product_valid) {
                vlSelfRef.d3_vgg_like_top__DOT__conv4__DOT__raw_sum 
                    = (vlSelfRef.d3_vgg_like_top__DOT__conv4__DOT__raw_sum 
                       + (((- (IData)((1U & ((IData)(vlSelfRef.d3_vgg_like_top__DOT__conv4__DOT__engine__DOT__product_reg) 
                                             >> 0x0fU)))) 
                           << 0x00000010U) | (IData)(vlSelfRef.d3_vgg_like_top__DOT__conv4__DOT__engine__DOT__product_reg)));
                if (vlSelfRef.d3_vgg_like_top__DOT__conv4__DOT__engine__DOT__product_last) {
                    vlSelfRef.__Vdly__d3_vgg_like_top__DOT__conv4__DOT__engine__DOT__product_valid = 0U;
                    vlSelfRef.__Vdly__d3_vgg_like_top__DOT__conv4__DOT__engine__DOT__state = 3U;
                } else {
                    vlSelfRef.__Vdly__d3_vgg_like_top__DOT__conv4__DOT__engine__DOT__state = 2U;
                }
            } else {
                vlSelfRef.__Vdly__d3_vgg_like_top__DOT__conv4__DOT__engine__DOT__state = 2U;
            }
        } else if (vlSelfRef.d3_vgg_like_top__DOT__conv4__DOT__engine_start) {
            vlSelfRef.__Vdly__d3_vgg_like_top__DOT__conv4__DOT__engine__DOT__ic = 0U;
            vlSelfRef.d3_vgg_like_top__DOT__conv4__DOT__engine__DOT__kr = 0U;
            vlSelfRef.d3_vgg_like_top__DOT__conv4__DOT__engine__DOT__kc = 0U;
            vlSelfRef.d3_vgg_like_top__DOT__conv4__DOT__raw_sum = 0U;
            vlSelfRef.__Vdly__d3_vgg_like_top__DOT__conv4__DOT__engine__DOT__product_valid = 0U;
            vlSelfRef.__Vdly__d3_vgg_like_top__DOT__conv4__DOT__engine__DOT__product_last = 0U;
            vlSelfRef.__Vdly__d3_vgg_like_top__DOT__conv4__DOT__engine__DOT__state = 1U;
        }
        vlSelfRef.d3_vgg_like_top__DOT__conv5__DOT__engine_done = 0U;
        if ((2U & (IData)(vlSelfRef.d3_vgg_like_top__DOT__conv5__DOT__engine__DOT__state))) {
            if ((1U & (IData)(vlSelfRef.d3_vgg_like_top__DOT__conv5__DOT__engine__DOT__state))) {
                vlSelfRef.d3_vgg_like_top__DOT__conv5__DOT__engine_done = 1U;
                vlSelfRef.__Vdly__d3_vgg_like_top__DOT__conv5__DOT__engine__DOT__state = 0U;
            } else {
                if ((1U & (~ (IData)(vlSelfRef.d3_vgg_like_top__DOT__conv5__DOT__engine__DOT__last_item)))) {
                    if ((2U != (IData)(vlSelfRef.d3_vgg_like_top__DOT__conv5__DOT__engine__DOT__kc))) {
                        vlSelfRef.d3_vgg_like_top__DOT__conv5__DOT__engine__DOT__kc 
                            = (3U & ((IData)(1U) + (IData)(vlSelfRef.d3_vgg_like_top__DOT__conv5__DOT__engine__DOT__kc)));
                    } else {
                        if ((2U != (IData)(vlSelfRef.d3_vgg_like_top__DOT__conv5__DOT__engine__DOT__kr))) {
                            vlSelfRef.d3_vgg_like_top__DOT__conv5__DOT__engine__DOT__kr 
                                = (3U & ((IData)(1U) 
                                         + (IData)(vlSelfRef.d3_vgg_like_top__DOT__conv5__DOT__engine__DOT__kr)));
                        } else {
                            vlSelfRef.__Vdly__d3_vgg_like_top__DOT__conv5__DOT__engine__DOT__ic 
                                = (0x0000ffffU & ((IData)(1U) 
                                                  + (IData)(vlSelfRef.d3_vgg_like_top__DOT__conv5__DOT__engine__DOT__ic)));
                            vlSelfRef.d3_vgg_like_top__DOT__conv5__DOT__engine__DOT__kr = 0U;
                        }
                        vlSelfRef.d3_vgg_like_top__DOT__conv5__DOT__engine__DOT__kc = 0U;
                    }
                }
                vlSelfRef.__Vdly__d3_vgg_like_top__DOT__conv5__DOT__engine__DOT__product_reg 
                    = (0x0000ffffU & (VL_MULS_III(16, 
                                                  (0x0000ffffU 
                                                   & VL_EXTENDS_II(16,8, (IData)(vlSelfRef.d3_vgg_like_top__DOT__a_rd))), 
                                                  (0x0000ffffU 
                                                   & VL_EXTENDS_II(16,8, (IData)(vlSelfRef.d3_vgg_like_top__DOT__c5_wd)))) 
                                      & (- (IData)((IData)(vlSelfRef.d3_vgg_like_top__DOT__conv5__DOT__engine__DOT__valid_pixel)))));
                vlSelfRef.__Vdly__d3_vgg_like_top__DOT__conv5__DOT__engine__DOT__product_valid = 1U;
                vlSelfRef.__Vdly__d3_vgg_like_top__DOT__conv5__DOT__engine__DOT__product_last 
                    = vlSelfRef.d3_vgg_like_top__DOT__conv5__DOT__engine__DOT__last_item;
                vlSelfRef.__Vdly__d3_vgg_like_top__DOT__conv5__DOT__engine__DOT__state = 1U;
            }
        } else if ((1U & (IData)(vlSelfRef.d3_vgg_like_top__DOT__conv5__DOT__engine__DOT__state))) {
            if (vlSelfRef.d3_vgg_like_top__DOT__conv5__DOT__engine__DOT__product_valid) {
                vlSelfRef.d3_vgg_like_top__DOT__conv5__DOT__raw_sum 
                    = (vlSelfRef.d3_vgg_like_top__DOT__conv5__DOT__raw_sum 
                       + (((- (IData)((1U & ((IData)(vlSelfRef.d3_vgg_like_top__DOT__conv5__DOT__engine__DOT__product_reg) 
                                             >> 0x0fU)))) 
                           << 0x00000010U) | (IData)(vlSelfRef.d3_vgg_like_top__DOT__conv5__DOT__engine__DOT__product_reg)));
                if (vlSelfRef.d3_vgg_like_top__DOT__conv5__DOT__engine__DOT__product_last) {
                    vlSelfRef.__Vdly__d3_vgg_like_top__DOT__conv5__DOT__engine__DOT__product_valid = 0U;
                    vlSelfRef.__Vdly__d3_vgg_like_top__DOT__conv5__DOT__engine__DOT__state = 3U;
                } else {
                    vlSelfRef.__Vdly__d3_vgg_like_top__DOT__conv5__DOT__engine__DOT__state = 2U;
                }
            } else {
                vlSelfRef.__Vdly__d3_vgg_like_top__DOT__conv5__DOT__engine__DOT__state = 2U;
            }
        } else if (vlSelfRef.d3_vgg_like_top__DOT__conv5__DOT__engine_start) {
            vlSelfRef.__Vdly__d3_vgg_like_top__DOT__conv5__DOT__engine__DOT__ic = 0U;
            vlSelfRef.d3_vgg_like_top__DOT__conv5__DOT__engine__DOT__kr = 0U;
            vlSelfRef.d3_vgg_like_top__DOT__conv5__DOT__engine__DOT__kc = 0U;
            vlSelfRef.d3_vgg_like_top__DOT__conv5__DOT__raw_sum = 0U;
            vlSelfRef.__Vdly__d3_vgg_like_top__DOT__conv5__DOT__engine__DOT__product_valid = 0U;
            vlSelfRef.__Vdly__d3_vgg_like_top__DOT__conv5__DOT__engine__DOT__product_last = 0U;
            vlSelfRef.__Vdly__d3_vgg_like_top__DOT__conv5__DOT__engine__DOT__state = 1U;
        }
    } else {
        vlSelfRef.d3_vgg_like_top__DOT__conv2__DOT__biased_sum_reg = 0U;
        vlSelfRef.__Vdly__d3_vgg_like_top__DOT__conv2__DOT__controller__DOT__state = 0U;
        vlSelfRef.d3_vgg_like_top__DOT__conv2__DOT__out_row = 0U;
        vlSelfRef.d3_vgg_like_top__DOT__conv2__DOT__out_col = 0U;
        vlSelfRef.d3_vgg_like_top__DOT__conv2__DOT__out_channel = 0U;
        vlSelfRef.__Vdly__d3_vgg_like_top__DOT__conv2__DOT__engine_start = 0U;
        vlSelfRef.d3_vgg_like_top__DOT__conv2__DOT__controller_out_addr = 0U;
        vlSelfRef.d3_vgg_like_top__DOT__conv2__DOT__controller_busy = 0U;
        vlSelfRef.d3_vgg_like_top__DOT__conv2__DOT__controller_done = 0U;
        vlSelfRef.d3_vgg_like_top__DOT__conv3__DOT__biased_sum_reg = 0U;
        vlSelfRef.__Vdly__d3_vgg_like_top__DOT__conv3__DOT__controller__DOT__state = 0U;
        vlSelfRef.d3_vgg_like_top__DOT__conv3__DOT__out_row = 0U;
        vlSelfRef.d3_vgg_like_top__DOT__conv3__DOT__out_col = 0U;
        vlSelfRef.d3_vgg_like_top__DOT__conv3__DOT__out_channel = 0U;
        vlSelfRef.__Vdly__d3_vgg_like_top__DOT__conv3__DOT__engine_start = 0U;
        vlSelfRef.d3_vgg_like_top__DOT__conv3__DOT__controller_out_addr = 0U;
        vlSelfRef.d3_vgg_like_top__DOT__conv3__DOT__controller_busy = 0U;
        vlSelfRef.d3_vgg_like_top__DOT__conv3__DOT__controller_done = 0U;
        vlSelfRef.d3_vgg_like_top__DOT__conv1__DOT__biased_sum_reg = 0U;
        vlSelfRef.__Vdly__d3_vgg_like_top__DOT__conv1__DOT__controller__DOT__state = 0U;
        vlSelfRef.d3_vgg_like_top__DOT__conv1__DOT__out_row = 0U;
        vlSelfRef.d3_vgg_like_top__DOT__conv1__DOT__out_col = 0U;
        vlSelfRef.d3_vgg_like_top__DOT__conv1__DOT__out_channel = 0U;
        vlSelfRef.__Vdly__d3_vgg_like_top__DOT__conv1__DOT__engine_start = 0U;
        vlSelfRef.d3_vgg_like_top__DOT__conv1__DOT__controller_out_addr = 0U;
        vlSelfRef.d3_vgg_like_top__DOT__conv1__DOT__controller_busy = 0U;
        vlSelfRef.d3_vgg_like_top__DOT__conv1__DOT__controller_done = 0U;
        vlSelfRef.d3_vgg_like_top__DOT__conv4__DOT__biased_sum_reg = 0U;
        vlSelfRef.__Vdly__d3_vgg_like_top__DOT__conv4__DOT__controller__DOT__state = 0U;
        vlSelfRef.d3_vgg_like_top__DOT__conv4__DOT__out_row = 0U;
        vlSelfRef.d3_vgg_like_top__DOT__conv4__DOT__out_col = 0U;
        vlSelfRef.d3_vgg_like_top__DOT__conv4__DOT__out_channel = 0U;
        vlSelfRef.__Vdly__d3_vgg_like_top__DOT__conv4__DOT__engine_start = 0U;
        vlSelfRef.d3_vgg_like_top__DOT__conv4__DOT__controller_out_addr = 0U;
        vlSelfRef.d3_vgg_like_top__DOT__conv4__DOT__controller_busy = 0U;
        vlSelfRef.d3_vgg_like_top__DOT__conv4__DOT__controller_done = 0U;
        vlSelfRef.d3_vgg_like_top__DOT__conv5__DOT__biased_sum_reg = 0U;
        vlSelfRef.__Vdly__d3_vgg_like_top__DOT__conv5__DOT__controller__DOT__state = 0U;
        vlSelfRef.d3_vgg_like_top__DOT__conv5__DOT__out_row = 0U;
        vlSelfRef.d3_vgg_like_top__DOT__conv5__DOT__out_col = 0U;
        vlSelfRef.d3_vgg_like_top__DOT__conv5__DOT__out_channel = 0U;
        vlSelfRef.__Vdly__d3_vgg_like_top__DOT__conv5__DOT__engine_start = 0U;
        vlSelfRef.d3_vgg_like_top__DOT__conv5__DOT__controller_out_addr = 0U;
        vlSelfRef.d3_vgg_like_top__DOT__conv5__DOT__controller_busy = 0U;
        vlSelfRef.d3_vgg_like_top__DOT__conv5__DOT__controller_done = 0U;
        vlSelfRef.__Vdly__d3_vgg_like_top__DOT__conv2__DOT__engine__DOT__ic = 0U;
        vlSelfRef.d3_vgg_like_top__DOT__conv2__DOT__engine__DOT__kr = 0U;
        vlSelfRef.d3_vgg_like_top__DOT__conv2__DOT__engine__DOT__kc = 0U;
        vlSelfRef.d3_vgg_like_top__DOT__conv2__DOT__raw_sum = 0U;
        vlSelfRef.__Vdly__d3_vgg_like_top__DOT__conv2__DOT__engine__DOT__state = 0U;
        vlSelfRef.__Vdly__d3_vgg_like_top__DOT__conv2__DOT__engine__DOT__product_reg = 0U;
        vlSelfRef.__Vdly__d3_vgg_like_top__DOT__conv2__DOT__engine__DOT__product_valid = 0U;
        vlSelfRef.__Vdly__d3_vgg_like_top__DOT__conv2__DOT__engine__DOT__product_last = 0U;
        vlSelfRef.d3_vgg_like_top__DOT__conv2__DOT__engine_done = 0U;
        vlSelfRef.__Vdly__d3_vgg_like_top__DOT__conv3__DOT__engine__DOT__ic = 0U;
        vlSelfRef.d3_vgg_like_top__DOT__conv3__DOT__engine__DOT__kr = 0U;
        vlSelfRef.d3_vgg_like_top__DOT__conv3__DOT__engine__DOT__kc = 0U;
        vlSelfRef.d3_vgg_like_top__DOT__conv3__DOT__raw_sum = 0U;
        vlSelfRef.__Vdly__d3_vgg_like_top__DOT__conv3__DOT__engine__DOT__state = 0U;
        vlSelfRef.__Vdly__d3_vgg_like_top__DOT__conv3__DOT__engine__DOT__product_reg = 0U;
        vlSelfRef.__Vdly__d3_vgg_like_top__DOT__conv3__DOT__engine__DOT__product_valid = 0U;
        vlSelfRef.__Vdly__d3_vgg_like_top__DOT__conv3__DOT__engine__DOT__product_last = 0U;
        vlSelfRef.d3_vgg_like_top__DOT__conv3__DOT__engine_done = 0U;
        vlSelfRef.__Vdly__d3_vgg_like_top__DOT__conv1__DOT__engine__DOT__ic = 0U;
        vlSelfRef.d3_vgg_like_top__DOT__conv1__DOT__engine__DOT__kr = 0U;
        vlSelfRef.d3_vgg_like_top__DOT__conv1__DOT__engine__DOT__kc = 0U;
        vlSelfRef.d3_vgg_like_top__DOT__conv1__DOT__raw_sum = 0U;
        vlSelfRef.__Vdly__d3_vgg_like_top__DOT__conv1__DOT__engine__DOT__state = 0U;
        vlSelfRef.__Vdly__d3_vgg_like_top__DOT__conv1__DOT__engine__DOT__product_reg = 0U;
        vlSelfRef.__Vdly__d3_vgg_like_top__DOT__conv1__DOT__engine__DOT__product_valid = 0U;
        vlSelfRef.__Vdly__d3_vgg_like_top__DOT__conv1__DOT__engine__DOT__product_last = 0U;
        vlSelfRef.d3_vgg_like_top__DOT__conv1__DOT__engine_done = 0U;
        vlSelfRef.__Vdly__d3_vgg_like_top__DOT__conv4__DOT__engine__DOT__ic = 0U;
        vlSelfRef.d3_vgg_like_top__DOT__conv4__DOT__engine__DOT__kr = 0U;
        vlSelfRef.d3_vgg_like_top__DOT__conv4__DOT__engine__DOT__kc = 0U;
        vlSelfRef.d3_vgg_like_top__DOT__conv4__DOT__raw_sum = 0U;
        vlSelfRef.__Vdly__d3_vgg_like_top__DOT__conv4__DOT__engine__DOT__state = 0U;
        vlSelfRef.__Vdly__d3_vgg_like_top__DOT__conv4__DOT__engine__DOT__product_reg = 0U;
        vlSelfRef.__Vdly__d3_vgg_like_top__DOT__conv4__DOT__engine__DOT__product_valid = 0U;
        vlSelfRef.__Vdly__d3_vgg_like_top__DOT__conv4__DOT__engine__DOT__product_last = 0U;
        vlSelfRef.d3_vgg_like_top__DOT__conv4__DOT__engine_done = 0U;
        vlSelfRef.__Vdly__d3_vgg_like_top__DOT__conv5__DOT__engine__DOT__ic = 0U;
        vlSelfRef.d3_vgg_like_top__DOT__conv5__DOT__engine__DOT__kr = 0U;
        vlSelfRef.d3_vgg_like_top__DOT__conv5__DOT__engine__DOT__kc = 0U;
        vlSelfRef.d3_vgg_like_top__DOT__conv5__DOT__raw_sum = 0U;
        vlSelfRef.__Vdly__d3_vgg_like_top__DOT__conv5__DOT__engine__DOT__state = 0U;
        vlSelfRef.__Vdly__d3_vgg_like_top__DOT__conv5__DOT__engine__DOT__product_reg = 0U;
        vlSelfRef.__Vdly__d3_vgg_like_top__DOT__conv5__DOT__engine__DOT__product_valid = 0U;
        vlSelfRef.__Vdly__d3_vgg_like_top__DOT__conv5__DOT__engine__DOT__product_last = 0U;
        vlSelfRef.d3_vgg_like_top__DOT__conv5__DOT__engine_done = 0U;
    }
    d3_vgg_like_top__DOT__conv2__DOT__quant__DOT__result43 
        = (0x000007ffffffffffULL & ((1U & (IData)((vlSelfRef.d3_vgg_like_top__DOT__conv2__DOT__product_reg 
                                                   >> 0x0000002aU)))
                                     ? (- d3_vgg_like_top__DOT__conv2__DOT__quant__DOT__gen_round_shift__DOT__shifted_magnitude)
                                     : d3_vgg_like_top__DOT__conv2__DOT__quant__DOT__gen_round_shift__DOT__shifted_magnitude));
    vlSelfRef.d3_vgg_like_top__DOT__c2_start = vlSelfRef.__Vdly__d3_vgg_like_top__DOT__c2_start;
    vlSelfRef.d3_vgg_like_top__DOT__conv2__DOT__controller__DOT__state 
        = vlSelfRef.__Vdly__d3_vgg_like_top__DOT__conv2__DOT__controller__DOT__state;
    d3_vgg_like_top__DOT__conv3__DOT__quant__DOT__result43 
        = (0x000007ffffffffffULL & ((1U & (IData)((vlSelfRef.d3_vgg_like_top__DOT__conv3__DOT__product_reg 
                                                   >> 0x0000002aU)))
                                     ? (- d3_vgg_like_top__DOT__conv3__DOT__quant__DOT__gen_round_shift__DOT__shifted_magnitude)
                                     : d3_vgg_like_top__DOT__conv3__DOT__quant__DOT__gen_round_shift__DOT__shifted_magnitude));
    vlSelfRef.d3_vgg_like_top__DOT__c3_start = vlSelfRef.__Vdly__d3_vgg_like_top__DOT__c3_start;
    vlSelfRef.d3_vgg_like_top__DOT__conv3__DOT__controller__DOT__state 
        = vlSelfRef.__Vdly__d3_vgg_like_top__DOT__conv3__DOT__controller__DOT__state;
    d3_vgg_like_top__DOT__conv1__DOT__quant__DOT__result43 
        = (0x000007ffffffffffULL & ((1U & (IData)((vlSelfRef.d3_vgg_like_top__DOT__conv1__DOT__product_reg 
                                                   >> 0x0000002aU)))
                                     ? (- d3_vgg_like_top__DOT__conv1__DOT__quant__DOT__gen_round_shift__DOT__shifted_magnitude)
                                     : d3_vgg_like_top__DOT__conv1__DOT__quant__DOT__gen_round_shift__DOT__shifted_magnitude));
    vlSelfRef.d3_vgg_like_top__DOT__c1_start = vlSelfRef.__Vdly__d3_vgg_like_top__DOT__c1_start;
    vlSelfRef.d3_vgg_like_top__DOT__conv1__DOT__controller__DOT__state 
        = vlSelfRef.__Vdly__d3_vgg_like_top__DOT__conv1__DOT__controller__DOT__state;
    d3_vgg_like_top__DOT__conv4__DOT__quant__DOT__result43 
        = (0x000007ffffffffffULL & ((1U & (IData)((vlSelfRef.d3_vgg_like_top__DOT__conv4__DOT__product_reg 
                                                   >> 0x0000002aU)))
                                     ? (- d3_vgg_like_top__DOT__conv4__DOT__quant__DOT__gen_round_shift__DOT__shifted_magnitude)
                                     : d3_vgg_like_top__DOT__conv4__DOT__quant__DOT__gen_round_shift__DOT__shifted_magnitude));
    vlSelfRef.d3_vgg_like_top__DOT__c4_start = vlSelfRef.__Vdly__d3_vgg_like_top__DOT__c4_start;
    vlSelfRef.d3_vgg_like_top__DOT__conv4__DOT__controller__DOT__state 
        = vlSelfRef.__Vdly__d3_vgg_like_top__DOT__conv4__DOT__controller__DOT__state;
    d3_vgg_like_top__DOT__conv5__DOT__quant__DOT__result43 
        = (0x000007ffffffffffULL & ((1U & (IData)((vlSelfRef.d3_vgg_like_top__DOT__conv5__DOT__product_reg 
                                                   >> 0x0000002aU)))
                                     ? (- d3_vgg_like_top__DOT__conv5__DOT__quant__DOT__gen_round_shift__DOT__shifted_magnitude)
                                     : d3_vgg_like_top__DOT__conv5__DOT__quant__DOT__gen_round_shift__DOT__shifted_magnitude));
    vlSelfRef.d3_vgg_like_top__DOT__c5_start = vlSelfRef.__Vdly__d3_vgg_like_top__DOT__c5_start;
    vlSelfRef.d3_vgg_like_top__DOT__conv5__DOT__controller__DOT__state 
        = vlSelfRef.__Vdly__d3_vgg_like_top__DOT__conv5__DOT__controller__DOT__state;
    vlSelfRef.d3_vgg_like_top__DOT__conv2__DOT__activated 
        = ((IData)(d3_vgg_like_top__DOT__conv2__DOT__quant__DOT__result43) 
           & (- (IData)((1U & (~ (IData)((d3_vgg_like_top__DOT__conv2__DOT__quant__DOT__result43 
                                          >> 0x0000001fU)))))));
    vlSelfRef.d3_vgg_like_top__DOT__conv3__DOT__activated 
        = ((IData)(d3_vgg_like_top__DOT__conv3__DOT__quant__DOT__result43) 
           & (- (IData)((1U & (~ (IData)((d3_vgg_like_top__DOT__conv3__DOT__quant__DOT__result43 
                                          >> 0x0000001fU)))))));
    vlSelfRef.d3_vgg_like_top__DOT__conv1__DOT__activated 
        = ((IData)(d3_vgg_like_top__DOT__conv1__DOT__quant__DOT__result43) 
           & (- (IData)((1U & (~ (IData)((d3_vgg_like_top__DOT__conv1__DOT__quant__DOT__result43 
                                          >> 0x0000001fU)))))));
    vlSelfRef.d3_vgg_like_top__DOT__conv4__DOT__activated 
        = ((IData)(d3_vgg_like_top__DOT__conv4__DOT__quant__DOT__result43) 
           & (- (IData)((1U & (~ (IData)((d3_vgg_like_top__DOT__conv4__DOT__quant__DOT__result43 
                                          >> 0x0000001fU)))))));
    vlSelfRef.d3_vgg_like_top__DOT__conv5__DOT__activated 
        = ((IData)(d3_vgg_like_top__DOT__conv5__DOT__quant__DOT__result43) 
           & (- (IData)((1U & (~ (IData)((d3_vgg_like_top__DOT__conv5__DOT__quant__DOT__result43 
                                          >> 0x0000001fU)))))));
    vlSelfRef.__VdfgRegularize_h6e95ff9d_0_13 = ((IData)(vlSelfRef.d3_vgg_like_top__DOT__conv5__DOT__out_channel) 
                                                 << 6U);
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
                                                                                | ((IData)(vlSelfRef.d3_vgg_like_top__DOT__conv5__DOT__postprocess_pending) 
                                                                                | ((IData)(vlSelfRef.d3_vgg_like_top__DOT__conv5__DOT__product_valid) 
                                                                                | ((IData)(vlSelfRef.d3_vgg_like_top__DOT__conv5__DOT__postprocess_valid) 
                                                                                | ((IData)(vlSelfRef.d3_vgg_like_top__DOT__conv1__DOT__controller_busy) 
                                                                                | ((IData)(vlSelfRef.d3_vgg_like_top__DOT__conv2__DOT__controller_busy) 
                                                                                | ((IData)(vlSelfRef.d3_vgg_like_top__DOT__conv3__DOT__controller_busy) 
                                                                                | ((IData)(vlSelfRef.d3_vgg_like_top__DOT__conv4__DOT__controller_busy) 
                                                                                | ((IData)(vlSelfRef.d3_vgg_like_top__DOT__conv5__DOT__controller_busy) 
                                                                                | ((IData)(vlSelfRef.d3_vgg_like_top__DOT__p1_busy) 
                                                                                | ((IData)(vlSelfRef.d3_vgg_like_top__DOT__p2_busy) 
                                                                                | ((IData)(vlSelfRef.d3_vgg_like_top__DOT__gap_busy) 
                                                                                | (IData)(vlSelfRef.d3_vgg_like_top__DOT__fc_busy))))))))))))))))))))))))) 
                                                   & (IData)(vlSelfRef.input_we)));
    vlSelfRef.d3_vgg_like_top__DOT__conv2__DOT__engine_start 
        = vlSelfRef.__Vdly__d3_vgg_like_top__DOT__conv2__DOT__engine_start;
    vlSelfRef.d3_vgg_like_top__DOT__conv2__DOT__engine__DOT__state 
        = vlSelfRef.__Vdly__d3_vgg_like_top__DOT__conv2__DOT__engine__DOT__state;
    vlSelfRef.d3_vgg_like_top__DOT__conv2__DOT__engine__DOT__product_reg 
        = vlSelfRef.__Vdly__d3_vgg_like_top__DOT__conv2__DOT__engine__DOT__product_reg;
    vlSelfRef.d3_vgg_like_top__DOT__conv2__DOT__engine__DOT__product_valid 
        = vlSelfRef.__Vdly__d3_vgg_like_top__DOT__conv2__DOT__engine__DOT__product_valid;
    vlSelfRef.d3_vgg_like_top__DOT__conv2__DOT__engine__DOT__product_last 
        = vlSelfRef.__Vdly__d3_vgg_like_top__DOT__conv2__DOT__engine__DOT__product_last;
    vlSelfRef.d3_vgg_like_top__DOT__conv3__DOT__engine_start 
        = vlSelfRef.__Vdly__d3_vgg_like_top__DOT__conv3__DOT__engine_start;
    vlSelfRef.d3_vgg_like_top__DOT__conv3__DOT__engine__DOT__state 
        = vlSelfRef.__Vdly__d3_vgg_like_top__DOT__conv3__DOT__engine__DOT__state;
    vlSelfRef.d3_vgg_like_top__DOT__conv3__DOT__engine__DOT__product_reg 
        = vlSelfRef.__Vdly__d3_vgg_like_top__DOT__conv3__DOT__engine__DOT__product_reg;
    vlSelfRef.d3_vgg_like_top__DOT__conv3__DOT__engine__DOT__product_valid 
        = vlSelfRef.__Vdly__d3_vgg_like_top__DOT__conv3__DOT__engine__DOT__product_valid;
    vlSelfRef.d3_vgg_like_top__DOT__conv3__DOT__engine__DOT__product_last 
        = vlSelfRef.__Vdly__d3_vgg_like_top__DOT__conv3__DOT__engine__DOT__product_last;
    vlSelfRef.d3_vgg_like_top__DOT__conv1__DOT__engine_start 
        = vlSelfRef.__Vdly__d3_vgg_like_top__DOT__conv1__DOT__engine_start;
    vlSelfRef.d3_vgg_like_top__DOT__conv1__DOT__engine__DOT__state 
        = vlSelfRef.__Vdly__d3_vgg_like_top__DOT__conv1__DOT__engine__DOT__state;
    vlSelfRef.d3_vgg_like_top__DOT__conv1__DOT__engine__DOT__product_reg 
        = vlSelfRef.__Vdly__d3_vgg_like_top__DOT__conv1__DOT__engine__DOT__product_reg;
    vlSelfRef.d3_vgg_like_top__DOT__conv1__DOT__engine__DOT__product_valid 
        = vlSelfRef.__Vdly__d3_vgg_like_top__DOT__conv1__DOT__engine__DOT__product_valid;
    vlSelfRef.d3_vgg_like_top__DOT__conv1__DOT__engine__DOT__product_last 
        = vlSelfRef.__Vdly__d3_vgg_like_top__DOT__conv1__DOT__engine__DOT__product_last;
    vlSelfRef.d3_vgg_like_top__DOT__conv4__DOT__engine_start 
        = vlSelfRef.__Vdly__d3_vgg_like_top__DOT__conv4__DOT__engine_start;
    vlSelfRef.d3_vgg_like_top__DOT__conv4__DOT__engine__DOT__state 
        = vlSelfRef.__Vdly__d3_vgg_like_top__DOT__conv4__DOT__engine__DOT__state;
    vlSelfRef.d3_vgg_like_top__DOT__conv4__DOT__engine__DOT__product_reg 
        = vlSelfRef.__Vdly__d3_vgg_like_top__DOT__conv4__DOT__engine__DOT__product_reg;
    vlSelfRef.d3_vgg_like_top__DOT__conv4__DOT__engine__DOT__product_valid 
        = vlSelfRef.__Vdly__d3_vgg_like_top__DOT__conv4__DOT__engine__DOT__product_valid;
    vlSelfRef.d3_vgg_like_top__DOT__conv4__DOT__engine__DOT__product_last 
        = vlSelfRef.__Vdly__d3_vgg_like_top__DOT__conv4__DOT__engine__DOT__product_last;
    vlSelfRef.d3_vgg_like_top__DOT__conv5__DOT__engine_start 
        = vlSelfRef.__Vdly__d3_vgg_like_top__DOT__conv5__DOT__engine_start;
    vlSelfRef.d3_vgg_like_top__DOT__conv5__DOT__engine__DOT__state 
        = vlSelfRef.__Vdly__d3_vgg_like_top__DOT__conv5__DOT__engine__DOT__state;
    vlSelfRef.d3_vgg_like_top__DOT__conv5__DOT__engine__DOT__product_reg 
        = vlSelfRef.__Vdly__d3_vgg_like_top__DOT__conv5__DOT__engine__DOT__product_reg;
    vlSelfRef.d3_vgg_like_top__DOT__conv5__DOT__engine__DOT__product_valid 
        = vlSelfRef.__Vdly__d3_vgg_like_top__DOT__conv5__DOT__engine__DOT__product_valid;
    vlSelfRef.d3_vgg_like_top__DOT__conv5__DOT__engine__DOT__product_last 
        = vlSelfRef.__Vdly__d3_vgg_like_top__DOT__conv5__DOT__engine__DOT__product_last;
}

void Vd3_vgg_like_top___024root___nba_sequent__TOP__3(Vd3_vgg_like_top___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vd3_vgg_like_top___024root___nba_sequent__TOP__3\n"); );
    Vd3_vgg_like_top__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    vlSelfRef.d3_vgg_like_top__DOT__c2_wd = ((0x23ffU 
                                              >= (IData)(vlSelfRef.d3_vgg_like_top__DOT__rw2__DOT__rom_addr))
                                              ? vlSelfRef.d3_vgg_like_top__DOT__rw2__DOT__rom
                                             [vlSelfRef.d3_vgg_like_top__DOT__rw2__DOT__rom_addr]
                                              : 0U);
    vlSelfRef.d3_vgg_like_top__DOT__c3_wd = ((0x47ffU 
                                              >= (IData)(vlSelfRef.d3_vgg_like_top__DOT__rw3__DOT__rom_addr))
                                              ? vlSelfRef.d3_vgg_like_top__DOT__rw3__DOT__rom
                                             [vlSelfRef.d3_vgg_like_top__DOT__rw3__DOT__rom_addr]
                                              : 0U);
    vlSelfRef.d3_vgg_like_top__DOT__b_rd = vlSelfRef.d3_vgg_like_top__DOT__buffer_b__DOT__mem
        [(0x00007fffU & ((2U == (IData)(vlSelfRef.state_dbg))
                          ? (0x0001ffffU & ((vlSelfRef.__VdfgRegularize_h6e95ff9d_0_5 
                                             + (VL_SHIFTL_III(15,32,32, (IData)(vlSelfRef.d3_vgg_like_top__DOT__conv2__DOT__engine__DOT__ic), 0x0000000aU) 
                                                + VL_SHIFTL_III(15,32,32, vlSelfRef.__VdfgRegularize_h6e95ff9d_0_4, 5U))) 
                                            & (- (IData)((IData)(vlSelfRef.d3_vgg_like_top__DOT__conv2__DOT__engine__DOT__valid_pixel)))))
                          : ((3U == (IData)(vlSelfRef.state_dbg))
                              ? (IData)(vlSelfRef.__VdfgRegularize_h6e95ff9d_0_14)
                              : ((4U == (IData)(vlSelfRef.state_dbg))
                                  ? (0x0001ffffU & 
                                     ((vlSelfRef.__VdfgRegularize_h6e95ff9d_0_7 
                                       + (VL_SHIFTL_III(15,32,32, (IData)(vlSelfRef.d3_vgg_like_top__DOT__conv3__DOT__engine__DOT__ic), 8U) 
                                          + VL_SHIFTL_III(15,32,32, vlSelfRef.__VdfgRegularize_h6e95ff9d_0_6, 4U))) 
                                      & (- (IData)((IData)(vlSelfRef.d3_vgg_like_top__DOT__conv3__DOT__engine__DOT__valid_pixel)))))
                                  : ((6U == (IData)(vlSelfRef.state_dbg))
                                      ? (VL_SHIFTL_III(15,32,32, (IData)(vlSelfRef.d3_vgg_like_top__DOT__pool2__DOT__ch), 8U) 
                                         + ((0x0000ffffU 
                                             & (VL_SHIFTL_III(16,16,32, (IData)(vlSelfRef.d3_vgg_like_top__DOT__pool2__DOT__col), 1U) 
                                                + (1U 
                                                   & (IData)(vlSelfRef.d3_vgg_like_top__DOT__pool2__DOT__tap)))) 
                                            + (0x000ffff0U 
                                               & ((VL_SHIFTL_III(16,16,32, (IData)(vlSelfRef.d3_vgg_like_top__DOT__pool2__DOT__row), 1U) 
                                                   + 
                                                   (1U 
                                                    & ((IData)(vlSelfRef.d3_vgg_like_top__DOT__pool2__DOT__tap) 
                                                       >> 1U))) 
                                                  << 4U))))
                                      : (0x0001ffffU 
                                         & ((- (IData)(
                                                       (8U 
                                                        == (IData)(vlSelfRef.state_dbg)))) 
                                            & (VL_SHIFTL_III(17,32,32, (IData)(vlSelfRef.d3_vgg_like_top__DOT__gap__DOT__ch), 6U) 
                                               + (IData)(vlSelfRef.d3_vgg_like_top__DOT__gap__DOT__idx)))))))))];
    vlSelfRef.d3_vgg_like_top__DOT__c1_wd = ((0x035fU 
                                              >= (IData)(vlSelfRef.d3_vgg_like_top__DOT__rw1__DOT__rom_addr))
                                              ? vlSelfRef.d3_vgg_like_top__DOT__rw1__DOT__rom
                                             [vlSelfRef.d3_vgg_like_top__DOT__rw1__DOT__rom_addr]
                                              : 0U);
    vlSelfRef.d3_vgg_like_top__DOT__c4_wd = ((0x00009000U 
                                              > vlSelfRef.d3_vgg_like_top__DOT__c4_wa)
                                              ? ((0x8fffU 
                                                  >= 
                                                  (0x0000ffffU 
                                                   & vlSelfRef.d3_vgg_like_top__DOT__c4_wa))
                                                  ? vlSelfRef.d3_vgg_like_top__DOT__rw4__DOT__rom
                                                 [(0x0000ffffU 
                                                   & vlSelfRef.d3_vgg_like_top__DOT__c4_wa)]
                                                  : 0U)
                                              : 0U);
    vlSelfRef.d3_vgg_like_top__DOT__c5_wd = ((0x00011fffU 
                                              >= vlSelfRef.d3_vgg_like_top__DOT__c5_wa)
                                              ? vlSelfRef.d3_vgg_like_top__DOT__rw5__DOT__rom
                                             [vlSelfRef.d3_vgg_like_top__DOT__c5_wa]
                                              : 0U);
    vlSelfRef.d3_vgg_like_top__DOT__a_rd = vlSelfRef.d3_vgg_like_top__DOT__buffer_a__DOT__mem
        [(0x00007fffU & ((1U == (IData)(vlSelfRef.state_dbg))
                          ? (0x0001ffffU & ((vlSelfRef.__VdfgRegularize_h6e95ff9d_0_3 
                                             + (VL_SHIFTL_III(15,32,32, (IData)(vlSelfRef.d3_vgg_like_top__DOT__conv1__DOT__engine__DOT__ic), 0x0000000aU) 
                                                + VL_SHIFTL_III(15,32,32, vlSelfRef.__VdfgRegularize_h6e95ff9d_0_2, 5U))) 
                                            & (- (IData)((IData)(vlSelfRef.d3_vgg_like_top__DOT__conv1__DOT__engine__DOT__valid_pixel)))))
                          : ((3U == (IData)(vlSelfRef.state_dbg))
                              ? (IData)(vlSelfRef.__VdfgRegularize_h6e95ff9d_0_14)
                              : ((5U == (IData)(vlSelfRef.state_dbg))
                                  ? (0x0001ffffU & 
                                     ((vlSelfRef.__VdfgRegularize_h6e95ff9d_0_9 
                                       + (VL_SHIFTL_III(15,32,32, (IData)(vlSelfRef.d3_vgg_like_top__DOT__conv4__DOT__engine__DOT__ic), 8U) 
                                          + VL_SHIFTL_III(15,32,32, vlSelfRef.__VdfgRegularize_h6e95ff9d_0_8, 4U))) 
                                      & (- (IData)((IData)(vlSelfRef.d3_vgg_like_top__DOT__conv4__DOT__engine__DOT__valid_pixel)))))
                                  : ((7U == (IData)(vlSelfRef.state_dbg))
                                      ? (0x0001ffffU 
                                         & ((vlSelfRef.__VdfgRegularize_h6e95ff9d_0_11 
                                             + (VL_SHIFTL_III(15,32,32, (IData)(vlSelfRef.d3_vgg_like_top__DOT__conv5__DOT__engine__DOT__ic), 6U) 
                                                + VL_SHIFTL_III(15,32,32, vlSelfRef.__VdfgRegularize_h6e95ff9d_0_10, 3U))) 
                                            & (- (IData)((IData)(vlSelfRef.d3_vgg_like_top__DOT__conv5__DOT__engine__DOT__valid_pixel)))))
                                      : ((IData)(vlSelfRef.d3_vgg_like_top__DOT__linear__DOT__feature_idx) 
                                         & (- (IData)(
                                                      (9U 
                                                       == (IData)(vlSelfRef.state_dbg))))))))))];
    if (vlSelfRef.__VdlySet__d3_vgg_like_top__DOT__buffer_b__DOT__mem__v0) {
        vlSelfRef.d3_vgg_like_top__DOT__buffer_b__DOT__mem[vlSelfRef.__VdlyDim0__d3_vgg_like_top__DOT__buffer_b__DOT__mem__v0] 
            = vlSelfRef.__VdlyVal__d3_vgg_like_top__DOT__buffer_b__DOT__mem__v0;
    }
    if (vlSelfRef.__VdlySet__d3_vgg_like_top__DOT__buffer_a__DOT__mem__v0) {
        vlSelfRef.d3_vgg_like_top__DOT__buffer_a__DOT__mem[vlSelfRef.__VdlyDim0__d3_vgg_like_top__DOT__buffer_a__DOT__mem__v0] 
            = vlSelfRef.__VdlyVal__d3_vgg_like_top__DOT__buffer_a__DOT__mem__v0;
    }
}

void Vd3_vgg_like_top___024root___nba_sequent__TOP__4(Vd3_vgg_like_top___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vd3_vgg_like_top___024root___nba_sequent__TOP__4\n"); );
    Vd3_vgg_like_top__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    vlSelfRef.d3_vgg_like_top__DOT__pool2__DOT__ch 
        = vlSelfRef.__Vdly__d3_vgg_like_top__DOT__pool2__DOT__ch;
    vlSelfRef.d3_vgg_like_top__DOT__pool2__DOT__col 
        = vlSelfRef.__Vdly__d3_vgg_like_top__DOT__pool2__DOT__col;
    vlSelfRef.d3_vgg_like_top__DOT__pool2__DOT__tap 
        = vlSelfRef.__Vdly__d3_vgg_like_top__DOT__pool2__DOT__tap;
    vlSelfRef.d3_vgg_like_top__DOT__pool2__DOT__row 
        = vlSelfRef.__Vdly__d3_vgg_like_top__DOT__pool2__DOT__row;
    vlSelfRef.d3_vgg_like_top__DOT__gap__DOT__ch = vlSelfRef.__Vdly__d3_vgg_like_top__DOT__gap__DOT__ch;
    vlSelfRef.d3_vgg_like_top__DOT__gap__DOT__idx = vlSelfRef.__Vdly__d3_vgg_like_top__DOT__gap__DOT__idx;
    vlSelfRef.__VdfgRegularize_h6e95ff9d_0_5 = (((IData)(vlSelfRef.d3_vgg_like_top__DOT__conv2__DOT__engine__DOT__kc) 
                                                 + (IData)(vlSelfRef.d3_vgg_like_top__DOT__conv2__DOT__out_col)) 
                                                - (IData)(1U));
    vlSelfRef.__VdfgRegularize_h6e95ff9d_0_4 = (((IData)(vlSelfRef.d3_vgg_like_top__DOT__conv2__DOT__engine__DOT__kr) 
                                                 + (IData)(vlSelfRef.d3_vgg_like_top__DOT__conv2__DOT__out_row)) 
                                                - (IData)(1U));
    vlSelfRef.__VdfgRegularize_h6e95ff9d_0_7 = (((IData)(vlSelfRef.d3_vgg_like_top__DOT__conv3__DOT__engine__DOT__kc) 
                                                 + (IData)(vlSelfRef.d3_vgg_like_top__DOT__conv3__DOT__out_col)) 
                                                - (IData)(1U));
    vlSelfRef.__VdfgRegularize_h6e95ff9d_0_6 = (((IData)(vlSelfRef.d3_vgg_like_top__DOT__conv3__DOT__engine__DOT__kr) 
                                                 + (IData)(vlSelfRef.d3_vgg_like_top__DOT__conv3__DOT__out_row)) 
                                                - (IData)(1U));
    vlSelfRef.d3_vgg_like_top__DOT__conv2__DOT__engine__DOT__ic 
        = vlSelfRef.__Vdly__d3_vgg_like_top__DOT__conv2__DOT__engine__DOT__ic;
    vlSelfRef.d3_vgg_like_top__DOT__conv3__DOT__engine__DOT__ic 
        = vlSelfRef.__Vdly__d3_vgg_like_top__DOT__conv3__DOT__engine__DOT__ic;
    vlSelfRef.state_dbg = vlSelfRef.__Vdly__state_dbg;
    vlSelfRef.__VdfgRegularize_h6e95ff9d_0_14 = (0x00007fffU 
                                                 & (VL_SHIFTL_III(15,32,32, (IData)(vlSelfRef.d3_vgg_like_top__DOT__pool1__DOT__ch), 0x0000000aU) 
                                                    + 
                                                    ((0x0000ffffU 
                                                      & (VL_SHIFTL_III(16,16,32, (IData)(vlSelfRef.d3_vgg_like_top__DOT__pool1__DOT__col), 1U) 
                                                         + 
                                                         (1U 
                                                          & (IData)(vlSelfRef.d3_vgg_like_top__DOT__pool1__DOT__tap)))) 
                                                     + 
                                                     (0x001fffe0U 
                                                      & ((VL_SHIFTL_III(16,16,32, (IData)(vlSelfRef.d3_vgg_like_top__DOT__pool1__DOT__row), 1U) 
                                                          + 
                                                          (1U 
                                                           & ((IData)(vlSelfRef.d3_vgg_like_top__DOT__pool1__DOT__tap) 
                                                              >> 1U))) 
                                                         << 5U)))));
    vlSelfRef.__VdfgRegularize_h6e95ff9d_0_3 = (((IData)(vlSelfRef.d3_vgg_like_top__DOT__conv1__DOT__engine__DOT__kc) 
                                                 + (IData)(vlSelfRef.d3_vgg_like_top__DOT__conv1__DOT__out_col)) 
                                                - (IData)(1U));
    vlSelfRef.__VdfgRegularize_h6e95ff9d_0_2 = (((IData)(vlSelfRef.d3_vgg_like_top__DOT__conv1__DOT__engine__DOT__kr) 
                                                 + (IData)(vlSelfRef.d3_vgg_like_top__DOT__conv1__DOT__out_row)) 
                                                - (IData)(1U));
    vlSelfRef.__VdfgRegularize_h6e95ff9d_0_9 = (((IData)(vlSelfRef.d3_vgg_like_top__DOT__conv4__DOT__engine__DOT__kc) 
                                                 + (IData)(vlSelfRef.d3_vgg_like_top__DOT__conv4__DOT__out_col)) 
                                                - (IData)(1U));
    vlSelfRef.__VdfgRegularize_h6e95ff9d_0_8 = (((IData)(vlSelfRef.d3_vgg_like_top__DOT__conv4__DOT__engine__DOT__kr) 
                                                 + (IData)(vlSelfRef.d3_vgg_like_top__DOT__conv4__DOT__out_row)) 
                                                - (IData)(1U));
    vlSelfRef.__VdfgRegularize_h6e95ff9d_0_11 = (((IData)(vlSelfRef.d3_vgg_like_top__DOT__conv5__DOT__engine__DOT__kc) 
                                                  + (IData)(vlSelfRef.d3_vgg_like_top__DOT__conv5__DOT__out_col)) 
                                                 - (IData)(1U));
    vlSelfRef.__VdfgRegularize_h6e95ff9d_0_10 = (((IData)(vlSelfRef.d3_vgg_like_top__DOT__conv5__DOT__engine__DOT__kr) 
                                                  + (IData)(vlSelfRef.d3_vgg_like_top__DOT__conv5__DOT__out_row)) 
                                                 - (IData)(1U));
    vlSelfRef.d3_vgg_like_top__DOT__linear__DOT__feature_idx 
        = vlSelfRef.__Vdly__d3_vgg_like_top__DOT__linear__DOT__feature_idx;
    vlSelfRef.d3_vgg_like_top__DOT__conv1__DOT__engine__DOT__ic 
        = vlSelfRef.__Vdly__d3_vgg_like_top__DOT__conv1__DOT__engine__DOT__ic;
    vlSelfRef.d3_vgg_like_top__DOT__conv4__DOT__engine__DOT__ic 
        = vlSelfRef.__Vdly__d3_vgg_like_top__DOT__conv4__DOT__engine__DOT__ic;
    vlSelfRef.d3_vgg_like_top__DOT__conv5__DOT__engine__DOT__ic 
        = vlSelfRef.__Vdly__d3_vgg_like_top__DOT__conv5__DOT__engine__DOT__ic;
    vlSelfRef.d3_vgg_like_top__DOT__conv2__DOT__engine__DOT__valid_pixel 
        = (VL_LTES_III(32, 0U, vlSelfRef.__VdfgRegularize_h6e95ff9d_0_4) 
           & (VL_GTS_III(32, 0x00000020U, vlSelfRef.__VdfgRegularize_h6e95ff9d_0_4) 
              & (VL_LTES_III(32, 0U, vlSelfRef.__VdfgRegularize_h6e95ff9d_0_5) 
                 & VL_GTS_III(32, 0x00000020U, vlSelfRef.__VdfgRegularize_h6e95ff9d_0_5))));
    vlSelfRef.d3_vgg_like_top__DOT__conv3__DOT__engine__DOT__valid_pixel 
        = (VL_LTES_III(32, 0U, vlSelfRef.__VdfgRegularize_h6e95ff9d_0_6) 
           & (VL_GTS_III(32, 0x00000010U, vlSelfRef.__VdfgRegularize_h6e95ff9d_0_6) 
              & (VL_LTES_III(32, 0U, vlSelfRef.__VdfgRegularize_h6e95ff9d_0_7) 
                 & VL_GTS_III(32, 0x00000010U, vlSelfRef.__VdfgRegularize_h6e95ff9d_0_7))));
    vlSelfRef.d3_vgg_like_top__DOT__conv2__DOT__engine__DOT__last_item 
        = ((0x001fU == (IData)(vlSelfRef.d3_vgg_like_top__DOT__conv2__DOT__engine__DOT__ic)) 
           & ((2U == (IData)(vlSelfRef.d3_vgg_like_top__DOT__conv2__DOT__engine__DOT__kr)) 
              & (2U == (IData)(vlSelfRef.d3_vgg_like_top__DOT__conv2__DOT__engine__DOT__kc))));
    vlSelfRef.d3_vgg_like_top__DOT__rw2__DOT__rom_addr 
        = (0x00003fffU & ((IData)(vlSelfRef.d3_vgg_like_top__DOT__conv2__DOT__engine__DOT__kc) 
                          + (((IData)(9U) * VL_SHIFTL_III(32,32,32, (IData)(vlSelfRef.d3_vgg_like_top__DOT__conv2__DOT__out_channel), 5U)) 
                             + (((IData)(9U) * (IData)(vlSelfRef.d3_vgg_like_top__DOT__conv2__DOT__engine__DOT__ic)) 
                                + ((IData)(3U) * (IData)(vlSelfRef.d3_vgg_like_top__DOT__conv2__DOT__engine__DOT__kr))))));
    vlSelfRef.d3_vgg_like_top__DOT__conv3__DOT__engine__DOT__last_item 
        = ((0x001fU == (IData)(vlSelfRef.d3_vgg_like_top__DOT__conv3__DOT__engine__DOT__ic)) 
           & ((2U == (IData)(vlSelfRef.d3_vgg_like_top__DOT__conv3__DOT__engine__DOT__kr)) 
              & (2U == (IData)(vlSelfRef.d3_vgg_like_top__DOT__conv3__DOT__engine__DOT__kc))));
    vlSelfRef.d3_vgg_like_top__DOT__rw3__DOT__rom_addr 
        = (0x00007fffU & ((IData)(vlSelfRef.d3_vgg_like_top__DOT__conv3__DOT__engine__DOT__kc) 
                          + (((IData)(9U) * VL_SHIFTL_III(32,32,32, (IData)(vlSelfRef.d3_vgg_like_top__DOT__conv3__DOT__out_channel), 5U)) 
                             + (((IData)(9U) * (IData)(vlSelfRef.d3_vgg_like_top__DOT__conv3__DOT__engine__DOT__ic)) 
                                + ((IData)(3U) * (IData)(vlSelfRef.d3_vgg_like_top__DOT__conv3__DOT__engine__DOT__kr))))));
    vlSelfRef.d3_vgg_like_top__DOT__conv1__DOT__engine__DOT__valid_pixel 
        = (VL_LTES_III(32, 0U, vlSelfRef.__VdfgRegularize_h6e95ff9d_0_2) 
           & (VL_GTS_III(32, 0x00000020U, vlSelfRef.__VdfgRegularize_h6e95ff9d_0_2) 
              & (VL_LTES_III(32, 0U, vlSelfRef.__VdfgRegularize_h6e95ff9d_0_3) 
                 & VL_GTS_III(32, 0x00000020U, vlSelfRef.__VdfgRegularize_h6e95ff9d_0_3))));
    vlSelfRef.d3_vgg_like_top__DOT__conv4__DOT__engine__DOT__valid_pixel 
        = (VL_LTES_III(32, 0U, vlSelfRef.__VdfgRegularize_h6e95ff9d_0_8) 
           & (VL_GTS_III(32, 0x00000010U, vlSelfRef.__VdfgRegularize_h6e95ff9d_0_8) 
              & (VL_LTES_III(32, 0U, vlSelfRef.__VdfgRegularize_h6e95ff9d_0_9) 
                 & VL_GTS_III(32, 0x00000010U, vlSelfRef.__VdfgRegularize_h6e95ff9d_0_9))));
    vlSelfRef.d3_vgg_like_top__DOT__conv5__DOT__engine__DOT__valid_pixel 
        = (VL_LTES_III(32, 0U, vlSelfRef.__VdfgRegularize_h6e95ff9d_0_10) 
           & (VL_GTS_III(32, 8U, vlSelfRef.__VdfgRegularize_h6e95ff9d_0_10) 
              & (VL_LTES_III(32, 0U, vlSelfRef.__VdfgRegularize_h6e95ff9d_0_11) 
                 & VL_GTS_III(32, 8U, vlSelfRef.__VdfgRegularize_h6e95ff9d_0_11))));
    vlSelfRef.d3_vgg_like_top__DOT__rwf__DOT__rom_addr 
        = (0x000007ffU & ((IData)(vlSelfRef.d3_vgg_like_top__DOT__linear__DOT__feature_idx) 
                          + ((IData)(vlSelfRef.d3_vgg_like_top__DOT__linear__DOT__class_idx) 
                             << 7U)));
    vlSelfRef.d3_vgg_like_top__DOT__conv1__DOT__engine__DOT__last_item 
        = ((2U == (IData)(vlSelfRef.d3_vgg_like_top__DOT__conv1__DOT__engine__DOT__ic)) 
           & ((2U == (IData)(vlSelfRef.d3_vgg_like_top__DOT__conv1__DOT__engine__DOT__kr)) 
              & (2U == (IData)(vlSelfRef.d3_vgg_like_top__DOT__conv1__DOT__engine__DOT__kc))));
    vlSelfRef.d3_vgg_like_top__DOT__rw1__DOT__rom_addr 
        = (0x000003ffU & ((IData)(vlSelfRef.d3_vgg_like_top__DOT__conv1__DOT__engine__DOT__kc) 
                          + (((IData)(0x0000001bU) 
                              * (IData)(vlSelfRef.d3_vgg_like_top__DOT__conv1__DOT__out_channel)) 
                             + (((IData)(9U) * (IData)(vlSelfRef.d3_vgg_like_top__DOT__conv1__DOT__engine__DOT__ic)) 
                                + ((IData)(3U) * (IData)(vlSelfRef.d3_vgg_like_top__DOT__conv1__DOT__engine__DOT__kr))))));
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
    vlSelfRef.d3_vgg_like_top__DOT__conv5__DOT__engine__DOT__last_item 
        = ((0x003fU == (IData)(vlSelfRef.d3_vgg_like_top__DOT__conv5__DOT__engine__DOT__ic)) 
           & ((2U == (IData)(vlSelfRef.d3_vgg_like_top__DOT__conv5__DOT__engine__DOT__kr)) 
              & (2U == (IData)(vlSelfRef.d3_vgg_like_top__DOT__conv5__DOT__engine__DOT__kc))));
    vlSelfRef.d3_vgg_like_top__DOT__c5_wa = (0x0001ffffU 
                                             & ((IData)(vlSelfRef.d3_vgg_like_top__DOT__conv5__DOT__engine__DOT__kc) 
                                                + (
                                                   ((IData)(9U) 
                                                    * 
                                                    ((IData)(vlSelfRef.d3_vgg_like_top__DOT__conv5__DOT__out_channel) 
                                                     << 6U)) 
                                                   + 
                                                   (((IData)(9U) 
                                                     * (IData)(vlSelfRef.d3_vgg_like_top__DOT__conv5__DOT__engine__DOT__ic)) 
                                                    + 
                                                    ((IData)(3U) 
                                                     * (IData)(vlSelfRef.d3_vgg_like_top__DOT__conv5__DOT__engine__DOT__kr))))));
}

void Vd3_vgg_like_top___024root___eval_nba(Vd3_vgg_like_top___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vd3_vgg_like_top___024root___eval_nba\n"); );
    Vd3_vgg_like_top__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    if ((3ULL & vlSelfRef.__VnbaTriggered[0U])) {
        Vd3_vgg_like_top___024root___nba_sequent__TOP__0(vlSelf);
    }
    if ((1ULL & vlSelfRef.__VnbaTriggered[0U])) {
        vlSelfRef.__VdlySet__d3_vgg_like_top__DOT__buffer_a__DOT__mem__v0 = 0U;
        vlSelfRef.__VdlySet__d3_vgg_like_top__DOT__buffer_b__DOT__mem__v0 = 0U;
        if (((IData)(vlSelfRef.__VdfgRegularize_h6e95ff9d_0_0) 
             | ((IData)(vlSelfRef.d3_vgg_like_top__DOT__conv2__DOT__postprocess_valid) 
                | ((IData)(vlSelfRef.d3_vgg_like_top__DOT__conv3__DOT__postprocess_valid) 
                   | ((IData)(vlSelfRef.d3_vgg_like_top__DOT__p2_owe) 
                      | (IData)(vlSelfRef.d3_vgg_like_top__DOT__gap_owe)))))) {
            if (vlSelfRef.__VdfgRegularize_h6e95ff9d_0_0) {
                vlSelfRef.__VdlyVal__d3_vgg_like_top__DOT__buffer_a__DOT__mem__v0 
                    = vlSelfRef.input_data;
                vlSelfRef.__VdlyDim0__d3_vgg_like_top__DOT__buffer_a__DOT__mem__v0 
                    = (0x00007fffU & vlSelfRef.input_addr);
            } else if (vlSelfRef.d3_vgg_like_top__DOT__conv2__DOT__postprocess_valid) {
                vlSelfRef.__VdlyVal__d3_vgg_like_top__DOT__buffer_a__DOT__mem__v0 
                    = vlSelfRef.d3_vgg_like_top__DOT__conv2__DOT__out_data_reg;
                vlSelfRef.__VdlyDim0__d3_vgg_like_top__DOT__buffer_a__DOT__mem__v0 
                    = (0x00007fffU & vlSelfRef.d3_vgg_like_top__DOT__conv2__DOT__out_addr_reg);
            } else if (vlSelfRef.d3_vgg_like_top__DOT__conv3__DOT__postprocess_valid) {
                vlSelfRef.__VdlyVal__d3_vgg_like_top__DOT__buffer_a__DOT__mem__v0 
                    = vlSelfRef.d3_vgg_like_top__DOT__conv3__DOT__out_data_reg;
                vlSelfRef.__VdlyDim0__d3_vgg_like_top__DOT__buffer_a__DOT__mem__v0 
                    = (0x00007fffU & vlSelfRef.d3_vgg_like_top__DOT__conv3__DOT__out_addr_reg);
            } else if (vlSelfRef.d3_vgg_like_top__DOT__p2_owe) {
                vlSelfRef.__VdlyVal__d3_vgg_like_top__DOT__buffer_a__DOT__mem__v0 
                    = vlSelfRef.d3_vgg_like_top__DOT__p2_od;
                vlSelfRef.__VdlyDim0__d3_vgg_like_top__DOT__buffer_a__DOT__mem__v0 
                    = (0x00007fffU & vlSelfRef.d3_vgg_like_top__DOT__p2_oa);
            } else {
                vlSelfRef.__VdlyVal__d3_vgg_like_top__DOT__buffer_a__DOT__mem__v0 
                    = vlSelfRef.d3_vgg_like_top__DOT__gap_od;
                vlSelfRef.__VdlyDim0__d3_vgg_like_top__DOT__buffer_a__DOT__mem__v0 
                    = (0x00007fffU & vlSelfRef.d3_vgg_like_top__DOT__gap_oa);
            }
            vlSelfRef.__VdlySet__d3_vgg_like_top__DOT__buffer_a__DOT__mem__v0 = 1U;
        }
        if (((IData)(vlSelfRef.d3_vgg_like_top__DOT__conv1__DOT__postprocess_valid) 
             | ((IData)(vlSelfRef.d3_vgg_like_top__DOT__p1_owe) 
                | ((IData)(vlSelfRef.d3_vgg_like_top__DOT__conv4__DOT__postprocess_valid) 
                   | (IData)(vlSelfRef.d3_vgg_like_top__DOT__conv5__DOT__postprocess_valid))))) {
            if (vlSelfRef.d3_vgg_like_top__DOT__conv1__DOT__postprocess_valid) {
                vlSelfRef.__VdlyVal__d3_vgg_like_top__DOT__buffer_b__DOT__mem__v0 
                    = vlSelfRef.d3_vgg_like_top__DOT__conv1__DOT__out_data_reg;
                vlSelfRef.__VdlyDim0__d3_vgg_like_top__DOT__buffer_b__DOT__mem__v0 
                    = (0x00007fffU & vlSelfRef.d3_vgg_like_top__DOT__conv1__DOT__out_addr_reg);
            } else if (vlSelfRef.d3_vgg_like_top__DOT__p1_owe) {
                vlSelfRef.__VdlyVal__d3_vgg_like_top__DOT__buffer_b__DOT__mem__v0 
                    = vlSelfRef.d3_vgg_like_top__DOT__p1_od;
                vlSelfRef.__VdlyDim0__d3_vgg_like_top__DOT__buffer_b__DOT__mem__v0 
                    = (0x00007fffU & vlSelfRef.d3_vgg_like_top__DOT__p1_oa);
            } else if (vlSelfRef.d3_vgg_like_top__DOT__conv4__DOT__postprocess_valid) {
                vlSelfRef.__VdlyVal__d3_vgg_like_top__DOT__buffer_b__DOT__mem__v0 
                    = vlSelfRef.d3_vgg_like_top__DOT__conv4__DOT__out_data_reg;
                vlSelfRef.__VdlyDim0__d3_vgg_like_top__DOT__buffer_b__DOT__mem__v0 
                    = (0x00007fffU & vlSelfRef.d3_vgg_like_top__DOT__conv4__DOT__out_addr_reg);
            } else {
                vlSelfRef.__VdlyVal__d3_vgg_like_top__DOT__buffer_b__DOT__mem__v0 
                    = vlSelfRef.d3_vgg_like_top__DOT__conv5__DOT__out_data_reg;
                vlSelfRef.__VdlyDim0__d3_vgg_like_top__DOT__buffer_b__DOT__mem__v0 
                    = (0x00007fffU & vlSelfRef.d3_vgg_like_top__DOT__conv5__DOT__out_addr_reg);
            }
            vlSelfRef.__VdlySet__d3_vgg_like_top__DOT__buffer_b__DOT__mem__v0 = 1U;
        }
        vlSelfRef.d3_vgg_like_top__DOT__fc_wd = ((0x04ffU 
                                                  >= (IData)(vlSelfRef.d3_vgg_like_top__DOT__rwf__DOT__rom_addr))
                                                  ? vlSelfRef.d3_vgg_like_top__DOT__rwf__DOT__rom
                                                 [vlSelfRef.d3_vgg_like_top__DOT__rwf__DOT__rom_addr]
                                                  : 0U);
    }
    if ((3ULL & vlSelfRef.__VnbaTriggered[0U])) {
        Vd3_vgg_like_top___024root___nba_sequent__TOP__2(vlSelf);
    }
    if ((1ULL & vlSelfRef.__VnbaTriggered[0U])) {
        Vd3_vgg_like_top___024root___nba_sequent__TOP__3(vlSelf);
    }
    if ((3ULL & vlSelfRef.__VnbaTriggered[0U])) {
        Vd3_vgg_like_top___024root___nba_sequent__TOP__4(vlSelf);
    }
}

void Vd3_vgg_like_top___024root___trigger_orInto__act_vec_vec(VlUnpacked<QData/*63:0*/, 1> &out, const VlUnpacked<QData/*63:0*/, 1> &in) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vd3_vgg_like_top___024root___trigger_orInto__act_vec_vec\n"); );
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
VL_ATTR_COLD void Vd3_vgg_like_top___024root___dump_triggers__act(const VlUnpacked<QData/*63:0*/, 1> &triggers, const std::string &tag);
#endif  // VL_DEBUG

bool Vd3_vgg_like_top___024root___eval_phase__act(Vd3_vgg_like_top___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vd3_vgg_like_top___024root___eval_phase__act\n"); );
    Vd3_vgg_like_top__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    Vd3_vgg_like_top___024root___eval_triggers_vec__act(vlSelf);
#ifdef VL_DEBUG
    if (VL_UNLIKELY(vlSymsp->_vm_contextp__->debug())) {
        Vd3_vgg_like_top___024root___dump_triggers__act(vlSelfRef.__VactTriggered, "act"s);
    }
#endif
    Vd3_vgg_like_top___024root___trigger_orInto__act_vec_vec(vlSelfRef.__VnbaTriggered, vlSelfRef.__VactTriggered);
    return (0U);
}

void Vd3_vgg_like_top___024root___trigger_clear__act(VlUnpacked<QData/*63:0*/, 1> &out) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vd3_vgg_like_top___024root___trigger_clear__act\n"); );
    // Locals
    IData/*31:0*/ n;
    // Body
    n = 0U;
    do {
        out[n] = 0ULL;
        n = ((IData)(1U) + n);
    } while ((1U > n));
}

bool Vd3_vgg_like_top___024root___eval_phase__nba(Vd3_vgg_like_top___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vd3_vgg_like_top___024root___eval_phase__nba\n"); );
    Vd3_vgg_like_top__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Locals
    CData/*0:0*/ __VnbaExecute;
    // Body
    __VnbaExecute = Vd3_vgg_like_top___024root___trigger_anySet__act(vlSelfRef.__VnbaTriggered);
    if (__VnbaExecute) {
        Vd3_vgg_like_top___024root___eval_nba(vlSelf);
        Vd3_vgg_like_top___024root___trigger_clear__act(vlSelfRef.__VnbaTriggered);
    }
    return (__VnbaExecute);
}

void Vd3_vgg_like_top___024root___eval(Vd3_vgg_like_top___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vd3_vgg_like_top___024root___eval\n"); );
    Vd3_vgg_like_top__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
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
            Vd3_vgg_like_top___024root___dump_triggers__ico(vlSelfRef.__VicoTriggered, "ico"s);
#endif
            VL_FATAL_MT("rtl/src/d3_vgg_like_top.v", 3, "", "DIDNOTCONVERGE: Input combinational region did not converge after '--converge-limit' of 10000 tries");
        }
        __VicoIterCount = ((IData)(1U) + __VicoIterCount);
        vlSelfRef.__VicoPhaseResult = Vd3_vgg_like_top___024root___eval_phase__ico(vlSelf);
        vlSelfRef.__VicoFirstIteration = 0U;
    } while (vlSelfRef.__VicoPhaseResult);
    __VnbaIterCount = 0U;
    do {
        if (VL_UNLIKELY(((0x00002710U < __VnbaIterCount)))) {
#ifdef VL_DEBUG
            Vd3_vgg_like_top___024root___dump_triggers__act(vlSelfRef.__VnbaTriggered, "nba"s);
#endif
            VL_FATAL_MT("rtl/src/d3_vgg_like_top.v", 3, "", "DIDNOTCONVERGE: NBA region did not converge after '--converge-limit' of 10000 tries");
        }
        __VnbaIterCount = ((IData)(1U) + __VnbaIterCount);
        vlSelfRef.__VactIterCount = 0U;
        do {
            if (VL_UNLIKELY(((0x00002710U < vlSelfRef.__VactIterCount)))) {
#ifdef VL_DEBUG
                Vd3_vgg_like_top___024root___dump_triggers__act(vlSelfRef.__VactTriggered, "act"s);
#endif
                VL_FATAL_MT("rtl/src/d3_vgg_like_top.v", 3, "", "DIDNOTCONVERGE: Active region did not converge after '--converge-limit' of 10000 tries");
            }
            vlSelfRef.__VactIterCount = ((IData)(1U) 
                                         + vlSelfRef.__VactIterCount);
            vlSelfRef.__VactPhaseResult = Vd3_vgg_like_top___024root___eval_phase__act(vlSelf);
        } while (vlSelfRef.__VactPhaseResult);
        vlSelfRef.__VnbaPhaseResult = Vd3_vgg_like_top___024root___eval_phase__nba(vlSelf);
    } while (vlSelfRef.__VnbaPhaseResult);
}

#ifdef VL_DEBUG
void Vd3_vgg_like_top___024root___eval_debug_assertions(Vd3_vgg_like_top___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vd3_vgg_like_top___024root___eval_debug_assertions\n"); );
    Vd3_vgg_like_top__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
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
