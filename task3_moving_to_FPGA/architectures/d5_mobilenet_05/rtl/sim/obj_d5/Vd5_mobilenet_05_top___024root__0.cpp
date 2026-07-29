// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design implementation internals
// See Vd5_mobilenet_05_top.h for the primary calling header

#include "Vd5_mobilenet_05_top__pch.h"

void Vd5_mobilenet_05_top___024root___eval_triggers_vec__act(Vd5_mobilenet_05_top___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vd5_mobilenet_05_top___024root___eval_triggers_vec__act\n"); );
    Vd5_mobilenet_05_top__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
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

bool Vd5_mobilenet_05_top___024root___trigger_anySet__act(const VlUnpacked<QData/*63:0*/, 1> &in) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vd5_mobilenet_05_top___024root___trigger_anySet__act\n"); );
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

void Vd5_mobilenet_05_top___024root___nba_sequent__TOP__0(Vd5_mobilenet_05_top___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vd5_mobilenet_05_top___024root___nba_sequent__TOP__0\n"); );
    Vd5_mobilenet_05_top__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Locals
    CData/*4:0*/ __Vfunc_d5_mobilenet_05_top__DOT__layer_weight_bytes__0__id;
    __Vfunc_d5_mobilenet_05_top__DOT__layer_weight_bytes__0__id = 0;
    CData/*4:0*/ __Vtask_d5_mobilenet_05_top__DOT__set_descriptor__1__id;
    __Vtask_d5_mobilenet_05_top__DOT__set_descriptor__1__id = 0;
    // Body
    if (vlSelfRef.rst_n) {
        vlSelfRef.layer_start = 0U;
        vlSelfRef.done = 0U;
        if (vlSelfRef.d5_mobilenet_05_top__DOT__active) {
            if (vlSelfRef.layer_done) {
                if ((0x16U == (IData)(vlSelfRef.layer_id))) {
                    vlSelfRef.d5_mobilenet_05_top__DOT__active = 0U;
                    vlSelfRef.busy = 0U;
                    vlSelfRef.done = 1U;
                } else {
                    __Vfunc_d5_mobilenet_05_top__DOT__layer_weight_bytes__0__id 
                        = vlSelfRef.layer_id;
                    vlSelfRef.d5_mobilenet_05_top__DOT____VlemCall_0__layer_weight_bytes 
                        = (((((((((0U == (IData)(__Vfunc_d5_mobilenet_05_top__DOT__layer_weight_bytes__0__id)) 
                                  | (1U == (IData)(__Vfunc_d5_mobilenet_05_top__DOT__layer_weight_bytes__0__id))) 
                                 | (2U == (IData)(__Vfunc_d5_mobilenet_05_top__DOT__layer_weight_bytes__0__id))) 
                                | (3U == (IData)(__Vfunc_d5_mobilenet_05_top__DOT__layer_weight_bytes__0__id))) 
                               | (4U == (IData)(__Vfunc_d5_mobilenet_05_top__DOT__layer_weight_bytes__0__id))) 
                              | (5U == (IData)(__Vfunc_d5_mobilenet_05_top__DOT__layer_weight_bytes__0__id))) 
                             | (6U == (IData)(__Vfunc_d5_mobilenet_05_top__DOT__layer_weight_bytes__0__id))) 
                            | (7U == (IData)(__Vfunc_d5_mobilenet_05_top__DOT__layer_weight_bytes__0__id)))
                            ? ((0U == (IData)(__Vfunc_d5_mobilenet_05_top__DOT__layer_weight_bytes__0__id))
                                ? 0x000001b0U : ((1U 
                                                  == (IData)(__Vfunc_d5_mobilenet_05_top__DOT__layer_weight_bytes__0__id))
                                                  ? 0x00000090U
                                                  : 
                                                 ((2U 
                                                   == (IData)(__Vfunc_d5_mobilenet_05_top__DOT__layer_weight_bytes__0__id))
                                                   ? 0x00000200U
                                                   : 
                                                  ((3U 
                                                    == (IData)(__Vfunc_d5_mobilenet_05_top__DOT__layer_weight_bytes__0__id))
                                                    ? 0x00000120U
                                                    : 
                                                   ((4U 
                                                     == (IData)(__Vfunc_d5_mobilenet_05_top__DOT__layer_weight_bytes__0__id))
                                                     ? 0x00000800U
                                                     : 
                                                    ((5U 
                                                      == (IData)(__Vfunc_d5_mobilenet_05_top__DOT__layer_weight_bytes__0__id))
                                                      ? 0x00000240U
                                                      : 
                                                     ((6U 
                                                       == (IData)(__Vfunc_d5_mobilenet_05_top__DOT__layer_weight_bytes__0__id))
                                                       ? 0x00001000U
                                                       : 0x00000240U)))))))
                            : (((((((((8U == (IData)(__Vfunc_d5_mobilenet_05_top__DOT__layer_weight_bytes__0__id)) 
                                      | (9U == (IData)(__Vfunc_d5_mobilenet_05_top__DOT__layer_weight_bytes__0__id))) 
                                     | (0x0aU == (IData)(__Vfunc_d5_mobilenet_05_top__DOT__layer_weight_bytes__0__id))) 
                                    | (0x0bU == (IData)(__Vfunc_d5_mobilenet_05_top__DOT__layer_weight_bytes__0__id))) 
                                   | (0x0cU == (IData)(__Vfunc_d5_mobilenet_05_top__DOT__layer_weight_bytes__0__id))) 
                                  | (0x0dU == (IData)(__Vfunc_d5_mobilenet_05_top__DOT__layer_weight_bytes__0__id))) 
                                 | (0x0eU == (IData)(__Vfunc_d5_mobilenet_05_top__DOT__layer_weight_bytes__0__id))) 
                                | (0x0fU == (IData)(__Vfunc_d5_mobilenet_05_top__DOT__layer_weight_bytes__0__id)))
                                ? ((8U == (IData)(__Vfunc_d5_mobilenet_05_top__DOT__layer_weight_bytes__0__id))
                                    ? 0x00002000U : 
                                   ((9U == (IData)(__Vfunc_d5_mobilenet_05_top__DOT__layer_weight_bytes__0__id))
                                     ? 0x00000480U : 
                                    ((0x0aU == (IData)(__Vfunc_d5_mobilenet_05_top__DOT__layer_weight_bytes__0__id))
                                      ? 0x00004000U
                                      : ((0x0bU == (IData)(__Vfunc_d5_mobilenet_05_top__DOT__layer_weight_bytes__0__id))
                                          ? 0x00000480U
                                          : ((0x0cU 
                                              == (IData)(__Vfunc_d5_mobilenet_05_top__DOT__layer_weight_bytes__0__id))
                                              ? 0x00004000U
                                              : ((0x0dU 
                                                  == (IData)(__Vfunc_d5_mobilenet_05_top__DOT__layer_weight_bytes__0__id))
                                                  ? 0x00000900U
                                                  : 
                                                 ((0x0eU 
                                                   == (IData)(__Vfunc_d5_mobilenet_05_top__DOT__layer_weight_bytes__0__id))
                                                   ? 0x00008000U
                                                   : 0x00000900U)))))))
                                : ((0x10U == (IData)(__Vfunc_d5_mobilenet_05_top__DOT__layer_weight_bytes__0__id))
                                    ? 0x00010000U : 
                                   ((0x11U == (IData)(__Vfunc_d5_mobilenet_05_top__DOT__layer_weight_bytes__0__id))
                                     ? 0x00000900U : 
                                    ((0x12U == (IData)(__Vfunc_d5_mobilenet_05_top__DOT__layer_weight_bytes__0__id))
                                      ? 0x00020000U
                                      : ((0x13U == (IData)(__Vfunc_d5_mobilenet_05_top__DOT__layer_weight_bytes__0__id))
                                          ? 0x00001200U
                                          : ((0x14U 
                                              == (IData)(__Vfunc_d5_mobilenet_05_top__DOT__layer_weight_bytes__0__id))
                                              ? 0x00040000U
                                              : ((0x16U 
                                                  == (IData)(__Vfunc_d5_mobilenet_05_top__DOT__layer_weight_bytes__0__id))
                                                  ? 0x00001400U
                                                  : 0U))))))));
                    vlSelfRef.weight_offset = (vlSelfRef.weight_offset 
                                               + vlSelfRef.d5_mobilenet_05_top__DOT____VlemCall_0__layer_weight_bytes);
                    vlSelfRef.layer_start = 1U;
                    __Vtask_d5_mobilenet_05_top__DOT__set_descriptor__1__id 
                        = (0x0000001fU & ((IData)(1U) 
                                          + (IData)(vlSelfRef.layer_id)));
                    vlSelfRef.layer_id = __Vtask_d5_mobilenet_05_top__DOT__set_descriptor__1__id;
                    vlSelfRef.stride = 0U;
                    vlSelfRef.in_channels = 0U;
                    vlSelfRef.out_channels = 0U;
                    vlSelfRef.height = 0U;
                    vlSelfRef.width = 0U;
                    vlSelfRef.op_kind = 0U;
                    if (((((((((0U == (IData)(__Vtask_d5_mobilenet_05_top__DOT__set_descriptor__1__id)) 
                               | (1U == (IData)(__Vtask_d5_mobilenet_05_top__DOT__set_descriptor__1__id))) 
                              | (2U == (IData)(__Vtask_d5_mobilenet_05_top__DOT__set_descriptor__1__id))) 
                             | (3U == (IData)(__Vtask_d5_mobilenet_05_top__DOT__set_descriptor__1__id))) 
                            | (4U == (IData)(__Vtask_d5_mobilenet_05_top__DOT__set_descriptor__1__id))) 
                           | (5U == (IData)(__Vtask_d5_mobilenet_05_top__DOT__set_descriptor__1__id))) 
                          | (6U == (IData)(__Vtask_d5_mobilenet_05_top__DOT__set_descriptor__1__id))) 
                         | (7U == (IData)(__Vtask_d5_mobilenet_05_top__DOT__set_descriptor__1__id)))) {
                        if ((0U == (IData)(__Vtask_d5_mobilenet_05_top__DOT__set_descriptor__1__id))) {
                            vlSelfRef.op_kind = 0U;
                            vlSelfRef.in_channels = 3U;
                            vlSelfRef.out_channels = 0x0010U;
                            vlSelfRef.height = 0x20U;
                            vlSelfRef.width = 0x20U;
                            vlSelfRef.stride = 1U;
                        } else if ((1U == (IData)(__Vtask_d5_mobilenet_05_top__DOT__set_descriptor__1__id))) {
                            vlSelfRef.op_kind = 1U;
                            vlSelfRef.in_channels = 0x0010U;
                            vlSelfRef.out_channels = 0x0010U;
                            vlSelfRef.height = 0x20U;
                            vlSelfRef.width = 0x20U;
                            vlSelfRef.stride = 1U;
                        } else if ((2U == (IData)(__Vtask_d5_mobilenet_05_top__DOT__set_descriptor__1__id))) {
                            vlSelfRef.op_kind = 2U;
                            vlSelfRef.in_channels = 0x0010U;
                            vlSelfRef.out_channels = 0x0020U;
                            vlSelfRef.height = 0x20U;
                            vlSelfRef.width = 0x20U;
                            vlSelfRef.stride = 1U;
                        } else if ((3U == (IData)(__Vtask_d5_mobilenet_05_top__DOT__set_descriptor__1__id))) {
                            vlSelfRef.op_kind = 1U;
                            vlSelfRef.in_channels = 0x0020U;
                            vlSelfRef.out_channels = 0x0020U;
                            vlSelfRef.height = 0x20U;
                            vlSelfRef.width = 0x20U;
                            vlSelfRef.stride = 2U;
                        } else if ((4U == (IData)(__Vtask_d5_mobilenet_05_top__DOT__set_descriptor__1__id))) {
                            vlSelfRef.op_kind = 2U;
                            vlSelfRef.in_channels = 0x0020U;
                            vlSelfRef.out_channels = 0x0040U;
                            vlSelfRef.height = 0x10U;
                            vlSelfRef.width = 0x10U;
                            vlSelfRef.stride = 1U;
                        } else if ((5U == (IData)(__Vtask_d5_mobilenet_05_top__DOT__set_descriptor__1__id))) {
                            vlSelfRef.op_kind = 1U;
                            vlSelfRef.in_channels = 0x0040U;
                            vlSelfRef.out_channels = 0x0040U;
                            vlSelfRef.height = 0x10U;
                            vlSelfRef.width = 0x10U;
                            vlSelfRef.stride = 1U;
                        } else if ((6U == (IData)(__Vtask_d5_mobilenet_05_top__DOT__set_descriptor__1__id))) {
                            vlSelfRef.op_kind = 2U;
                            vlSelfRef.in_channels = 0x0040U;
                            vlSelfRef.out_channels = 0x0040U;
                            vlSelfRef.height = 0x10U;
                            vlSelfRef.width = 0x10U;
                            vlSelfRef.stride = 1U;
                        } else {
                            vlSelfRef.op_kind = 1U;
                            vlSelfRef.in_channels = 0x0040U;
                            vlSelfRef.out_channels = 0x0040U;
                            vlSelfRef.height = 0x10U;
                            vlSelfRef.width = 0x10U;
                            vlSelfRef.stride = 2U;
                        }
                    } else if (((((((((8U == (IData)(__Vtask_d5_mobilenet_05_top__DOT__set_descriptor__1__id)) 
                                      | (9U == (IData)(__Vtask_d5_mobilenet_05_top__DOT__set_descriptor__1__id))) 
                                     | (0x0aU == (IData)(__Vtask_d5_mobilenet_05_top__DOT__set_descriptor__1__id))) 
                                    | (0x0bU == (IData)(__Vtask_d5_mobilenet_05_top__DOT__set_descriptor__1__id))) 
                                   | (0x0cU == (IData)(__Vtask_d5_mobilenet_05_top__DOT__set_descriptor__1__id))) 
                                  | ((0x0dU == (IData)(__Vtask_d5_mobilenet_05_top__DOT__set_descriptor__1__id)) 
                                     || (0x0fU == (IData)(__Vtask_d5_mobilenet_05_top__DOT__set_descriptor__1__id)))) 
                                 | ((0x0eU == (IData)(__Vtask_d5_mobilenet_05_top__DOT__set_descriptor__1__id)) 
                                    || (0x10U == (IData)(__Vtask_d5_mobilenet_05_top__DOT__set_descriptor__1__id)))) 
                                | (0x11U == (IData)(__Vtask_d5_mobilenet_05_top__DOT__set_descriptor__1__id)))) {
                        if ((8U == (IData)(__Vtask_d5_mobilenet_05_top__DOT__set_descriptor__1__id))) {
                            vlSelfRef.op_kind = 2U;
                            vlSelfRef.in_channels = 0x0040U;
                            vlSelfRef.out_channels = 0x0080U;
                            vlSelfRef.height = 8U;
                            vlSelfRef.width = 8U;
                            vlSelfRef.stride = 1U;
                        } else if ((9U == (IData)(__Vtask_d5_mobilenet_05_top__DOT__set_descriptor__1__id))) {
                            vlSelfRef.op_kind = 1U;
                            vlSelfRef.in_channels = 0x0080U;
                            vlSelfRef.out_channels = 0x0080U;
                            vlSelfRef.height = 8U;
                            vlSelfRef.width = 8U;
                            vlSelfRef.stride = 1U;
                        } else if ((0x0aU == (IData)(__Vtask_d5_mobilenet_05_top__DOT__set_descriptor__1__id))) {
                            vlSelfRef.op_kind = 2U;
                            vlSelfRef.in_channels = 0x0080U;
                            vlSelfRef.out_channels = 0x0080U;
                            vlSelfRef.height = 8U;
                            vlSelfRef.width = 8U;
                            vlSelfRef.stride = 1U;
                        } else if ((0x0bU == (IData)(__Vtask_d5_mobilenet_05_top__DOT__set_descriptor__1__id))) {
                            vlSelfRef.op_kind = 1U;
                            vlSelfRef.in_channels = 0x0080U;
                            vlSelfRef.out_channels = 0x0080U;
                            vlSelfRef.height = 8U;
                            vlSelfRef.width = 8U;
                            vlSelfRef.stride = 2U;
                        } else if ((0x0cU == (IData)(__Vtask_d5_mobilenet_05_top__DOT__set_descriptor__1__id))) {
                            vlSelfRef.op_kind = 2U;
                            vlSelfRef.in_channels = 0x0080U;
                            vlSelfRef.out_channels = 0x0100U;
                            vlSelfRef.height = 4U;
                            vlSelfRef.width = 4U;
                            vlSelfRef.stride = 1U;
                        } else if (((0x0dU == (IData)(__Vtask_d5_mobilenet_05_top__DOT__set_descriptor__1__id)) 
                                    || (0x0fU == (IData)(__Vtask_d5_mobilenet_05_top__DOT__set_descriptor__1__id)))) {
                            vlSelfRef.op_kind = 1U;
                            vlSelfRef.in_channels = 0x0100U;
                            vlSelfRef.out_channels = 0x0100U;
                            vlSelfRef.height = 4U;
                            vlSelfRef.width = 4U;
                            vlSelfRef.stride = 1U;
                        } else if (((0x0eU == (IData)(__Vtask_d5_mobilenet_05_top__DOT__set_descriptor__1__id)) 
                                    || (0x10U == (IData)(__Vtask_d5_mobilenet_05_top__DOT__set_descriptor__1__id)))) {
                            vlSelfRef.op_kind = 2U;
                            vlSelfRef.in_channels = 0x0100U;
                            vlSelfRef.out_channels = 0x0100U;
                            vlSelfRef.height = 4U;
                            vlSelfRef.width = 4U;
                            vlSelfRef.stride = 1U;
                        } else {
                            vlSelfRef.op_kind = 1U;
                            vlSelfRef.in_channels = 0x0100U;
                            vlSelfRef.out_channels = 0x0100U;
                            vlSelfRef.height = 4U;
                            vlSelfRef.width = 4U;
                            vlSelfRef.stride = 2U;
                        }
                    } else if ((0x12U == (IData)(__Vtask_d5_mobilenet_05_top__DOT__set_descriptor__1__id))) {
                        vlSelfRef.op_kind = 2U;
                        vlSelfRef.in_channels = 0x0100U;
                        vlSelfRef.out_channels = 0x0200U;
                        vlSelfRef.height = 2U;
                        vlSelfRef.width = 2U;
                        vlSelfRef.stride = 1U;
                    } else if ((0x13U == (IData)(__Vtask_d5_mobilenet_05_top__DOT__set_descriptor__1__id))) {
                        vlSelfRef.op_kind = 1U;
                        vlSelfRef.in_channels = 0x0200U;
                        vlSelfRef.out_channels = 0x0200U;
                        vlSelfRef.height = 2U;
                        vlSelfRef.width = 2U;
                        vlSelfRef.stride = 1U;
                    } else if ((0x14U == (IData)(__Vtask_d5_mobilenet_05_top__DOT__set_descriptor__1__id))) {
                        vlSelfRef.op_kind = 2U;
                        vlSelfRef.in_channels = 0x0200U;
                        vlSelfRef.out_channels = 0x0200U;
                        vlSelfRef.height = 2U;
                        vlSelfRef.width = 2U;
                        vlSelfRef.stride = 1U;
                    } else if ((0x15U == (IData)(__Vtask_d5_mobilenet_05_top__DOT__set_descriptor__1__id))) {
                        vlSelfRef.op_kind = 3U;
                        vlSelfRef.in_channels = 0x0200U;
                        vlSelfRef.out_channels = 0x0200U;
                        vlSelfRef.height = 2U;
                        vlSelfRef.width = 2U;
                        vlSelfRef.stride = 1U;
                    } else if ((0x16U == (IData)(__Vtask_d5_mobilenet_05_top__DOT__set_descriptor__1__id))) {
                        vlSelfRef.op_kind = 4U;
                        vlSelfRef.in_channels = 0x0200U;
                        vlSelfRef.out_channels = 0x000aU;
                        vlSelfRef.height = 1U;
                        vlSelfRef.width = 1U;
                        vlSelfRef.stride = 1U;
                    }
                }
            }
        } else {
            vlSelfRef.busy = 0U;
            if (vlSelfRef.start) {
                vlSelfRef.d5_mobilenet_05_top__DOT__active = 1U;
                vlSelfRef.busy = 1U;
                vlSelfRef.weight_offset = 0U;
                vlSelfRef.layer_id = 0U;
                vlSelfRef.stride = 0U;
                vlSelfRef.layer_start = 1U;
                vlSelfRef.in_channels = 0U;
                vlSelfRef.out_channels = 0U;
                vlSelfRef.height = 0U;
                vlSelfRef.width = 0U;
                vlSelfRef.op_kind = 0U;
                vlSelfRef.op_kind = 0U;
                vlSelfRef.in_channels = 3U;
                vlSelfRef.out_channels = 0x0010U;
                vlSelfRef.height = 0x20U;
                vlSelfRef.width = 0x20U;
                vlSelfRef.stride = 1U;
            }
        }
    } else {
        vlSelfRef.d5_mobilenet_05_top__DOT__active = 0U;
        vlSelfRef.layer_start = 0U;
        vlSelfRef.layer_id = 0U;
        vlSelfRef.op_kind = 0U;
        vlSelfRef.in_channels = 0U;
        vlSelfRef.out_channels = 0U;
        vlSelfRef.height = 0U;
        vlSelfRef.width = 0U;
        vlSelfRef.stride = 0U;
        vlSelfRef.weight_offset = 0U;
        vlSelfRef.busy = 0U;
        vlSelfRef.done = 0U;
    }
}

void Vd5_mobilenet_05_top___024root___eval_nba(Vd5_mobilenet_05_top___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vd5_mobilenet_05_top___024root___eval_nba\n"); );
    Vd5_mobilenet_05_top__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    if ((3ULL & vlSelfRef.__VnbaTriggered[0U])) {
        Vd5_mobilenet_05_top___024root___nba_sequent__TOP__0(vlSelf);
    }
}

void Vd5_mobilenet_05_top___024root___trigger_orInto__act_vec_vec(VlUnpacked<QData/*63:0*/, 1> &out, const VlUnpacked<QData/*63:0*/, 1> &in) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vd5_mobilenet_05_top___024root___trigger_orInto__act_vec_vec\n"); );
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
VL_ATTR_COLD void Vd5_mobilenet_05_top___024root___dump_triggers__act(const VlUnpacked<QData/*63:0*/, 1> &triggers, const std::string &tag);
#endif  // VL_DEBUG

bool Vd5_mobilenet_05_top___024root___eval_phase__act(Vd5_mobilenet_05_top___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vd5_mobilenet_05_top___024root___eval_phase__act\n"); );
    Vd5_mobilenet_05_top__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    Vd5_mobilenet_05_top___024root___eval_triggers_vec__act(vlSelf);
#ifdef VL_DEBUG
    if (VL_UNLIKELY(vlSymsp->_vm_contextp__->debug())) {
        Vd5_mobilenet_05_top___024root___dump_triggers__act(vlSelfRef.__VactTriggered, "act"s);
    }
#endif
    Vd5_mobilenet_05_top___024root___trigger_orInto__act_vec_vec(vlSelfRef.__VnbaTriggered, vlSelfRef.__VactTriggered);
    return (0U);
}

void Vd5_mobilenet_05_top___024root___trigger_clear__act(VlUnpacked<QData/*63:0*/, 1> &out) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vd5_mobilenet_05_top___024root___trigger_clear__act\n"); );
    // Locals
    IData/*31:0*/ n;
    // Body
    n = 0U;
    do {
        out[n] = 0ULL;
        n = ((IData)(1U) + n);
    } while ((1U > n));
}

bool Vd5_mobilenet_05_top___024root___eval_phase__nba(Vd5_mobilenet_05_top___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vd5_mobilenet_05_top___024root___eval_phase__nba\n"); );
    Vd5_mobilenet_05_top__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Locals
    CData/*0:0*/ __VnbaExecute;
    // Body
    __VnbaExecute = Vd5_mobilenet_05_top___024root___trigger_anySet__act(vlSelfRef.__VnbaTriggered);
    if (__VnbaExecute) {
        Vd5_mobilenet_05_top___024root___eval_nba(vlSelf);
        Vd5_mobilenet_05_top___024root___trigger_clear__act(vlSelfRef.__VnbaTriggered);
    }
    return (__VnbaExecute);
}

void Vd5_mobilenet_05_top___024root___eval(Vd5_mobilenet_05_top___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vd5_mobilenet_05_top___024root___eval\n"); );
    Vd5_mobilenet_05_top__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Locals
    IData/*31:0*/ __VnbaIterCount;
    // Body
    __VnbaIterCount = 0U;
    do {
        if (VL_UNLIKELY(((0x00002710U < __VnbaIterCount)))) {
#ifdef VL_DEBUG
            Vd5_mobilenet_05_top___024root___dump_triggers__act(vlSelfRef.__VnbaTriggered, "nba"s);
#endif
            VL_FATAL_MT("rtl/src/d5_mobilenet_05_top.v", 10, "", "DIDNOTCONVERGE: NBA region did not converge after '--converge-limit' of 10000 tries");
        }
        __VnbaIterCount = ((IData)(1U) + __VnbaIterCount);
        vlSelfRef.__VactIterCount = 0U;
        do {
            if (VL_UNLIKELY(((0x00002710U < vlSelfRef.__VactIterCount)))) {
#ifdef VL_DEBUG
                Vd5_mobilenet_05_top___024root___dump_triggers__act(vlSelfRef.__VactTriggered, "act"s);
#endif
                VL_FATAL_MT("rtl/src/d5_mobilenet_05_top.v", 10, "", "DIDNOTCONVERGE: Active region did not converge after '--converge-limit' of 10000 tries");
            }
            vlSelfRef.__VactIterCount = ((IData)(1U) 
                                         + vlSelfRef.__VactIterCount);
            vlSelfRef.__VactPhaseResult = Vd5_mobilenet_05_top___024root___eval_phase__act(vlSelf);
        } while (vlSelfRef.__VactPhaseResult);
        vlSelfRef.__VnbaPhaseResult = Vd5_mobilenet_05_top___024root___eval_phase__nba(vlSelf);
    } while (vlSelfRef.__VnbaPhaseResult);
}

#ifdef VL_DEBUG
void Vd5_mobilenet_05_top___024root___eval_debug_assertions(Vd5_mobilenet_05_top___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vd5_mobilenet_05_top___024root___eval_debug_assertions\n"); );
    Vd5_mobilenet_05_top__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
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
    if (VL_UNLIKELY(((vlSelfRef.layer_done & 0xfeU)))) {
        Verilated::overWidthError("layer_done");
    }
}
#endif  // VL_DEBUG
