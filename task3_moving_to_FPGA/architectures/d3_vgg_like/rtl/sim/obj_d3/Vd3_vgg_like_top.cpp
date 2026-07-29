// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Model implementation (design independent parts)

#include "Vd3_vgg_like_top__pch.h"

//============================================================
// Constructors

Vd3_vgg_like_top::Vd3_vgg_like_top(VerilatedContext* _vcontextp__, const char* _vcname__)
    : VerilatedModel{*_vcontextp__}
    , vlSymsp{new Vd3_vgg_like_top__Syms(contextp(), _vcname__, this)}
    , clk{vlSymsp->TOP.clk}
    , rst_n{vlSymsp->TOP.rst_n}
    , start{vlSymsp->TOP.start}
    , input_we{vlSymsp->TOP.input_we}
    , input_data{vlSymsp->TOP.input_data}
    , busy{vlSymsp->TOP.busy}
    , done{vlSymsp->TOP.done}
    , class_id{vlSymsp->TOP.class_id}
    , argmax_valid{vlSymsp->TOP.argmax_valid}
    , state_dbg{vlSymsp->TOP.state_dbg}
    , input_addr{vlSymsp->TOP.input_addr}
    , max_logit{vlSymsp->TOP.max_logit}
    , rootp{&(vlSymsp->TOP)}
{
    // Register model with the context
    contextp()->addModel(this);
}

Vd3_vgg_like_top::Vd3_vgg_like_top(const char* _vcname__)
    : Vd3_vgg_like_top(Verilated::threadContextp(), _vcname__)
{
}

//============================================================
// Destructor

Vd3_vgg_like_top::~Vd3_vgg_like_top() {
    delete vlSymsp;
}

//============================================================
// Evaluation function

#ifdef VL_DEBUG
void Vd3_vgg_like_top___024root___eval_debug_assertions(Vd3_vgg_like_top___024root* vlSelf);
#endif  // VL_DEBUG
void Vd3_vgg_like_top___024root___eval_static(Vd3_vgg_like_top___024root* vlSelf);
void Vd3_vgg_like_top___024root___eval_initial(Vd3_vgg_like_top___024root* vlSelf);
void Vd3_vgg_like_top___024root___eval_settle(Vd3_vgg_like_top___024root* vlSelf);
void Vd3_vgg_like_top___024root___eval(Vd3_vgg_like_top___024root* vlSelf);

void Vd3_vgg_like_top::eval_step() {
    VL_DEBUG_IF(VL_DBG_MSGF("+++++TOP Evaluate Vd3_vgg_like_top::eval_step\n"); );
#ifdef VL_DEBUG
    // Debug assertions
    Vd3_vgg_like_top___024root___eval_debug_assertions(&(vlSymsp->TOP));
#endif  // VL_DEBUG
    vlSymsp->__Vm_deleter.deleteAll();
    if (VL_UNLIKELY(!vlSymsp->__Vm_didInit)) {
        VL_DEBUG_IF(VL_DBG_MSGF("+ Initial\n"););
        Vd3_vgg_like_top___024root___eval_static(&(vlSymsp->TOP));
        Vd3_vgg_like_top___024root___eval_initial(&(vlSymsp->TOP));
        Vd3_vgg_like_top___024root___eval_settle(&(vlSymsp->TOP));
        vlSymsp->__Vm_didInit = true;
    }
    VL_DEBUG_IF(VL_DBG_MSGF("+ Eval\n"););
    Vd3_vgg_like_top___024root___eval(&(vlSymsp->TOP));
    // Evaluate cleanup
    Verilated::endOfEval(vlSymsp->__Vm_evalMsgQp);
}

//============================================================
// Events and timing
bool Vd3_vgg_like_top::eventsPending() { return false; }

uint64_t Vd3_vgg_like_top::nextTimeSlot() {
    VL_FATAL_MT(__FILE__, __LINE__, "", "No delays in the design");
    return 0;
}

//============================================================
// Utilities

const char* Vd3_vgg_like_top::name() const {
    return vlSymsp->name();
}

//============================================================
// Invoke final blocks

void Vd3_vgg_like_top___024root___eval_final(Vd3_vgg_like_top___024root* vlSelf);

VL_ATTR_COLD void Vd3_vgg_like_top::final() {
    contextp()->executingFinal(true);
    Vd3_vgg_like_top___024root___eval_final(&(vlSymsp->TOP));
    contextp()->executingFinal(false);
}

//============================================================
// Implementations of abstract methods from VerilatedModel

const char* Vd3_vgg_like_top::hierName() const { return vlSymsp->name(); }
const char* Vd3_vgg_like_top::modelName() const { return "Vd3_vgg_like_top"; }
unsigned Vd3_vgg_like_top::threads() const { return 1; }
void Vd3_vgg_like_top::prepareClone() const { contextp()->prepareClone(); }
void Vd3_vgg_like_top::atClone() const {
    contextp()->threadPoolpOnClone();
}
