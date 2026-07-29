// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Model implementation (design independent parts)

#include "Vd2_small_top__pch.h"

//============================================================
// Constructors

Vd2_small_top::Vd2_small_top(VerilatedContext* _vcontextp__, const char* _vcname__)
    : VerilatedModel{*_vcontextp__}
    , vlSymsp{new Vd2_small_top__Syms(contextp(), _vcname__, this)}
    , clk{vlSymsp->TOP.clk}
    , rst_n{vlSymsp->TOP.rst_n}
    , start{vlSymsp->TOP.start}
    , input_we{vlSymsp->TOP.input_we}
    , input_data{vlSymsp->TOP.input_data}
    , conv1_start{vlSymsp->TOP.conv1_start}
    , pool1_start{vlSymsp->TOP.pool1_start}
    , conv2_start{vlSymsp->TOP.conv2_start}
    , pool2_start{vlSymsp->TOP.pool2_start}
    , conv3_start{vlSymsp->TOP.conv3_start}
    , gap_start{vlSymsp->TOP.gap_start}
    , linear_start{vlSymsp->TOP.linear_start}
    , argmax_valid{vlSymsp->TOP.argmax_valid}
    , busy{vlSymsp->TOP.busy}
    , done{vlSymsp->TOP.done}
    , class_id{vlSymsp->TOP.class_id}
    , state_dbg{vlSymsp->TOP.state_dbg}
    , input_addr{vlSymsp->TOP.input_addr}
    , max_logit{vlSymsp->TOP.max_logit}
    , rootp{&(vlSymsp->TOP)}
{
    // Register model with the context
    contextp()->addModel(this);
}

Vd2_small_top::Vd2_small_top(const char* _vcname__)
    : Vd2_small_top(Verilated::threadContextp(), _vcname__)
{
}

//============================================================
// Destructor

Vd2_small_top::~Vd2_small_top() {
    delete vlSymsp;
}

//============================================================
// Evaluation function

#ifdef VL_DEBUG
void Vd2_small_top___024root___eval_debug_assertions(Vd2_small_top___024root* vlSelf);
#endif  // VL_DEBUG
void Vd2_small_top___024root___eval_static(Vd2_small_top___024root* vlSelf);
void Vd2_small_top___024root___eval_initial(Vd2_small_top___024root* vlSelf);
void Vd2_small_top___024root___eval_settle(Vd2_small_top___024root* vlSelf);
void Vd2_small_top___024root___eval(Vd2_small_top___024root* vlSelf);

void Vd2_small_top::eval_step() {
    VL_DEBUG_IF(VL_DBG_MSGF("+++++TOP Evaluate Vd2_small_top::eval_step\n"); );
#ifdef VL_DEBUG
    // Debug assertions
    Vd2_small_top___024root___eval_debug_assertions(&(vlSymsp->TOP));
#endif  // VL_DEBUG
    vlSymsp->__Vm_deleter.deleteAll();
    if (VL_UNLIKELY(!vlSymsp->__Vm_didInit)) {
        VL_DEBUG_IF(VL_DBG_MSGF("+ Initial\n"););
        Vd2_small_top___024root___eval_static(&(vlSymsp->TOP));
        Vd2_small_top___024root___eval_initial(&(vlSymsp->TOP));
        Vd2_small_top___024root___eval_settle(&(vlSymsp->TOP));
        vlSymsp->__Vm_didInit = true;
    }
    VL_DEBUG_IF(VL_DBG_MSGF("+ Eval\n"););
    Vd2_small_top___024root___eval(&(vlSymsp->TOP));
    // Evaluate cleanup
    Verilated::endOfEval(vlSymsp->__Vm_evalMsgQp);
}

//============================================================
// Events and timing
bool Vd2_small_top::eventsPending() { return false; }

uint64_t Vd2_small_top::nextTimeSlot() {
    VL_FATAL_MT(__FILE__, __LINE__, "", "No delays in the design");
    return 0;
}

//============================================================
// Utilities

const char* Vd2_small_top::name() const {
    return vlSymsp->name();
}

//============================================================
// Invoke final blocks

void Vd2_small_top___024root___eval_final(Vd2_small_top___024root* vlSelf);

VL_ATTR_COLD void Vd2_small_top::final() {
    contextp()->executingFinal(true);
    Vd2_small_top___024root___eval_final(&(vlSymsp->TOP));
    contextp()->executingFinal(false);
}

//============================================================
// Implementations of abstract methods from VerilatedModel

const char* Vd2_small_top::hierName() const { return vlSymsp->name(); }
const char* Vd2_small_top::modelName() const { return "Vd2_small_top"; }
unsigned Vd2_small_top::threads() const { return 1; }
void Vd2_small_top::prepareClone() const { contextp()->prepareClone(); }
void Vd2_small_top::atClone() const {
    contextp()->threadPoolpOnClone();
}
