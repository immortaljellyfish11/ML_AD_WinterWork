// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Model implementation (design independent parts)

#include "Vedge_cnn_top__pch.h"

//============================================================
// Constructors

Vedge_cnn_top::Vedge_cnn_top(VerilatedContext* _vcontextp__, const char* _vcname__)
    : VerilatedModel{*_vcontextp__}
    , vlSymsp{new Vedge_cnn_top__Syms(contextp(), _vcname__, this)}
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
    , conv4_start{vlSymsp->TOP.conv4_start}
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

Vedge_cnn_top::Vedge_cnn_top(const char* _vcname__)
    : Vedge_cnn_top(Verilated::threadContextp(), _vcname__)
{
}

//============================================================
// Destructor

Vedge_cnn_top::~Vedge_cnn_top() {
    delete vlSymsp;
}

//============================================================
// Evaluation function

#ifdef VL_DEBUG
void Vedge_cnn_top___024root___eval_debug_assertions(Vedge_cnn_top___024root* vlSelf);
#endif  // VL_DEBUG
void Vedge_cnn_top___024root___eval_static(Vedge_cnn_top___024root* vlSelf);
void Vedge_cnn_top___024root___eval_initial(Vedge_cnn_top___024root* vlSelf);
void Vedge_cnn_top___024root___eval_settle(Vedge_cnn_top___024root* vlSelf);
void Vedge_cnn_top___024root___eval(Vedge_cnn_top___024root* vlSelf);

void Vedge_cnn_top::eval_step() {
    VL_DEBUG_IF(VL_DBG_MSGF("+++++TOP Evaluate Vedge_cnn_top::eval_step\n"); );
#ifdef VL_DEBUG
    // Debug assertions
    Vedge_cnn_top___024root___eval_debug_assertions(&(vlSymsp->TOP));
#endif  // VL_DEBUG
    vlSymsp->__Vm_deleter.deleteAll();
    if (VL_UNLIKELY(!vlSymsp->__Vm_didInit)) {
        VL_DEBUG_IF(VL_DBG_MSGF("+ Initial\n"););
        Vedge_cnn_top___024root___eval_static(&(vlSymsp->TOP));
        Vedge_cnn_top___024root___eval_initial(&(vlSymsp->TOP));
        Vedge_cnn_top___024root___eval_settle(&(vlSymsp->TOP));
        vlSymsp->__Vm_didInit = true;
    }
    VL_DEBUG_IF(VL_DBG_MSGF("+ Eval\n"););
    Vedge_cnn_top___024root___eval(&(vlSymsp->TOP));
    // Evaluate cleanup
    Verilated::endOfEval(vlSymsp->__Vm_evalMsgQp);
}

//============================================================
// Events and timing
bool Vedge_cnn_top::eventsPending() { return false; }

uint64_t Vedge_cnn_top::nextTimeSlot() {
    VL_FATAL_MT(__FILE__, __LINE__, "", "No delays in the design");
    return 0;
}

//============================================================
// Utilities

const char* Vedge_cnn_top::name() const {
    return vlSymsp->name();
}

//============================================================
// Invoke final blocks

void Vedge_cnn_top___024root___eval_final(Vedge_cnn_top___024root* vlSelf);

VL_ATTR_COLD void Vedge_cnn_top::final() {
    contextp()->executingFinal(true);
    Vedge_cnn_top___024root___eval_final(&(vlSymsp->TOP));
    contextp()->executingFinal(false);
}

//============================================================
// Implementations of abstract methods from VerilatedModel

const char* Vedge_cnn_top::hierName() const { return vlSymsp->name(); }
const char* Vedge_cnn_top::modelName() const { return "Vedge_cnn_top"; }
unsigned Vedge_cnn_top::threads() const { return 1; }
void Vedge_cnn_top::prepareClone() const { contextp()->prepareClone(); }
void Vedge_cnn_top::atClone() const {
    contextp()->threadPoolpOnClone();
}
