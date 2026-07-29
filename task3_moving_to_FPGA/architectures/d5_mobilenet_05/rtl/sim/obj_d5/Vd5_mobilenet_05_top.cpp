// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Model implementation (design independent parts)

#include "Vd5_mobilenet_05_top__pch.h"

//============================================================
// Constructors

Vd5_mobilenet_05_top::Vd5_mobilenet_05_top(VerilatedContext* _vcontextp__, const char* _vcname__)
    : VerilatedModel{*_vcontextp__}
    , vlSymsp{new Vd5_mobilenet_05_top__Syms(contextp(), _vcname__, this)}
    , clk{vlSymsp->TOP.clk}
    , rst_n{vlSymsp->TOP.rst_n}
    , start{vlSymsp->TOP.start}
    , layer_done{vlSymsp->TOP.layer_done}
    , layer_start{vlSymsp->TOP.layer_start}
    , layer_id{vlSymsp->TOP.layer_id}
    , op_kind{vlSymsp->TOP.op_kind}
    , height{vlSymsp->TOP.height}
    , width{vlSymsp->TOP.width}
    , stride{vlSymsp->TOP.stride}
    , busy{vlSymsp->TOP.busy}
    , done{vlSymsp->TOP.done}
    , in_channels{vlSymsp->TOP.in_channels}
    , out_channels{vlSymsp->TOP.out_channels}
    , weight_offset{vlSymsp->TOP.weight_offset}
    , rootp{&(vlSymsp->TOP)}
{
    // Register model with the context
    contextp()->addModel(this);
}

Vd5_mobilenet_05_top::Vd5_mobilenet_05_top(const char* _vcname__)
    : Vd5_mobilenet_05_top(Verilated::threadContextp(), _vcname__)
{
}

//============================================================
// Destructor

Vd5_mobilenet_05_top::~Vd5_mobilenet_05_top() {
    delete vlSymsp;
}

//============================================================
// Evaluation function

#ifdef VL_DEBUG
void Vd5_mobilenet_05_top___024root___eval_debug_assertions(Vd5_mobilenet_05_top___024root* vlSelf);
#endif  // VL_DEBUG
void Vd5_mobilenet_05_top___024root___eval_static(Vd5_mobilenet_05_top___024root* vlSelf);
void Vd5_mobilenet_05_top___024root___eval_initial(Vd5_mobilenet_05_top___024root* vlSelf);
void Vd5_mobilenet_05_top___024root___eval_settle(Vd5_mobilenet_05_top___024root* vlSelf);
void Vd5_mobilenet_05_top___024root___eval(Vd5_mobilenet_05_top___024root* vlSelf);

void Vd5_mobilenet_05_top::eval_step() {
    VL_DEBUG_IF(VL_DBG_MSGF("+++++TOP Evaluate Vd5_mobilenet_05_top::eval_step\n"); );
#ifdef VL_DEBUG
    // Debug assertions
    Vd5_mobilenet_05_top___024root___eval_debug_assertions(&(vlSymsp->TOP));
#endif  // VL_DEBUG
    vlSymsp->__Vm_deleter.deleteAll();
    if (VL_UNLIKELY(!vlSymsp->__Vm_didInit)) {
        VL_DEBUG_IF(VL_DBG_MSGF("+ Initial\n"););
        Vd5_mobilenet_05_top___024root___eval_static(&(vlSymsp->TOP));
        Vd5_mobilenet_05_top___024root___eval_initial(&(vlSymsp->TOP));
        Vd5_mobilenet_05_top___024root___eval_settle(&(vlSymsp->TOP));
        vlSymsp->__Vm_didInit = true;
    }
    VL_DEBUG_IF(VL_DBG_MSGF("+ Eval\n"););
    Vd5_mobilenet_05_top___024root___eval(&(vlSymsp->TOP));
    // Evaluate cleanup
    Verilated::endOfEval(vlSymsp->__Vm_evalMsgQp);
}

//============================================================
// Events and timing
bool Vd5_mobilenet_05_top::eventsPending() { return false; }

uint64_t Vd5_mobilenet_05_top::nextTimeSlot() {
    VL_FATAL_MT(__FILE__, __LINE__, "", "No delays in the design");
    return 0;
}

//============================================================
// Utilities

const char* Vd5_mobilenet_05_top::name() const {
    return vlSymsp->name();
}

//============================================================
// Invoke final blocks

void Vd5_mobilenet_05_top___024root___eval_final(Vd5_mobilenet_05_top___024root* vlSelf);

VL_ATTR_COLD void Vd5_mobilenet_05_top::final() {
    contextp()->executingFinal(true);
    Vd5_mobilenet_05_top___024root___eval_final(&(vlSymsp->TOP));
    contextp()->executingFinal(false);
}

//============================================================
// Implementations of abstract methods from VerilatedModel

const char* Vd5_mobilenet_05_top::hierName() const { return vlSymsp->name(); }
const char* Vd5_mobilenet_05_top::modelName() const { return "Vd5_mobilenet_05_top"; }
unsigned Vd5_mobilenet_05_top::threads() const { return 1; }
void Vd5_mobilenet_05_top::prepareClone() const { contextp()->prepareClone(); }
void Vd5_mobilenet_05_top::atClone() const {
    contextp()->threadPoolpOnClone();
}
