// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Model implementation (design independent parts)

#include "Vtb_d5_mobilenet_05_xsim__pch.h"

//============================================================
// Constructors

Vtb_d5_mobilenet_05_xsim::Vtb_d5_mobilenet_05_xsim(VerilatedContext* _vcontextp__, const char* _vcname__)
    : VerilatedModel{*_vcontextp__}
    , vlSymsp{new Vtb_d5_mobilenet_05_xsim__Syms(contextp(), _vcname__, this)}
    , rootp{&(vlSymsp->TOP)}
{
    // Register model with the context
    contextp()->addModel(this);
}

Vtb_d5_mobilenet_05_xsim::Vtb_d5_mobilenet_05_xsim(const char* _vcname__)
    : Vtb_d5_mobilenet_05_xsim(Verilated::threadContextp(), _vcname__)
{
}

//============================================================
// Destructor

Vtb_d5_mobilenet_05_xsim::~Vtb_d5_mobilenet_05_xsim() {
    delete vlSymsp;
}

//============================================================
// Evaluation function

#ifdef VL_DEBUG
void Vtb_d5_mobilenet_05_xsim___024root___eval_debug_assertions(Vtb_d5_mobilenet_05_xsim___024root* vlSelf);
#endif  // VL_DEBUG
void Vtb_d5_mobilenet_05_xsim___024root___eval_static(Vtb_d5_mobilenet_05_xsim___024root* vlSelf);
void Vtb_d5_mobilenet_05_xsim___024root___eval_initial(Vtb_d5_mobilenet_05_xsim___024root* vlSelf);
void Vtb_d5_mobilenet_05_xsim___024root___eval_settle(Vtb_d5_mobilenet_05_xsim___024root* vlSelf);
void Vtb_d5_mobilenet_05_xsim___024root___eval(Vtb_d5_mobilenet_05_xsim___024root* vlSelf);

void Vtb_d5_mobilenet_05_xsim::eval_step() {
    VL_DEBUG_IF(VL_DBG_MSGF("+++++TOP Evaluate Vtb_d5_mobilenet_05_xsim::eval_step\n"); );
#ifdef VL_DEBUG
    // Debug assertions
    Vtb_d5_mobilenet_05_xsim___024root___eval_debug_assertions(&(vlSymsp->TOP));
#endif  // VL_DEBUG
    vlSymsp->__Vm_deleter.deleteAll();
    if (VL_UNLIKELY(!vlSymsp->__Vm_didInit)) {
        VL_DEBUG_IF(VL_DBG_MSGF("+ Initial\n"););
        Vtb_d5_mobilenet_05_xsim___024root___eval_static(&(vlSymsp->TOP));
        Vtb_d5_mobilenet_05_xsim___024root___eval_initial(&(vlSymsp->TOP));
        Vtb_d5_mobilenet_05_xsim___024root___eval_settle(&(vlSymsp->TOP));
        vlSymsp->__Vm_didInit = true;
    }
    VL_DEBUG_IF(VL_DBG_MSGF("+ Eval\n"););
    Vtb_d5_mobilenet_05_xsim___024root___eval(&(vlSymsp->TOP));
    // Evaluate cleanup
    Verilated::endOfEval(vlSymsp->__Vm_evalMsgQp);
}

//============================================================
// Events and timing
bool Vtb_d5_mobilenet_05_xsim::eventsPending() { return !vlSymsp->TOP.__VdlySched.empty() && !contextp()->gotFinish(); }

uint64_t Vtb_d5_mobilenet_05_xsim::nextTimeSlot() { return vlSymsp->TOP.__VdlySched.nextTimeSlot(); }

//============================================================
// Utilities

const char* Vtb_d5_mobilenet_05_xsim::name() const {
    return vlSymsp->name();
}

//============================================================
// Invoke final blocks

void Vtb_d5_mobilenet_05_xsim___024root___eval_final(Vtb_d5_mobilenet_05_xsim___024root* vlSelf);

VL_ATTR_COLD void Vtb_d5_mobilenet_05_xsim::final() {
    contextp()->executingFinal(true);
    Vtb_d5_mobilenet_05_xsim___024root___eval_final(&(vlSymsp->TOP));
    contextp()->executingFinal(false);
}

//============================================================
// Implementations of abstract methods from VerilatedModel

const char* Vtb_d5_mobilenet_05_xsim::hierName() const { return vlSymsp->name(); }
const char* Vtb_d5_mobilenet_05_xsim::modelName() const { return "Vtb_d5_mobilenet_05_xsim"; }
unsigned Vtb_d5_mobilenet_05_xsim::threads() const { return 1; }
void Vtb_d5_mobilenet_05_xsim::prepareClone() const { contextp()->prepareClone(); }
void Vtb_d5_mobilenet_05_xsim::atClone() const {
    contextp()->threadPoolpOnClone();
}
