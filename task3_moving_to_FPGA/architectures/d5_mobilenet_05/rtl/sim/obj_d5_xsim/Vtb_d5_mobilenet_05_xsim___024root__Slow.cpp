// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design implementation internals
// See Vtb_d5_mobilenet_05_xsim.h for the primary calling header

#include "Vtb_d5_mobilenet_05_xsim__pch.h"

void Vtb_d5_mobilenet_05_xsim___024root___ctor_var_reset(Vtb_d5_mobilenet_05_xsim___024root* vlSelf);

Vtb_d5_mobilenet_05_xsim___024root::Vtb_d5_mobilenet_05_xsim___024root(Vtb_d5_mobilenet_05_xsim__Syms* symsp, const char* namep)
    : __VdlySched{*symsp->_vm_contextp__}
 {
    vlSymsp = symsp;
    vlNamep = strdup(namep);
    // Reset structure values
    Vtb_d5_mobilenet_05_xsim___024root___ctor_var_reset(this);
}

void Vtb_d5_mobilenet_05_xsim___024root::__Vconfigure(bool first) {
    (void)first;  // Prevent unused variable warning
}

Vtb_d5_mobilenet_05_xsim___024root::~Vtb_d5_mobilenet_05_xsim___024root() {
    VL_DO_DANGLING(std::free(const_cast<char*>(vlNamep)), vlNamep);
}
