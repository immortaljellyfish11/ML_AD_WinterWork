// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design implementation internals
// See Vd5_mobilenet_05_top.h for the primary calling header

#include "Vd5_mobilenet_05_top__pch.h"

void Vd5_mobilenet_05_top___024root___ctor_var_reset(Vd5_mobilenet_05_top___024root* vlSelf);

Vd5_mobilenet_05_top___024root::Vd5_mobilenet_05_top___024root(Vd5_mobilenet_05_top__Syms* symsp, const char* namep)
 {
    vlSymsp = symsp;
    vlNamep = strdup(namep);
    // Reset structure values
    Vd5_mobilenet_05_top___024root___ctor_var_reset(this);
}

void Vd5_mobilenet_05_top___024root::__Vconfigure(bool first) {
    (void)first;  // Prevent unused variable warning
}

Vd5_mobilenet_05_top___024root::~Vd5_mobilenet_05_top___024root() {
    VL_DO_DANGLING(std::free(const_cast<char*>(vlNamep)), vlNamep);
}
