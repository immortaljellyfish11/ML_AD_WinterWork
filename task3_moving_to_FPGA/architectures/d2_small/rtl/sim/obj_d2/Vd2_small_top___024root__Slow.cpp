// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design implementation internals
// See Vd2_small_top.h for the primary calling header

#include "Vd2_small_top__pch.h"

void Vd2_small_top___024root___ctor_var_reset(Vd2_small_top___024root* vlSelf);

Vd2_small_top___024root::Vd2_small_top___024root(Vd2_small_top__Syms* symsp, const char* namep)
 {
    vlSymsp = symsp;
    vlNamep = strdup(namep);
    // Reset structure values
    Vd2_small_top___024root___ctor_var_reset(this);
}

void Vd2_small_top___024root::__Vconfigure(bool first) {
    (void)first;  // Prevent unused variable warning
}

Vd2_small_top___024root::~Vd2_small_top___024root() {
    VL_DO_DANGLING(std::free(const_cast<char*>(vlNamep)), vlNamep);
}
