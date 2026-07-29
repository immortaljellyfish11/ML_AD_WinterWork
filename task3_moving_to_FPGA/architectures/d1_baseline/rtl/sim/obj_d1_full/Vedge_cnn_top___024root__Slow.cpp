// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design implementation internals
// See Vedge_cnn_top.h for the primary calling header

#include "Vedge_cnn_top__pch.h"

void Vedge_cnn_top___024root___ctor_var_reset(Vedge_cnn_top___024root* vlSelf);

Vedge_cnn_top___024root::Vedge_cnn_top___024root(Vedge_cnn_top__Syms* symsp, const char* namep)
 {
    vlSymsp = symsp;
    vlNamep = strdup(namep);
    // Reset structure values
    Vedge_cnn_top___024root___ctor_var_reset(this);
}

void Vedge_cnn_top___024root::__Vconfigure(bool first) {
    (void)first;  // Prevent unused variable warning
}

Vedge_cnn_top___024root::~Vedge_cnn_top___024root() {
    VL_DO_DANGLING(std::free(const_cast<char*>(vlNamep)), vlNamep);
}
