// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design implementation internals
// See Vd3_vgg_like_top.h for the primary calling header

#include "Vd3_vgg_like_top__pch.h"

void Vd3_vgg_like_top___024root___ctor_var_reset(Vd3_vgg_like_top___024root* vlSelf);

Vd3_vgg_like_top___024root::Vd3_vgg_like_top___024root(Vd3_vgg_like_top__Syms* symsp, const char* namep)
 {
    vlSymsp = symsp;
    vlNamep = strdup(namep);
    // Reset structure values
    Vd3_vgg_like_top___024root___ctor_var_reset(this);
}

void Vd3_vgg_like_top___024root::__Vconfigure(bool first) {
    (void)first;  // Prevent unused variable warning
}

Vd3_vgg_like_top___024root::~Vd3_vgg_like_top___024root() {
    VL_DO_DANGLING(std::free(const_cast<char*>(vlNamep)), vlNamep);
}
