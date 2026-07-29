// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Symbol table internal header
//
// Internal details; most calling programs do not need this header,
// unless using verilator public meta comments.

#ifndef VERILATED_VD2_SMALL_TOP__SYMS_H_
#define VERILATED_VD2_SMALL_TOP__SYMS_H_  // guard

#include "verilated.h"

// INCLUDE MODEL CLASS

#include "Vd2_small_top.h"

// INCLUDE MODULE CLASSES
#include "Vd2_small_top___024root.h"

// SYMS CLASS (contains all model state)
class alignas(VL_CACHE_LINE_BYTES) Vd2_small_top__Syms final : public VerilatedSyms {
  public:
    // INTERNAL STATE
    Vd2_small_top* const __Vm_modelp;
    VlDeleter __Vm_deleter;
    bool __Vm_didInit = false;

    // MODULE INSTANCE STATE
    Vd2_small_top___024root        TOP;

    // CONSTRUCTORS
    Vd2_small_top__Syms(VerilatedContext* contextp, const char* namep, Vd2_small_top* modelp);
    ~Vd2_small_top__Syms();

    // METHODS
    const char* name() const { return TOP.vlNamep; }
};

#endif  // guard
