// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Symbol table internal header
//
// Internal details; most calling programs do not need this header,
// unless using verilator public meta comments.

#ifndef VERILATED_VD5_MOBILENET_05_TOP__SYMS_H_
#define VERILATED_VD5_MOBILENET_05_TOP__SYMS_H_  // guard

#include "verilated.h"

// INCLUDE MODEL CLASS

#include "Vd5_mobilenet_05_top.h"

// INCLUDE MODULE CLASSES
#include "Vd5_mobilenet_05_top___024root.h"

// SYMS CLASS (contains all model state)
class alignas(VL_CACHE_LINE_BYTES) Vd5_mobilenet_05_top__Syms final : public VerilatedSyms {
  public:
    // INTERNAL STATE
    Vd5_mobilenet_05_top* const __Vm_modelp;
    VlDeleter __Vm_deleter;
    bool __Vm_didInit = false;

    // MODULE INSTANCE STATE
    Vd5_mobilenet_05_top___024root TOP;

    // CONSTRUCTORS
    Vd5_mobilenet_05_top__Syms(VerilatedContext* contextp, const char* namep, Vd5_mobilenet_05_top* modelp);
    ~Vd5_mobilenet_05_top__Syms();

    // METHODS
    const char* name() const { return TOP.vlNamep; }
};

#endif  // guard
