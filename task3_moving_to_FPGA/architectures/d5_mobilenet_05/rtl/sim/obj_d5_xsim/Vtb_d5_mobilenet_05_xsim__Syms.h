// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Symbol table internal header
//
// Internal details; most calling programs do not need this header,
// unless using verilator public meta comments.

#ifndef VERILATED_VTB_D5_MOBILENET_05_XSIM__SYMS_H_
#define VERILATED_VTB_D5_MOBILENET_05_XSIM__SYMS_H_  // guard

#include "verilated.h"

// INCLUDE MODEL CLASS

#include "Vtb_d5_mobilenet_05_xsim.h"

// INCLUDE MODULE CLASSES
#include "Vtb_d5_mobilenet_05_xsim___024root.h"

// SYMS CLASS (contains all model state)
class alignas(VL_CACHE_LINE_BYTES) Vtb_d5_mobilenet_05_xsim__Syms final : public VerilatedSyms {
  public:
    // INTERNAL STATE
    Vtb_d5_mobilenet_05_xsim* const __Vm_modelp;
    VlDeleter __Vm_deleter;
    bool __Vm_didInit = false;

    // MODULE INSTANCE STATE
    Vtb_d5_mobilenet_05_xsim___024root TOP;

    // CONSTRUCTORS
    Vtb_d5_mobilenet_05_xsim__Syms(VerilatedContext* contextp, const char* namep, Vtb_d5_mobilenet_05_xsim* modelp);
    ~Vtb_d5_mobilenet_05_xsim__Syms();

    // METHODS
    const char* name() const { return TOP.vlNamep; }
};

#endif  // guard
