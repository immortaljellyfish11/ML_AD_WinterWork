module requantize #(
    parameter integer FRAC_BITS = 7
) (
    input  signed [31:0] value_in,
    output signed [31:0] value_out
);
    assign value_out = value_in >>> FRAC_BITS;
endmodule
