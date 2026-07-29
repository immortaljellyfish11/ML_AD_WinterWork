module requantize #(
    parameter integer FRAC_BITS = 7,
    parameter signed [17:0] MULTIPLIER = 18'sd1,
    parameter integer SHIFT = FRAC_BITS
) (
    input  signed [31:0] value_in,
    output signed [31:0] value_out
);
    wire signed [24:0] value_narrow;
    wire signed [42:0] product;

    assign value_narrow = value_in[24:0];
    assign product = $signed(value_narrow) * $signed(MULTIPLIER);

    requantize_round_shift #(
        .SHIFT(SHIFT)
    ) round_shift (
        .product_in(product),
        .value_out(value_out)
    );

    /* verilator lint_off UNUSEDSIGNAL */
    wire unused_input_bits = |(value_in[31:25] ^ {7{value_in[24]}});
    /* verilator lint_on UNUSEDSIGNAL */
endmodule

/* verilator lint_off DECLFILENAME */
module requantize_round_shift #(
    parameter integer SHIFT = 7
) (
    input signed [42:0] product_in,
    output signed [31:0] value_out
);
    wire signed [42:0] result43;

    generate
        if (SHIFT == 0) begin : gen_no_shift
            assign result43 = product_in;
        end else begin : gen_round_shift
            localparam signed [42:0] ROUND_OFFSET = 43'sd1 <<< (SHIFT - 1);
            wire signed [42:0] magnitude;
            wire signed [42:0] rounded_magnitude;
            wire signed [42:0] shifted_magnitude;

            assign magnitude = product_in[42] ? -product_in : product_in;
            assign rounded_magnitude = magnitude + ROUND_OFFSET;
            assign shifted_magnitude = rounded_magnitude >>> SHIFT;
            assign result43 = product_in[42] ? -shifted_magnitude : shifted_magnitude;
        end
    endgenerate

    /* verilator lint_off UNUSEDSIGNAL */
    wire unused_result_bits = |result43[42:32];
    /* verilator lint_on UNUSEDSIGNAL */
    assign value_out = result43[31:0];
endmodule
/* verilator lint_on DECLFILENAME */
