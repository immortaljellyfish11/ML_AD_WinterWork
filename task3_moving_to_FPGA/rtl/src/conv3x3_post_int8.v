module conv3x3_post_int8 (
    input  signed [7:0]  act0,
    input  signed [7:0]  act1,
    input  signed [7:0]  act2,
    input  signed [7:0]  act3,
    input  signed [7:0]  act4,
    input  signed [7:0]  act5,
    input  signed [7:0]  act6,
    input  signed [7:0]  act7,
    input  signed [7:0]  act8,

    input  signed [7:0]  wgt0,
    input  signed [7:0]  wgt1,
    input  signed [7:0]  wgt2,
    input  signed [7:0]  wgt3,
    input  signed [7:0]  wgt4,
    input  signed [7:0]  wgt5,
    input  signed [7:0]  wgt6,
    input  signed [7:0]  wgt7,
    input  signed [7:0]  wgt8,

    input  signed [31:0] bias,
    output signed [7:0]  out
);
    wire signed [31:0] conv_raw;
    wire signed [31:0] conv_bias;
    wire signed [31:0] conv_quant;
    wire signed [31:0] conv_relu;

    pe_3x3 pe (
        .act0(act0), .act1(act1), .act2(act2),
        .act3(act3), .act4(act4), .act5(act5),
        .act6(act6), .act7(act7), .act8(act8),
        .wgt0(wgt0), .wgt1(wgt1), .wgt2(wgt2),
        .wgt3(wgt3), .wgt4(wgt4), .wgt5(wgt5),
        .wgt6(wgt6), .wgt7(wgt7), .wgt8(wgt8),
        .result(conv_raw)
    );

    assign conv_bias = conv_raw + bias;

    requantize #(.FRAC_BITS(7)) quant (
        .value_in(conv_bias),
        .value_out(conv_quant)
    );

    relu #(.WIDTH(32)) activation (
        .value_in(conv_quant),
        .value_out(conv_relu)
    );

    saturate_int8 output_saturate (
        .value_in(conv_relu),
        .value_out(out)
    );
endmodule
