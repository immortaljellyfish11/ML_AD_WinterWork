module cnn_tiny_4x4_top (
    input  signed [7:0] x00,
    input  signed [7:0] x01,
    input  signed [7:0] x02,
    input  signed [7:0] x03,
    input  signed [7:0] x10,
    input  signed [7:0] x11,
    input  signed [7:0] x12,
    input  signed [7:0] x13,
    input  signed [7:0] x20,
    input  signed [7:0] x21,
    input  signed [7:0] x22,
    input  signed [7:0] x23,
    input  signed [7:0] x30,
    input  signed [7:0] x31,
    input  signed [7:0] x32,
    input  signed [7:0] x33,

    input  signed [7:0] k00,
    input  signed [7:0] k01,
    input  signed [7:0] k02,
    input  signed [7:0] k10,
    input  signed [7:0] k11,
    input  signed [7:0] k12,
    input  signed [7:0] k20,
    input  signed [7:0] k21,
    input  signed [7:0] k22,

    input  signed [31:0] conv_bias,
    input  signed [7:0]  fc0_weight,
    input  signed [7:0]  fc1_weight,
    input  signed [31:0] fc0_bias,
    input  signed [31:0] fc1_bias,

    output signed [7:0]  conv_y00,
    output signed [7:0]  conv_y01,
    output signed [7:0]  conv_y02,
    output signed [7:0]  conv_y03,
    output signed [7:0]  conv_y10,
    output signed [7:0]  conv_y11,
    output signed [7:0]  conv_y12,
    output signed [7:0]  conv_y13,
    output signed [7:0]  conv_y20,
    output signed [7:0]  conv_y21,
    output signed [7:0]  conv_y22,
    output signed [7:0]  conv_y23,
    output signed [7:0]  conv_y30,
    output signed [7:0]  conv_y31,
    output signed [7:0]  conv_y32,
    output signed [7:0]  conv_y33,

    output signed [7:0]  pool_y00,
    output signed [7:0]  pool_y01,
    output signed [7:0]  pool_y10,
    output signed [7:0]  pool_y11,
    output signed [31:0] gap_sum,
    output signed [7:0]  gap_avg,
    output signed [31:0] logit0,
    output signed [31:0] logit1,
    output              class_id
);
    wire signed [31:0] fc0_product;
    wire signed [31:0] fc1_product;

    single_channel_conv3x3_4x4 conv (
        .x00(x00), .x01(x01), .x02(x02), .x03(x03),
        .x10(x10), .x11(x11), .x12(x12), .x13(x13),
        .x20(x20), .x21(x21), .x22(x22), .x23(x23),
        .x30(x30), .x31(x31), .x32(x32), .x33(x33),
        .k00(k00), .k01(k01), .k02(k02),
        .k10(k10), .k11(k11), .k12(k12),
        .k20(k20), .k21(k21), .k22(k22),
        .bias(conv_bias),
        .y00(conv_y00), .y01(conv_y01), .y02(conv_y02), .y03(conv_y03),
        .y10(conv_y10), .y11(conv_y11), .y12(conv_y12), .y13(conv_y13),
        .y20(conv_y20), .y21(conv_y21), .y22(conv_y22), .y23(conv_y23),
        .y30(conv_y30), .y31(conv_y31), .y32(conv_y32), .y33(conv_y33)
    );

    maxpool_2x2 pool00 (
        .in0(conv_y00), .in1(conv_y01), .in2(conv_y10), .in3(conv_y11), .out(pool_y00)
    );

    maxpool_2x2 pool01 (
        .in0(conv_y02), .in1(conv_y03), .in2(conv_y12), .in3(conv_y13), .out(pool_y01)
    );

    maxpool_2x2 pool10 (
        .in0(conv_y20), .in1(conv_y21), .in2(conv_y30), .in3(conv_y31), .out(pool_y10)
    );

    maxpool_2x2 pool11 (
        .in0(conv_y22), .in1(conv_y23), .in2(conv_y32), .in3(conv_y33), .out(pool_y11)
    );

    assign gap_sum =
        {{24{pool_y00[7]}}, pool_y00} + {{24{pool_y01[7]}}, pool_y01} +
        {{24{pool_y10[7]}}, pool_y10} + {{24{pool_y11[7]}}, pool_y11};
    assign gap_avg = gap_sum[9:2];

    assign fc0_product = {{24{gap_avg[7]}}, gap_avg} * {{24{fc0_weight[7]}}, fc0_weight};
    assign fc1_product = {{24{gap_avg[7]}}, gap_avg} * {{24{fc1_weight[7]}}, fc1_weight};
    assign logit0 = (fc0_product >>> 7) + fc0_bias;
    assign logit1 = (fc1_product >>> 7) + fc1_bias;
    assign class_id = (logit1 > logit0);
endmodule
