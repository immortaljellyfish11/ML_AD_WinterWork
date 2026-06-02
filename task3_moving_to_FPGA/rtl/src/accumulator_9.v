module accumulator_9 (
    input  signed [15:0] in0,
    input  signed [15:0] in1,
    input  signed [15:0] in2,
    input  signed [15:0] in3,
    input  signed [15:0] in4,
    input  signed [15:0] in5,
    input  signed [15:0] in6,
    input  signed [15:0] in7,
    input  signed [15:0] in8,
    output signed [31:0] sum
);
    assign sum =
        {{16{in0[15]}}, in0} + {{16{in1[15]}}, in1} + {{16{in2[15]}}, in2} +
        {{16{in3[15]}}, in3} + {{16{in4[15]}}, in4} + {{16{in5[15]}}, in5} +
        {{16{in6[15]}}, in6} + {{16{in7[15]}}, in7} + {{16{in8[15]}}, in8};
endmodule
