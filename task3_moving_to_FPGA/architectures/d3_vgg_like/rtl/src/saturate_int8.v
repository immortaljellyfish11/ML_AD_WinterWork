module saturate_int8 (
    input  signed [31:0] value_in,
    output signed [7:0]  value_out
);
    assign value_out =
        (value_in > 32'sd127)  ? 8'sd127 :
        (value_in < -32'sd128) ? -8'sd128 :
                                  value_in[7:0];
endmodule
