module relu #(
    parameter integer WIDTH = 32
) (
    input  signed [WIDTH-1:0] value_in,
    output signed [WIDTH-1:0] value_out
);
    assign value_out = value_in[WIDTH-1] ? {WIDTH{1'b0}} : value_in;
endmodule
