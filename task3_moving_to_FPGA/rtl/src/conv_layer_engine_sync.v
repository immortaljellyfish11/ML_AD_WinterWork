// conv_layer_engine_sync
// 作用: 同步 feature buffer 版本的完整卷积层执行器。
// 输出 out_we/out_addr/out_data，供上层写入 ping-pong feature buffer。
module conv_layer_engine_sync #(
    parameter integer IN_HEIGHT = 32,
    parameter integer IN_WIDTH = 32,
    parameter integer IN_CHANNELS = 3,
    parameter integer OUT_HEIGHT = 32,
    parameter integer OUT_WIDTH = 32,
    parameter integer OUT_CHANNELS = 32,
    parameter integer FRAC_BITS = 7,
    parameter integer ADDR_WIDTH = 32,
    parameter integer BIAS_ADDR_WIDTH = 10
) (
    input clk,
    input rst_n,
    input start,
    input signed [7:0] feature_data,
    input signed [7:0] weight_data,
    input signed [31:0] bias_data,
    output [ADDR_WIDTH-1:0] feature_addr,
    output [ADDR_WIDTH-1:0] weight_addr,
    output [BIAS_ADDR_WIDTH-1:0] bias_addr,
    output out_we,
    output [ADDR_WIDTH-1:0] out_addr,
    output signed [7:0] out_data,
    output busy,
    output done
);
    wire engine_start;
    wire engine_done;
    wire controller_valid;
    wire controller_busy;
    wire signed [31:0] raw_sum;
    wire signed [31:0] biased_sum;
    wire signed [31:0] quantized;
    wire signed [31:0] activated;
    wire [15:0] out_row;
    wire [15:0] out_col;
    wire [15:0] out_channel;
    wire engine_busy_unused;

    conv_controller #(
        .OUT_HEIGHT(OUT_HEIGHT),
        .OUT_WIDTH(OUT_WIDTH),
        .OUT_CHANNELS(OUT_CHANNELS),
        .OUT_ADDR_WIDTH(ADDR_WIDTH)
    ) controller (
        .clk(clk),
        .rst_n(rst_n),
        .start(start),
        .engine_done(engine_done),
        .engine_start(engine_start),
        .out_row(out_row),
        .out_col(out_col),
        .out_channel(out_channel),
        .out_addr(out_addr),
        .out_valid(controller_valid),
        .busy(controller_busy),
        .done(done)
    );

    multi_channel_conv3x3_sync #(
        .IN_HEIGHT(IN_HEIGHT),
        .IN_WIDTH(IN_WIDTH),
        .IN_CHANNELS(IN_CHANNELS),
        .ADDR_WIDTH(ADDR_WIDTH)
    ) engine (
        .clk(clk),
        .rst_n(rst_n),
        .start(engine_start),
        .out_row(out_row),
        .out_col(out_col),
        .out_channel(out_channel),
        .feature_data(feature_data),
        .weight_data(weight_data),
        .feature_addr(feature_addr),
        .weight_addr(weight_addr),
        .raw_sum(raw_sum),
        .busy(engine_busy_unused),
        .done(engine_done)
    );

    assign bias_addr = out_channel[BIAS_ADDR_WIDTH-1:0];
    assign biased_sum = raw_sum + bias_data;

    requantize #(.FRAC_BITS(FRAC_BITS)) quant (
        .value_in(biased_sum),
        .value_out(quantized)
    );

    relu #(.WIDTH(32)) activation (
        .value_in(quantized),
        .value_out(activated)
    );

    saturate_int8 saturate (
        .value_in(activated),
        .value_out(out_data)
    );

    assign out_we = controller_valid;
    assign busy = controller_busy;
endmodule
