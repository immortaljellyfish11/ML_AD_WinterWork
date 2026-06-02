// edge_cnn_top
// 作用: CIFAR-10 边缘部署 CNN 的顶层集成入口。
//
// 本版本采用用户指定方案:
// 1B: 输入图像由外部逐像素写入 feature_buffer_a。
// 2A: 训练好的权重/bias 通过 .mem/.hex 固化到 ROM。
// 3B: feature buffer 使用同步读 BRAM 风格。
//
// 数据流:
// input write -> buffer_a
// Conv1: buffer_a -> buffer_b
// Pool1: buffer_b -> buffer_a
// Conv2: buffer_a -> buffer_b
// Pool2: buffer_b -> buffer_a
// Conv3: buffer_a -> buffer_b
// GAP:   buffer_b -> buffer_a[0..127]
// Linear: buffer_a[0..127] -> logits
// Argmax: logits -> class_id
module edge_cnn_top #(
    parameter integer ADDR_WIDTH = 17,
    parameter integer FEATURE_DEPTH = 131072,
    parameter integer FRAC_BITS = 7,
    parameter CONV1_WEIGHT_FILE = "sim/data/fpga_params/q07/conv1_weight.mem", //卷积网络权重参数路径
    parameter CONV2_WEIGHT_FILE = "sim/data/fpga_params/q07/conv2_weight.mem",
    parameter CONV3_WEIGHT_FILE = "sim/data/fpga_params/q07/conv3_weight.mem",
    parameter LINEAR_WEIGHT_FILE = "sim/data/fpga_params/q07/linear_weight.mem", //线性层权重参数路径
    parameter CONV1_BIAS_FILE = "sim/data/fpga_params/q07/conv1_bias.mem",//卷积网络偏置参数路径
    parameter CONV2_BIAS_FILE = "sim/data/fpga_params/q07/conv2_bias.mem",
    parameter CONV3_BIAS_FILE = "sim/data/fpga_params/q07/conv3_bias.mem",
    parameter LINEAR_BIAS_FILE = "sim/data/fpga_params/q07/linear_bias.mem"
) (
    input clk,
    input rst_n,
    input start,

    // 外部输入写口: 在 start 前，把 3x32x32 的 INT8 输入按 CHW 顺序写入 buffer_a。
    input input_we,
    input [ADDR_WIDTH-1:0] input_addr,
    input signed [7:0] input_data,

    output conv1_start,
    output pool1_start,
    output conv2_start,
    output pool2_start,
    output conv3_start,
    output gap_start,
    output linear_start,
    output argmax_valid,
    output busy,
    output done,
    output [3:0] class_id,
    output signed [31:0] max_logit,
    output [3:0] state_dbg
);
    localparam S_CONV1  = 4'd1;
    localparam S_POOL1  = 4'd2;
    localparam S_CONV2  = 4'd3;
    localparam S_POOL2  = 4'd4;
    localparam S_CONV3  = 4'd5;
    localparam S_GAP    = 4'd6;
    localparam S_LINEAR = 4'd7;
    localparam integer CONV1_WEIGHT_DEPTH = 32 * 3 * 3 * 3;
    localparam integer CONV2_WEIGHT_DEPTH = 64 * 32 * 3 * 3;
    localparam integer CONV3_WEIGHT_DEPTH = 128 * 64 * 3 * 3;
    localparam integer LINEAR_WEIGHT_DEPTH = 10 * 128;
    localparam integer CONV1_BIAS_DEPTH = 32;
    localparam integer CONV2_BIAS_DEPTH = 64;
    localparam integer CONV3_BIAS_DEPTH = 128;
    localparam integer LINEAR_BIAS_DEPTH = 10;

    wire conv1_done;
    wire pool1_done;
    wire conv2_done;
    wire pool2_done;
    wire conv3_done;
    wire gap_done;
    wire linear_done;

    wire conv1_busy;
    wire pool1_busy;
    wire conv2_busy;
    wire pool2_busy;
    wire conv3_busy;
    wire gap_busy;
    wire linear_busy;
    wire internal_layer_busy;

    wire [ADDR_WIDTH-1:0] buf_a_raddr;
    wire [ADDR_WIDTH-1:0] buf_b_raddr;
    wire [ADDR_WIDTH-1:0] buf_a_waddr;
    wire [ADDR_WIDTH-1:0] buf_b_waddr;
    wire signed [7:0] buf_a_rdata;
    wire signed [7:0] buf_b_rdata;
    wire signed [7:0] buf_a_wdata;
    wire signed [7:0] buf_b_wdata;
    wire buf_a_we;
    wire buf_b_we;

    wire [ADDR_WIDTH-1:0] conv1_feature_addr;
    wire [ADDR_WIDTH-1:0] conv2_feature_addr;
    wire [ADDR_WIDTH-1:0] conv3_feature_addr;
    wire [ADDR_WIDTH-1:0] conv1_weight_addr;
    wire [ADDR_WIDTH-1:0] conv2_weight_addr;
    wire [ADDR_WIDTH-1:0] conv3_weight_addr;
    wire [9:0] conv1_bias_addr;
    wire [9:0] conv2_bias_addr;
    wire [9:0] conv3_bias_addr;
    wire conv1_out_we;
    wire conv2_out_we;
    wire conv3_out_we;
    wire [ADDR_WIDTH-1:0] conv1_out_addr;
    wire [ADDR_WIDTH-1:0] conv2_out_addr;
    wire [ADDR_WIDTH-1:0] conv3_out_addr;
    wire signed [7:0] conv1_out_data;
    wire signed [7:0] conv2_out_data;
    wire signed [7:0] conv3_out_data;

    wire [ADDR_WIDTH-1:0] pool1_in_addr;
    wire [ADDR_WIDTH-1:0] pool2_in_addr;
    wire pool1_out_we;
    wire pool2_out_we;
    wire [ADDR_WIDTH-1:0] pool1_out_addr;
    wire [ADDR_WIDTH-1:0] pool2_out_addr;
    wire signed [7:0] pool1_out_data;
    wire signed [7:0] pool2_out_data;

    wire [ADDR_WIDTH-1:0] gap_in_addr;
    wire gap_out_we;
    wire [ADDR_WIDTH-1:0] gap_out_addr;
    wire signed [7:0] gap_out_data;

    wire [ADDR_WIDTH-1:0] linear_feature_addr;
    wire [ADDR_WIDTH-1:0] linear_weight_addr;
    wire [3:0] linear_bias_addr;
    wire [3:0] linear_logit_index;
    wire signed [31:0] linear_logit_data;
    wire linear_logit_valid;

    wire signed [7:0] conv1_weight_data;
    wire signed [7:0] conv2_weight_data;
    wire signed [7:0] conv3_weight_data;
    wire signed [7:0] linear_weight_data;
    wire signed [31:0] conv1_bias_data;
    wire signed [31:0] conv2_bias_data;
    wire signed [31:0] conv3_bias_data;
    wire signed [31:0] linear_bias_data;

    reg signed [31:0] logit0;
    reg signed [31:0] logit1;
    reg signed [31:0] logit2;
    reg signed [31:0] logit3;
    reg signed [31:0] logit4;
    reg signed [31:0] logit5;
    reg signed [31:0] logit6;
    reg signed [31:0] logit7;
    reg signed [31:0] logit8;
    reg signed [31:0] logit9;

    cnn_top_fsm fsm (
        .clk(clk),
        .rst_n(rst_n),
        .start(start),
        .conv1_done(conv1_done),
        .pool1_done(pool1_done),
        .conv2_done(conv2_done),
        .pool2_done(pool2_done),
        .conv3_done(conv3_done),
        .gap_done(gap_done),
        .linear_done(linear_done),
        .conv1_start(conv1_start),
        .pool1_start(pool1_start),
        .conv2_start(conv2_start),
        .pool2_start(pool2_start),
        .conv3_start(conv3_start),
        .gap_start(gap_start),
        .linear_start(linear_start),
        .argmax_valid(argmax_valid),
        .busy(busy),
        .done(done),
        .state_dbg(state_dbg)
    );

    // buffer_a: 输入图像、Pool1/Pool2 输出、GAP 输出、Linear 输入。
    feature_buffer_sync #(
        .DATA_WIDTH(8),
        .ADDR_WIDTH(ADDR_WIDTH),
        .DEPTH(FEATURE_DEPTH)
    ) buffer_a (
        .clk(clk),
        .we(buf_a_we),
        .waddr(buf_a_waddr),
        .wdata(buf_a_wdata),
        .raddr(buf_a_raddr),
        .rdata(buf_a_rdata)
    );

    // buffer_b: Conv1/Conv2/Conv3 输出，Pool/GAP 输入。
    feature_buffer_sync #(
        .DATA_WIDTH(8),
        .ADDR_WIDTH(ADDR_WIDTH),
        .DEPTH(FEATURE_DEPTH)
    ) buffer_b (
        .clk(clk),
        .we(buf_b_we),
        .waddr(buf_b_waddr),
        .wdata(buf_b_wdata),
        .raddr(buf_b_raddr),
        .rdata(buf_b_rdata)
    );

    assign internal_layer_busy = conv1_busy | pool1_busy | conv2_busy | pool2_busy | conv3_busy | gap_busy | linear_busy;

    assign buf_a_raddr =
        (state_dbg == S_CONV1)  ? conv1_feature_addr :
        (state_dbg == S_CONV2)  ? conv2_feature_addr :
        (state_dbg == S_CONV3)  ? conv3_feature_addr :
        (state_dbg == S_LINEAR) ? linear_feature_addr :
        {ADDR_WIDTH{1'b0}};

    assign buf_b_raddr =
        (state_dbg == S_POOL1) ? pool1_in_addr :
        (state_dbg == S_POOL2) ? pool2_in_addr :
        (state_dbg == S_GAP)   ? gap_in_addr :
        {ADDR_WIDTH{1'b0}};

    assign buf_a_we =
        (!busy && !internal_layer_busy && input_we) || pool1_out_we || pool2_out_we || gap_out_we;
    assign buf_a_waddr =
        (!busy && !internal_layer_busy && input_we) ? input_addr :
        pool1_out_we       ? pool1_out_addr :
        pool2_out_we       ? pool2_out_addr :
                             gap_out_addr;
    assign buf_a_wdata =
        (!busy && !internal_layer_busy && input_we) ? input_data :
        pool1_out_we       ? pool1_out_data :
        pool2_out_we       ? pool2_out_data :
                             gap_out_data;

    assign buf_b_we = conv1_out_we || conv2_out_we || conv3_out_we;
    assign buf_b_waddr =
        conv1_out_we ? conv1_out_addr :
        conv2_out_we ? conv2_out_addr :
                       conv3_out_addr;
    assign buf_b_wdata =
        conv1_out_we ? conv1_out_data :
        conv2_out_we ? conv2_out_data :
                       conv3_out_data;

    weight_rom #(.DATA_WIDTH(8), .ADDR_WIDTH(ADDR_WIDTH), .DEPTH(CONV1_WEIGHT_DEPTH), .INIT_FILE(CONV1_WEIGHT_FILE))
        conv1_weight_rom (.addr(conv1_weight_addr), .data(conv1_weight_data));
    weight_rom #(.DATA_WIDTH(8), .ADDR_WIDTH(ADDR_WIDTH), .DEPTH(CONV2_WEIGHT_DEPTH), .INIT_FILE(CONV2_WEIGHT_FILE))
        conv2_weight_rom (.addr(conv2_weight_addr), .data(conv2_weight_data));
    weight_rom #(.DATA_WIDTH(8), .ADDR_WIDTH(ADDR_WIDTH), .DEPTH(CONV3_WEIGHT_DEPTH), .INIT_FILE(CONV3_WEIGHT_FILE))
        conv3_weight_rom (.addr(conv3_weight_addr), .data(conv3_weight_data));
    weight_rom #(.DATA_WIDTH(8), .ADDR_WIDTH(ADDR_WIDTH), .DEPTH(LINEAR_WEIGHT_DEPTH), .INIT_FILE(LINEAR_WEIGHT_FILE))
        linear_weight_rom (.addr(linear_weight_addr), .data(linear_weight_data));

    bias_rom #(.DATA_WIDTH(32), .ADDR_WIDTH(10), .DEPTH(CONV1_BIAS_DEPTH), .INIT_FILE(CONV1_BIAS_FILE))
        conv1_bias_rom (.addr(conv1_bias_addr), .data(conv1_bias_data));
    bias_rom #(.DATA_WIDTH(32), .ADDR_WIDTH(10), .DEPTH(CONV2_BIAS_DEPTH), .INIT_FILE(CONV2_BIAS_FILE))
        conv2_bias_rom (.addr(conv2_bias_addr), .data(conv2_bias_data));
    bias_rom #(.DATA_WIDTH(32), .ADDR_WIDTH(10), .DEPTH(CONV3_BIAS_DEPTH), .INIT_FILE(CONV3_BIAS_FILE))
        conv3_bias_rom (.addr(conv3_bias_addr), .data(conv3_bias_data));
    bias_rom #(.DATA_WIDTH(32), .ADDR_WIDTH(4), .DEPTH(LINEAR_BIAS_DEPTH), .INIT_FILE(LINEAR_BIAS_FILE))
        linear_bias_rom (.addr(linear_bias_addr), .data(linear_bias_data));

    conv_layer_engine_sync #(
        .IN_HEIGHT(32), .IN_WIDTH(32), .IN_CHANNELS(3),
        .OUT_HEIGHT(32), .OUT_WIDTH(32), .OUT_CHANNELS(32),
        .FRAC_BITS(FRAC_BITS), .ADDR_WIDTH(ADDR_WIDTH), .BIAS_ADDR_WIDTH(10)
    ) conv1 (
        .clk(clk), .rst_n(rst_n), .start(conv1_start),
        .feature_data(buf_a_rdata), .weight_data(conv1_weight_data), .bias_data(conv1_bias_data),
        .feature_addr(conv1_feature_addr), .weight_addr(conv1_weight_addr), .bias_addr(conv1_bias_addr),
        .out_we(conv1_out_we), .out_addr(conv1_out_addr), .out_data(conv1_out_data),
        .busy(conv1_busy), .done(conv1_done)
    );

    pool_layer_2x2_engine_sync #(.IN_HEIGHT(32), .IN_WIDTH(32), .CHANNELS(32), .ADDR_WIDTH(ADDR_WIDTH)) pool1 (
        .clk(clk), .rst_n(rst_n), .start(pool1_start),
        .in_data(buf_b_rdata), .in_addr(pool1_in_addr),
        .out_we(pool1_out_we), .out_addr(pool1_out_addr), .out_data(pool1_out_data),
        .busy(pool1_busy), .done(pool1_done)
    );

    conv_layer_engine_sync #(
        .IN_HEIGHT(16), .IN_WIDTH(16), .IN_CHANNELS(32),
        .OUT_HEIGHT(16), .OUT_WIDTH(16), .OUT_CHANNELS(64),
        .FRAC_BITS(FRAC_BITS), .ADDR_WIDTH(ADDR_WIDTH), .BIAS_ADDR_WIDTH(10)
    ) conv2 (
        .clk(clk), .rst_n(rst_n), .start(conv2_start),
        .feature_data(buf_a_rdata), .weight_data(conv2_weight_data), .bias_data(conv2_bias_data),
        .feature_addr(conv2_feature_addr), .weight_addr(conv2_weight_addr), .bias_addr(conv2_bias_addr),
        .out_we(conv2_out_we), .out_addr(conv2_out_addr), .out_data(conv2_out_data),
        .busy(conv2_busy), .done(conv2_done)
    );

    pool_layer_2x2_engine_sync #(.IN_HEIGHT(16), .IN_WIDTH(16), .CHANNELS(64), .ADDR_WIDTH(ADDR_WIDTH)) pool2 (
        .clk(clk), .rst_n(rst_n), .start(pool2_start),
        .in_data(buf_b_rdata), .in_addr(pool2_in_addr),
        .out_we(pool2_out_we), .out_addr(pool2_out_addr), .out_data(pool2_out_data),
        .busy(pool2_busy), .done(pool2_done)
    );

    conv_layer_engine_sync #(
        .IN_HEIGHT(8), .IN_WIDTH(8), .IN_CHANNELS(64),
        .OUT_HEIGHT(8), .OUT_WIDTH(8), .OUT_CHANNELS(128),
        .FRAC_BITS(FRAC_BITS), .ADDR_WIDTH(ADDR_WIDTH), .BIAS_ADDR_WIDTH(10)
    ) conv3 (
        .clk(clk), .rst_n(rst_n), .start(conv3_start),
        .feature_data(buf_a_rdata), .weight_data(conv3_weight_data), .bias_data(conv3_bias_data),
        .feature_addr(conv3_feature_addr), .weight_addr(conv3_weight_addr), .bias_addr(conv3_bias_addr),
        .out_we(conv3_out_we), .out_addr(conv3_out_addr), .out_data(conv3_out_data),
        .busy(conv3_busy), .done(conv3_done)
    );

    global_avg_pool_layer_engine_sync #(.IN_HEIGHT(8), .IN_WIDTH(8), .CHANNELS(128), .ADDR_WIDTH(ADDR_WIDTH)) gap (
        .clk(clk), .rst_n(rst_n), .start(gap_start),
        .in_data(buf_b_rdata), .in_addr(gap_in_addr),
        .out_we(gap_out_we), .out_addr(gap_out_addr), .out_data(gap_out_data),
        .busy(gap_busy), .done(gap_done)
    );

    linear_128x10_sync #(
        .FRAC_BITS(FRAC_BITS),
        .FEATURE_ADDR_WIDTH(ADDR_WIDTH),
        .WEIGHT_ADDR_WIDTH(ADDR_WIDTH),
        .BIAS_ADDR_WIDTH(4)
    ) linear (
        .clk(clk), .rst_n(rst_n), .start(linear_start),
        .feature_data(buf_a_rdata), .weight_data(linear_weight_data), .bias_data(linear_bias_data),
        .feature_addr(linear_feature_addr), .weight_addr(linear_weight_addr), .bias_addr(linear_bias_addr),
        .logit_index(linear_logit_index), .logit_data(linear_logit_data), .logit_valid(linear_logit_valid),
        .busy(linear_busy), .done(linear_done)
    );

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            logit0 <= 32'sd0; logit1 <= 32'sd0; logit2 <= 32'sd0; logit3 <= 32'sd0; logit4 <= 32'sd0;
            logit5 <= 32'sd0; logit6 <= 32'sd0; logit7 <= 32'sd0; logit8 <= 32'sd0; logit9 <= 32'sd0;
        end else if (linear_logit_valid) begin
            case (linear_logit_index)
                4'd0: logit0 <= linear_logit_data;
                4'd1: logit1 <= linear_logit_data;
                4'd2: logit2 <= linear_logit_data;
                4'd3: logit3 <= linear_logit_data;
                4'd4: logit4 <= linear_logit_data;
                4'd5: logit5 <= linear_logit_data;
                4'd6: logit6 <= linear_logit_data;
                4'd7: logit7 <= linear_logit_data;
                4'd8: logit8 <= linear_logit_data;
                4'd9: logit9 <= linear_logit_data;
                default: logit0 <= logit0;
            endcase
        end
    end

    argmax_10 argmax (
        .logit0(logit0), .logit1(logit1), .logit2(logit2), .logit3(logit3), .logit4(logit4),
        .logit5(logit5), .logit6(logit6), .logit7(logit7), .logit8(logit8), .logit9(logit9),
        .class_id(class_id), .max_logit(max_logit)
    );
endmodule
