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
    parameter integer REQUANT_MULTIPLIER = 1,
    parameter integer REQUANT_SHIFT = FRAC_BITS,
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
    wire controller_valid_unused;
    wire controller_busy;
    wire controller_done;
    wire signed [31:0] raw_sum;
    reg signed [31:0] biased_sum_reg;
    wire signed [24:0] biased_sum_narrow;
    wire signed [42:0] product;
    reg signed [42:0] product_reg;
    wire signed [31:0] quantized;
    wire signed [31:0] activated;
    wire signed [7:0] saturated;
    wire [15:0] out_row;
    wire [15:0] out_col;
    wire [15:0] out_channel;
    wire [ADDR_WIDTH-1:0] controller_out_addr;
    wire engine_busy_unused;
    reg postprocess_pending;
    reg product_valid;
    reg postprocess_valid;
    reg [ADDR_WIDTH-1:0] product_addr_reg;
    reg signed [7:0] out_data_reg;
    reg [ADDR_WIDTH-1:0] out_addr_reg;
    reg controller_done_d1;
    reg controller_done_d2;
    localparam signed [17:0] REQUANT_MULTIPLIER_SIGNED = REQUANT_MULTIPLIER[17:0];

    /* verilator lint_off UNUSEDSIGNAL */
    wire unused_biased_sum_bits = |(biased_sum_reg[31:25] ^ {7{biased_sum_reg[24]}});
    /* verilator lint_on UNUSEDSIGNAL */

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
        .out_addr(controller_out_addr),
        .out_valid(controller_valid_unused),
        .busy(controller_busy),
        .done(controller_done)
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

    assign biased_sum_narrow = biased_sum_reg[24:0];
    assign product = $signed(biased_sum_narrow) * $signed(REQUANT_MULTIPLIER_SIGNED);

    requantize_round_shift #(
        .SHIFT(REQUANT_SHIFT)
    ) quant (
        .product_in(product_reg),
        .value_out(quantized)
    );

    relu #(.WIDTH(32)) activation (
        .value_in(quantized),
        .value_out(activated)
    );

    saturate_int8 saturate (
        .value_in(activated),
        .value_out(saturated)
    );

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            biased_sum_reg <= 32'sd0;
            postprocess_pending <= 1'b0;
            product_reg <= 43'sd0;
            product_valid <= 1'b0;
            product_addr_reg <= {ADDR_WIDTH{1'b0}};
            postprocess_valid <= 1'b0;
            out_data_reg <= 8'sd0;
            out_addr_reg <= {ADDR_WIDTH{1'b0}};
            controller_done_d1 <= 1'b0;
            controller_done_d2 <= 1'b0;
        end else begin
            // Capture this channel's bias before the controller advances.
            postprocess_pending <= engine_done;
            if (engine_done) begin
                biased_sum_reg <= raw_sum + bias_data;
            end

            // Register the DSP product separately from rounding and saturation.
            product_valid <= postprocess_pending;
            if (postprocess_pending) begin
                product_reg <= product;
                product_addr_reg <= controller_out_addr;
            end

            postprocess_valid <= product_valid;
            if (product_valid) begin
                out_data_reg <= saturated;
                out_addr_reg <= product_addr_reg;
            end

            // Keep layer done behind the final feature-buffer write.
            controller_done_d1 <= controller_done;
            controller_done_d2 <= controller_done_d1;
        end
    end

    assign out_we = postprocess_valid;
    assign out_addr = out_addr_reg;
    assign out_data = out_data_reg;
    assign busy = controller_busy | postprocess_pending | product_valid | postprocess_valid;
    assign done = controller_done_d2;
endmodule
