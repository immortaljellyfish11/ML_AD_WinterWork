// linear_128x10_sync
// 作用: 同步读 feature buffer 版本的 128->10 全连接层。
// 流程: 给出 feature_addr/weight_addr -> 等 1 拍 -> 累加 -> 输出 logit。
module linear_128x10_sync #(
    parameter integer FRAC_BITS = 7,
    parameter integer FEATURE_ADDR_WIDTH = 32,
    parameter integer WEIGHT_ADDR_WIDTH = 32,
    parameter integer BIAS_ADDR_WIDTH = 4
) (
    input clk,
    input rst_n,
    input start,
    input signed [7:0] feature_data,
    input signed [7:0] weight_data,
    input signed [31:0] bias_data,
    output reg [FEATURE_ADDR_WIDTH-1:0] feature_addr,
    output reg [WEIGHT_ADDR_WIDTH-1:0] weight_addr,
    output reg [BIAS_ADDR_WIDTH-1:0] bias_addr,
    output reg [3:0] logit_index,
    output reg signed [31:0] logit_data,
    output reg logit_valid,
    output reg busy,
    output reg done
);
    localparam S_IDLE = 3'd0;
    localparam S_ADDR = 3'd1;
    localparam S_ACC  = 3'd2;
    localparam S_EMIT = 3'd3;
    localparam S_DONE = 3'd4;

    reg [2:0] state;
    reg [3:0] class_idx;
    reg [6:0] feature_idx;
    reg signed [31:0] acc;
    /* verilator lint_off UNUSEDSIGNAL */
    reg [31:0] feature_addr_full;
    reg [31:0] weight_addr_full;
    /* verilator lint_on UNUSEDSIGNAL */
    wire signed [31:0] product32;
    wire last_feature;
    wire last_class;

    assign product32 = {{24{feature_data[7]}}, feature_data} * {{24{weight_data[7]}}, weight_data};
    assign last_feature = (feature_idx == 7'd127);
    assign last_class = (class_idx == 4'd9);

    always @(*) begin
        feature_addr_full = {25'd0, feature_idx};
        weight_addr_full = {28'd0, class_idx} * 32'd128 + {25'd0, feature_idx};
        feature_addr = feature_addr_full[FEATURE_ADDR_WIDTH-1:0];
        weight_addr = weight_addr_full[WEIGHT_ADDR_WIDTH-1:0];
        bias_addr = class_idx;
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= S_IDLE;
            class_idx <= 4'd0;
            feature_idx <= 7'd0;
            acc <= 32'sd0;
            logit_index <= 4'd0;
            logit_data <= 32'sd0;
            logit_valid <= 1'b0;
            busy <= 1'b0;
            done <= 1'b0;
        end else begin
            logit_valid <= 1'b0;
            done <= 1'b0;

            case (state)
                S_IDLE: begin
                    busy <= 1'b0;
                    if (start) begin
                        class_idx <= 4'd0;
                        feature_idx <= 7'd0;
                        acc <= 32'sd0;
                        busy <= 1'b1;
                        state <= S_ADDR;
                    end
                end

                S_ADDR: begin
                    state <= S_ACC;
                end

                S_ACC: begin
                    acc <= acc + product32;
                    if (last_feature) begin
                        state <= S_EMIT;
                    end else begin
                        feature_idx <= feature_idx + 7'd1;
                        state <= S_ADDR;
                    end
                end

                S_EMIT: begin
                    logit_index <= class_idx;
                    logit_data <= (acc >>> FRAC_BITS) + bias_data;
                    logit_valid <= 1'b1;
                    if (last_class) begin
                        state <= S_DONE;
                    end else begin
                        class_idx <= class_idx + 4'd1;
                        feature_idx <= 7'd0;
                        acc <= 32'sd0;
                        state <= S_ADDR;
                    end
                end

                S_DONE: begin
                    busy <= 1'b0;
                    done <= 1'b1;
                    state <= S_IDLE;
                end

                default: state <= S_IDLE;
            endcase
        end
    end
endmodule
