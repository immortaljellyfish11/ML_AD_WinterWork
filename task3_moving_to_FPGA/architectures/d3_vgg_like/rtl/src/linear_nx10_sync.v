// Configurable synchronous N-to-10 linear layer.
// 可配置的同步 N 输入、10 类全连接层；特征和权重均采用同步 ROM 读时序。
module linear_nx10_sync #(
    parameter integer IN_FEATURES = 128,
    parameter integer INDEX_WIDTH = 7,
    parameter integer FEATURE_ADDR_WIDTH = 16,
    parameter integer WEIGHT_ADDR_WIDTH = 16,
    parameter integer BIAS_ADDR_WIDTH = 4
) (
    input clk, input rst_n, input start,
    input signed [7:0] feature_data, input signed [7:0] weight_data,
    input signed [31:0] bias_data,
    output reg [FEATURE_ADDR_WIDTH-1:0] feature_addr,
    output reg [WEIGHT_ADDR_WIDTH-1:0] weight_addr,
    output reg [BIAS_ADDR_WIDTH-1:0] bias_addr,
    output reg [3:0] logit_index, output reg signed [31:0] logit_data,
    output reg logit_valid, output reg busy, output reg done
);
    localparam S_IDLE=3'd0, S_ADDR=3'd1, S_ACC=3'd2, S_EMIT=3'd3, S_DONE=3'd4;
    reg [2:0] state; reg [3:0] class_idx; reg [INDEX_WIDTH-1:0] feature_idx; reg signed [31:0] acc;
    /* verilator lint_off UNUSEDSIGNAL */ reg [31:0] weight_addr_full; /* verilator lint_on UNUSEDSIGNAL */
    wire signed [31:0] product32 = $signed(feature_data) * $signed(weight_data);
    wire last_feature = &feature_idx; wire last_class = (class_idx == 4'd9);
    always @(*) begin feature_addr={{(FEATURE_ADDR_WIDTH-INDEX_WIDTH){1'b0}},feature_idx}; weight_addr_full=class_idx*IN_FEATURES+{{(32-INDEX_WIDTH){1'b0}},feature_idx}; weight_addr=weight_addr_full[WEIGHT_ADDR_WIDTH-1:0]; bias_addr=class_idx; end
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin state<=S_IDLE; class_idx<=0; feature_idx<=0; acc<=0; logit_index<=0; logit_data<=0; logit_valid<=0; busy<=0; done<=0; end
        else begin logit_valid<=0; done<=0; case(state)
            S_IDLE: begin busy<=0; if(start) begin class_idx<=0; feature_idx<=0; acc<=0; busy<=1; state<=S_ADDR; end end
            S_ADDR: state<=S_ACC;
            S_ACC: begin acc<=acc+product32; if(last_feature) state<=S_EMIT; else begin feature_idx<=feature_idx+1'b1; state<=S_ADDR; end end
            // Bias and accumulator share one scale / bias 与累加器共用同一个 scale。
            S_EMIT: begin logit_index<=class_idx; logit_data<=acc+bias_data; logit_valid<=1; if(last_class) state<=S_DONE; else begin class_idx<=class_idx+1'b1; feature_idx<=0; acc<=0; state<=S_ADDR; end end
            S_DONE: begin busy<=0; done<=1; state<=S_IDLE; end default: state<=S_IDLE;
        endcase end
    end
endmodule
