// cnn_top_fsm
// 作用: 完整 CNN 推理的顶层状态机。
// 它负责按顺序启动 Conv1/Pool1/Conv2/Pool2/Conv3/GAP/Linear/Argmax。
// 注意: 这里是控制骨架，具体每层的数据搬运由 conv_controller、buffer、ROM 和算子模块完成。
module cnn_top_fsm (
    input clk,
    input rst_n,
    input start,
    input conv1_done,
    input pool1_done,
    input conv2_done,
    input pool2_done,
    input conv3_done,
    input gap_done,
    input linear_done,
    output reg conv1_start,
    output reg pool1_start,
    output reg conv2_start,
    output reg pool2_start,
    output reg conv3_start,
    output reg gap_start,
    output reg linear_start,
    output reg argmax_valid,
    output reg busy,
    output reg done,
    output reg [3:0] state_dbg
);
    localparam S_IDLE   = 4'd0;
    localparam S_CONV1  = 4'd1;
    localparam S_POOL1  = 4'd2;
    localparam S_CONV2  = 4'd3;
    localparam S_POOL2  = 4'd4;
    localparam S_CONV3  = 4'd5;
    localparam S_GAP    = 4'd6;
    localparam S_LINEAR = 4'd7;
    localparam S_ARGMAX = 4'd8;
    localparam S_DONE   = 4'd9;

    reg [3:0] state;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= S_IDLE;
            conv1_start <= 1'b0;
            pool1_start <= 1'b0;
            conv2_start <= 1'b0;
            pool2_start <= 1'b0;
            conv3_start <= 1'b0;
            gap_start <= 1'b0;
            linear_start <= 1'b0;
            argmax_valid <= 1'b0;
            busy <= 1'b0;
            done <= 1'b0;
            state_dbg <= S_IDLE;
        end else begin
            conv1_start <= 1'b0;
            pool1_start <= 1'b0;
            conv2_start <= 1'b0;
            pool2_start <= 1'b0;
            conv3_start <= 1'b0;
            gap_start <= 1'b0;
            linear_start <= 1'b0;
            argmax_valid <= 1'b0;
            done <= 1'b0;
            state_dbg <= state;

            case (state)
                S_IDLE: begin
                    busy <= 1'b0;
                    if (start) begin
                        busy <= 1'b1;
                        conv1_start <= 1'b1;
                        state <= S_CONV1;
                    end
                end

                S_CONV1: if (conv1_done) begin pool1_start <= 1'b1; state <= S_POOL1; end
                S_POOL1: if (pool1_done) begin conv2_start <= 1'b1; state <= S_CONV2; end
                S_CONV2: if (conv2_done) begin pool2_start <= 1'b1; state <= S_POOL2; end
                S_POOL2: if (pool2_done) begin conv3_start <= 1'b1; state <= S_CONV3; end
                S_CONV3: if (conv3_done) begin gap_start <= 1'b1; state <= S_GAP; end
                S_GAP: if (gap_done) begin linear_start <= 1'b1; state <= S_LINEAR; end
                S_LINEAR: if (linear_done) begin argmax_valid <= 1'b1; state <= S_ARGMAX; end

                S_ARGMAX: begin
                    state <= S_DONE;
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
