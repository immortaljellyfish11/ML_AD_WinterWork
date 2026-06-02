// conv_controller
// 作用: 遍历一个卷积层的所有 output channel / row / col。
// 它不直接做乘法，而是反复启动 multi_channel_conv3x3，并在每个输出像素完成时给出 out_valid。
// 下一步可把 out_valid 接到后处理模块: bias -> requantize -> ReLU -> saturate -> feature_buffer 写回。
module conv_controller #(
    parameter integer OUT_HEIGHT = 32,
    parameter integer OUT_WIDTH = 32,
    parameter integer OUT_CHANNELS = 32,
    parameter integer OUT_ADDR_WIDTH = 32
) (
    input clk,
    input rst_n,
    input start,
    input engine_done,
    output reg engine_start,
    output reg [15:0] out_row,
    output reg [15:0] out_col,
    output reg [15:0] out_channel,
    output reg [OUT_ADDR_WIDTH-1:0] out_addr,
    output reg out_valid,
    output reg busy,
    output reg done
);
    localparam S_IDLE  = 2'd0;
    localparam S_START = 2'd1;
    localparam S_WAIT  = 2'd2;
    localparam S_DONE  = 2'd3;

    reg [1:0] state;
    /* verilator lint_off UNUSEDSIGNAL */
    reg [31:0] out_addr_full;
    /* verilator lint_on UNUSEDSIGNAL */
    wire last_pixel;
    localparam [31:0] LAST_OUT_HEIGHT = OUT_HEIGHT - 1;
    localparam [31:0] LAST_OUT_WIDTH = OUT_WIDTH - 1;
    localparam [31:0] LAST_OUT_CHANNEL = OUT_CHANNELS - 1;

    assign last_pixel = ({16'd0, out_channel} == LAST_OUT_CHANNEL) &&
        ({16'd0, out_row} == LAST_OUT_HEIGHT) &&
        ({16'd0, out_col} == LAST_OUT_WIDTH);

    always @(*) begin
        out_addr_full = {16'd0, out_channel} * OUT_HEIGHT * OUT_WIDTH + {16'd0, out_row} * OUT_WIDTH + {16'd0, out_col};
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= S_IDLE;
            engine_start <= 1'b0;
            out_row <= 16'd0;
            out_col <= 16'd0;
            out_channel <= 16'd0;
            out_addr <= {OUT_ADDR_WIDTH{1'b0}};
            out_valid <= 1'b0;
            busy <= 1'b0;
            done <= 1'b0;
        end else begin
            engine_start <= 1'b0;
            out_valid <= 1'b0;
            done <= 1'b0;

            case (state)
                S_IDLE: begin
                    busy <= 1'b0;
                    if (start) begin
                        out_row <= 16'd0;
                        out_col <= 16'd0;
                        out_channel <= 16'd0;
                        busy <= 1'b1;
                        state <= S_START;
                    end
                end

                S_START: begin
                    engine_start <= 1'b1;
                    state <= S_WAIT;
                end

                S_WAIT: begin
                    if (engine_done) begin
                        // out_valid 拉高的这一拍必须锁存“当前像素”的地址。
                        // 如果 out_addr 继续由 out_row/out_col 组合生成，下面计数器递增后会变成下一个像素地址。
                        out_addr <= out_addr_full[OUT_ADDR_WIDTH-1:0];
                        out_valid <= 1'b1;
                        if (last_pixel) begin
                            state <= S_DONE;
                        end else begin
                            if ({16'd0, out_col} != LAST_OUT_WIDTH) begin
                                out_col <= out_col + 16'd1;
                            end else begin
                                out_col <= 16'd0;
                                if ({16'd0, out_row} != LAST_OUT_HEIGHT) begin
                                    out_row <= out_row + 16'd1;
                                end else begin
                                    out_row <= 16'd0;
                                    out_channel <= out_channel + 16'd1;
                                end
                            end
                            state <= S_START;
                        end
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
