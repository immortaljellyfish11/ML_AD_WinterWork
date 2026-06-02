// global_avg_pool_layer_engine_sync
// 作用: 对每个通道做全局平均池化，把 CxHxW 压缩成 C 个数。
// 输出写回地址为 channel index，即 0..CHANNELS-1。
module global_avg_pool_layer_engine_sync #(
    parameter integer IN_HEIGHT = 8,
    parameter integer IN_WIDTH = 8,
    parameter integer CHANNELS = 128,
    parameter integer ADDR_WIDTH = 32
) (
    input clk,
    input rst_n,
    input start,
    input signed [7:0] in_data,
    output reg [ADDR_WIDTH-1:0] in_addr,
    output reg out_we,
    output reg [ADDR_WIDTH-1:0] out_addr,
    output reg signed [7:0] out_data,
    output reg busy,
    output reg done
);
    localparam integer PIXELS = IN_HEIGHT * IN_WIDTH;
    localparam integer SHIFT_BITS = 6;
    localparam S_IDLE = 3'd0;
    localparam S_ADDR = 3'd1;
    localparam S_ACC = 3'd2;
    localparam S_WRITE = 3'd3;
    localparam S_DONE = 3'd4;

    reg [2:0] state;
    reg [15:0] ch;
    reg [15:0] idx;
    reg signed [31:0] sum;
    /* verilator lint_off UNUSEDSIGNAL */
    reg [31:0] in_addr_full;
    reg [31:0] out_addr_full;
    /* verilator lint_on UNUSEDSIGNAL */
    wire last_channel;
    wire last_pixel;

    assign last_channel = ({16'd0, ch} == CHANNELS - 1);
    assign last_pixel = ({16'd0, idx} == PIXELS - 1);

    always @(*) begin
        in_addr_full = {16'd0, ch} * PIXELS + {16'd0, idx};
        out_addr_full = {16'd0, ch};
        in_addr = in_addr_full[ADDR_WIDTH-1:0];
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= S_IDLE;
            ch <= 16'd0;
            idx <= 16'd0;
            sum <= 32'sd0;
            out_we <= 1'b0;
            out_addr <= {ADDR_WIDTH{1'b0}};
            out_data <= 8'sd0;
            busy <= 1'b0;
            done <= 1'b0;
        end else begin
            out_we <= 1'b0;
            done <= 1'b0;

            case (state)
                S_IDLE: begin
                    busy <= 1'b0;
                    if (start) begin
                        ch <= 16'd0;
                        idx <= 16'd0;
                        sum <= 32'sd0;
                        busy <= 1'b1;
                        state <= S_ADDR;
                    end
                end

                S_ADDR: begin
                    state <= S_ACC;
                end

                S_ACC: begin
                    sum <= sum + {{24{in_data[7]}}, in_data};
                    if (last_pixel) begin
                        state <= S_WRITE;
                    end else begin
                        idx <= idx + 16'd1;
                        state <= S_ADDR;
                    end
                end

                S_WRITE: begin
                    // out_we 拉高时锁存当前 channel 的输出地址。
                    // 否则 ch 在同一拍递增后，组合 out_addr 会提前跳到下一个 channel。
                    out_addr <= out_addr_full[ADDR_WIDTH-1:0];
                    out_data <= sum[SHIFT_BITS+7:SHIFT_BITS];
                    out_we <= 1'b1;
                    idx <= 16'd0;
                    sum <= 32'sd0;

                    if (last_channel) begin
                        state <= S_DONE;
                    end else begin
                        ch <= ch + 16'd1;
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
