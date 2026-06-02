// pool_layer_2x2_engine_sync
// 作用: 对一个 CHW 布局的 feature map 做 2x2 maxpool，stride=2。
// 同步读版本: 每个输出像素依次读取 4 个输入地址，比较最大值后写出。
module pool_layer_2x2_engine_sync #(
    parameter integer IN_HEIGHT = 32,
    parameter integer IN_WIDTH = 32,
    parameter integer CHANNELS = 32,
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
    localparam integer OUT_HEIGHT = IN_HEIGHT / 2;
    localparam integer OUT_WIDTH = IN_WIDTH / 2;
    localparam S_IDLE = 3'd0;
    localparam S_ADDR = 3'd1;
    localparam S_READ = 3'd2;
    localparam S_WRITE = 3'd3;
    localparam S_DONE = 3'd4;

    reg [2:0] state;
    reg [15:0] ch;
    reg [15:0] row;
    reg [15:0] col;
    reg [1:0] tap;
    reg signed [7:0] current_max;
    /* verilator lint_off UNUSEDSIGNAL */
    reg [31:0] in_addr_full;
    reg [31:0] out_addr_full;
    /* verilator lint_on UNUSEDSIGNAL */
    wire last_pixel;
    wire [15:0] src_row;
    wire [15:0] src_col;

    assign last_pixel = ({16'd0, ch} == CHANNELS - 1) &&
        ({16'd0, row} == OUT_HEIGHT - 1) &&
        ({16'd0, col} == OUT_WIDTH - 1);
    assign src_row = (row << 1) + {15'd0, tap[1]};
    assign src_col = (col << 1) + {15'd0, tap[0]};

    always @(*) begin
        in_addr_full = {16'd0, ch} * IN_HEIGHT * IN_WIDTH + {16'd0, src_row} * IN_WIDTH + {16'd0, src_col};
        out_addr_full = {16'd0, ch} * OUT_HEIGHT * OUT_WIDTH + {16'd0, row} * OUT_WIDTH + {16'd0, col};
        in_addr = in_addr_full[ADDR_WIDTH-1:0];
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= S_IDLE;
            ch <= 16'd0;
            row <= 16'd0;
            col <= 16'd0;
            tap <= 2'd0;
            current_max <= -8'sd128;
            out_data <= 8'sd0;
            out_addr <= {ADDR_WIDTH{1'b0}};
            out_we <= 1'b0;
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
                        row <= 16'd0;
                        col <= 16'd0;
                        tap <= 2'd0;
                        current_max <= -8'sd128;
                        busy <= 1'b1;
                        state <= S_ADDR;
                    end
                end

                S_ADDR: begin
                    state <= S_READ;
                end

                S_READ: begin
                    if (tap == 2'd0 || in_data > current_max) begin
                        current_max <= in_data;
                    end

                    if (tap == 2'd3) begin
                        state <= S_WRITE;
                    end else begin
                        tap <= tap + 2'd1;
                        state <= S_ADDR;
                    end
                end

                S_WRITE: begin
                    // out_we 拉高时，out_addr 必须对应当前 row/col/ch。
                    // 下面会更新 row/col/ch，所以地址要先寄存下来。
                    out_addr <= out_addr_full[ADDR_WIDTH-1:0];
                    out_data <= current_max;
                    out_we <= 1'b1;
                    tap <= 2'd0;
                    current_max <= -8'sd128;

                    if (last_pixel) begin
                        state <= S_DONE;
                    end else begin
                        if ({16'd0, col} != OUT_WIDTH - 1) begin
                            col <= col + 16'd1;
                        end else begin
                            col <= 16'd0;
                            if ({16'd0, row} != OUT_HEIGHT - 1) begin
                                row <= row + 16'd1;
                            end else begin
                                row <= 16'd0;
                                ch <= ch + 16'd1;
                            end
                        end
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
