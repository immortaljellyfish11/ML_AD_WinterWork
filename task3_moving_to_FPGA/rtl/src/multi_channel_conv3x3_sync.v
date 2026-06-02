// multi_channel_conv3x3_sync
// 作用: 同步读 feature buffer 版本的多通道 3x3 卷积核心。
// 与 multi_channel_conv3x3 的区别:
// - 先输出 feature_addr/weight_addr。
// - 等 1 拍，让 feature_buffer_sync 输出 feature_data。
// - 再累加 feature_data * weight_data。
module multi_channel_conv3x3_sync #(
    parameter integer IN_HEIGHT = 32,
    parameter integer IN_WIDTH = 32,
    parameter integer IN_CHANNELS = 3,
    parameter integer ADDR_WIDTH = 32
) (
    input clk,
    input rst_n,
    input start,
    input [15:0] out_row,
    input [15:0] out_col,
    input [15:0] out_channel,
    input signed [7:0] feature_data,
    input signed [7:0] weight_data,
    output [ADDR_WIDTH-1:0] feature_addr,
    output [ADDR_WIDTH-1:0] weight_addr,
    output reg signed [31:0] raw_sum,
    output reg busy,
    output reg done
);
    localparam S_IDLE = 2'd0;
    localparam S_ADDR = 2'd1;
    localparam S_ACC  = 2'd2;
    localparam S_DONE = 2'd3;

    reg [1:0] state;
    reg [15:0] ic;
    reg [1:0] kr;
    reg [1:0] kc;
    wire valid_pixel;
    wire signed [31:0] product32;
    wire last_item;
    localparam [31:0] LAST_IN_CHANNEL = IN_CHANNELS - 1;

    conv3x3_addr_gen #(
        .IN_HEIGHT(IN_HEIGHT),
        .IN_WIDTH(IN_WIDTH),
        .IN_CHANNELS(IN_CHANNELS),
        .ADDR_WIDTH(ADDR_WIDTH)
    ) addr_gen (
        .out_row(out_row),
        .out_col(out_col),
        .in_channel(ic),
        .out_channel(out_channel),
        .kernel_row(kr),
        .kernel_col(kc),
        .valid(valid_pixel),
        .feature_addr(feature_addr),
        .weight_addr(weight_addr)
    );

    assign product32 = valid_pixel ? ({{24{feature_data[7]}}, feature_data} * {{24{weight_data[7]}}, weight_data}) : 32'sd0;
    assign last_item = ({16'd0, ic} == LAST_IN_CHANNEL) && (kr == 2'd2) && (kc == 2'd2);

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= S_IDLE;
            ic <= 16'd0;
            kr <= 2'd0;
            kc <= 2'd0;
            raw_sum <= 32'sd0;
            busy <= 1'b0;
            done <= 1'b0;
        end else begin
            done <= 1'b0;

            case (state)
                S_IDLE: begin
                    busy <= 1'b0;
                    if (start) begin
                        ic <= 16'd0;
                        kr <= 2'd0;
                        kc <= 2'd0;
                        raw_sum <= 32'sd0;
                        busy <= 1'b1;
                        state <= S_ADDR;
                    end
                end

                S_ADDR: begin
                    state <= S_ACC;
                end

                S_ACC: begin
                    raw_sum <= raw_sum + product32;
                    if (last_item) begin
                        state <= S_DONE;
                    end else begin
                        if (kc != 2'd2) begin
                            kc <= kc + 2'd1;
                        end else begin
                            kc <= 2'd0;
                            if (kr != 2'd2) begin
                                kr <= kr + 2'd1;
                            end else begin
                                kr <= 2'd0;
                                ic <= ic + 16'd1;
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
