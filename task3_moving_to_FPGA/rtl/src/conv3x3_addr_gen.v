// conv3x3_addr_gen
// 作用: 根据当前输出坐标、输入通道、卷积核坐标生成 feature/weight 地址。
// feature map 布局: CHW, addr = ic * H * W + row * W + col。
// weight 布局: OIHW, addr = oc * IC * 9 + ic * 9 + kr * 3 + kc。
// padding=1 时，越界输入由 valid=0 表示，卷积引擎应把该输入当作 0。
module conv3x3_addr_gen #(
    parameter integer IN_HEIGHT = 32,
    parameter integer IN_WIDTH = 32,
    parameter integer IN_CHANNELS = 3,
    parameter integer ADDR_WIDTH = 32
) (
    input [15:0] out_row,
    input [15:0] out_col,
    input [15:0] in_channel,
    input [15:0] out_channel,
    input [1:0] kernel_row,
    input [1:0] kernel_col,
    output reg valid,
    output reg [ADDR_WIDTH-1:0] feature_addr,
    output reg [ADDR_WIDTH-1:0] weight_addr
);
    integer src_row;
    integer src_col;
    /* verilator lint_off UNUSEDSIGNAL */
    reg [31:0] feature_index;
    reg [31:0] weight_index;
    /* verilator lint_on UNUSEDSIGNAL */

    always @(*) begin
        src_row = {16'd0, out_row} + {30'd0, kernel_row} - 1;
        src_col = {16'd0, out_col} + {30'd0, kernel_col} - 1;

        valid = (src_row >= 0) && (src_row < IN_HEIGHT) && (src_col >= 0) && (src_col < IN_WIDTH);

        if (valid) begin
            feature_index = in_channel * IN_HEIGHT * IN_WIDTH + src_row * IN_WIDTH + src_col;
        end else begin
            feature_index = 0;
        end

        weight_index = {16'd0, out_channel} * IN_CHANNELS * 9 + {16'd0, in_channel} * 9 +
            {30'd0, kernel_row} * 3 + {30'd0, kernel_col};
        feature_addr = feature_index[ADDR_WIDTH-1:0];
        weight_addr = weight_index[ADDR_WIDTH-1:0];
    end
endmodule
