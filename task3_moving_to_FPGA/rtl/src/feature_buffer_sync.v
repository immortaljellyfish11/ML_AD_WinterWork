// feature_buffer_sync
// 作用: FPGA 风格的 feature map 缓冲区。
// 写口: 时钟上升沿写入。
// 读口: 同步读，给出 raddr 后，rdata 在下一拍更新。
// 设计原因: Xilinx/Intel FPGA 的 BRAM 通常是同步读，真实部署时应按这种时序设计 controller。
module feature_buffer_sync #(
    parameter integer DATA_WIDTH = 8,
    parameter integer ADDR_WIDTH = 32,
    parameter integer DEPTH = 65536
) (
    input clk,
    input we,
    input [ADDR_WIDTH-1:0] waddr,
    input signed [DATA_WIDTH-1:0] wdata,
    input [ADDR_WIDTH-1:0] raddr,
    output reg signed [DATA_WIDTH-1:0] rdata
);
    reg signed [DATA_WIDTH-1:0] mem [0:DEPTH-1];

    always @(posedge clk) begin
        if (we) begin
            mem[waddr] <= wdata;
        end
        rdata <= mem[raddr];
    end
endmodule
