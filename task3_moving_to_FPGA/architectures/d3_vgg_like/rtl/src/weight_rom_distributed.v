// Distributed-ROM alternative used when BRAM capacity is the limiting resource.
// 当 BRAM 容量成为瓶颈时使用的分布式 ROM，以 LUT 容量换取 BRAM 容量。
module weight_rom_distributed #(
    parameter integer DATA_WIDTH=8,
    parameter integer ADDR_WIDTH=17,
    parameter integer DEPTH=36864,
    parameter INIT_FILE="conv4_weight.mem"
)(
    input clk, input [ADDR_WIDTH-1:0] addr, output reg signed [DATA_WIDTH-1:0] data
);
    (* rom_style = "distributed" *) reg signed [DATA_WIDTH-1:0] rom [0:DEPTH-1];
    integer index;
    localparam integer INDEX_WIDTH = (DEPTH <= 1) ? 1 : $clog2(DEPTH);
    /* verilator lint_off WIDTHTRUNC */
    localparam [ADDR_WIDTH-1:0] DEPTH_LIMIT = DEPTH;
    /* verilator lint_on WIDTHTRUNC */
    wire [INDEX_WIDTH-1:0] rom_addr = addr[INDEX_WIDTH-1:0];
    initial begin
        for(index=0;index<DEPTH;index=index+1) rom[index]={DATA_WIDTH{1'b0}};
        if(INIT_FILE!="") $readmemh(INIT_FILE,rom);
    end
    always @(posedge clk) begin
        if(addr<DEPTH_LIMIT) data<=rom[rom_addr]; else data<={DATA_WIDTH{1'b0}};
    end
endmodule
