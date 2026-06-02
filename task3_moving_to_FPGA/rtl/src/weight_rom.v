// weight_rom
// 作用: 保存训练后量化好的卷积/全连接权重。
// 地址由 addr_gen 或 linear_engine 产生，数据只读。
// INIT_FILE 可指向由 Python 导出的 .hex/.mem 权重文件。
module weight_rom #(
    parameter integer DATA_WIDTH = 8,
    parameter integer ADDR_WIDTH = 18,
    parameter integer DEPTH = 262144,
    parameter INIT_FILE = ""
) (
    input [ADDR_WIDTH-1:0] addr,
    output signed [DATA_WIDTH-1:0] data
);
    reg signed [DATA_WIDTH-1:0] rom [0:DEPTH-1];
    localparam integer INDEX_WIDTH = (DEPTH <= 1) ? 1 : $clog2(DEPTH);
    wire [INDEX_WIDTH-1:0] rom_addr;

    assign rom_addr = addr[INDEX_WIDTH-1:0];

    generate
        if (ADDR_WIDTH > INDEX_WIDTH) begin : gen_unused_addr_bits
            /* verilator lint_off UNUSEDSIGNAL */
            wire unused_addr_bits = |addr[ADDR_WIDTH-1:INDEX_WIDTH];
            /* verilator lint_on UNUSEDSIGNAL */
        end
    endgenerate

    initial begin
        if (INIT_FILE != "") begin
            $readmemh(INIT_FILE, rom);
        end
    end

    assign data = rom[rom_addr];
endmodule
