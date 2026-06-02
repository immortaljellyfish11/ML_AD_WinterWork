// bias_rom
// 作用: 保存每个输出通道或每个分类输出对应的 bias。
// bias 通常使用更宽的 INT32，避免量化累加后溢出。
module bias_rom #(
    parameter integer DATA_WIDTH = 32,
    parameter integer ADDR_WIDTH = 10,
    parameter integer DEPTH = 1024,
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
