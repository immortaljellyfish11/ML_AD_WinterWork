`timescale 1ns / 1ps

// XSim testbench for the D5 external-weight descriptor sequencer only.
// It does not model the tiled convolution engines, external memory, or
// numerical MobileNet inference.  layer_done is the accelerator completion
// handshake returned after each descriptor has been consumed.
module tb_d5_mobilenet_05_xsim;
    localparam [2:0] OP_CONV      = 3'd0;
    localparam [2:0] OP_DEPTHWISE = 3'd1;
    localparam [2:0] OP_POINTWISE = 3'd2;
    localparam [2:0] OP_GAP       = 3'd3;
    localparam [2:0] OP_FC        = 3'd4;

    reg clk;
    reg rst_n;
    reg start;
    reg layer_done;
    wire layer_start;
    wire [4:0] layer_id;
    wire [2:0] op_kind;
    wire [9:0] in_channels;
    wire [9:0] out_channels;
    wire [5:0] height;
    wire [5:0] width;
    wire [1:0] stride;
    wire [31:0] weight_offset;
    wire busy;
    wire done;

    d5_mobilenet_05_top dut (
        .clk(clk), .rst_n(rst_n), .start(start), .layer_done(layer_done),
        .layer_start(layer_start), .layer_id(layer_id), .op_kind(op_kind),
        .in_channels(in_channels), .out_channels(out_channels),
        .height(height), .width(width), .stride(stride),
        .weight_offset(weight_offset), .busy(busy), .done(done)
    );

    always #5 clk = ~clk;

    function automatic [31:0] weight_bytes(input integer id);
        begin
            case (id)
                0: weight_bytes = 432;     1: weight_bytes = 144;
                2: weight_bytes = 512;     3: weight_bytes = 288;
                4: weight_bytes = 2048;    5: weight_bytes = 576;
                6: weight_bytes = 4096;    7: weight_bytes = 576;
                8: weight_bytes = 8192;    9: weight_bytes = 1152;
                10: weight_bytes = 16384;  11: weight_bytes = 1152;
                12: weight_bytes = 16384;  13: weight_bytes = 2304;
                14: weight_bytes = 32768;  15: weight_bytes = 2304;
                16: weight_bytes = 65536;  17: weight_bytes = 2304;
                18: weight_bytes = 131072; 19: weight_bytes = 4608;
                20: weight_bytes = 262144; 22: weight_bytes = 5120;
                default: weight_bytes = 0;
            endcase
        end
    endfunction

    function automatic [31:0] expected_offset(input integer id);
        integer i;
        begin
            expected_offset = 0;
            for (i = 0; i < id; i = i + 1)
                expected_offset = expected_offset + weight_bytes(i);
        end
    endfunction

    task automatic check_descriptor(input integer expected_id);
        begin
            if (!layer_start)
                $fatal(1, "descriptor %0d has no layer_start pulse", expected_id);
            if (layer_id !== expected_id[4:0])
                $fatal(1, "expected descriptor %0d, got %0d", expected_id, layer_id);
            if (weight_offset !== expected_offset(expected_id))
                $fatal(1, "descriptor %0d offset %0d, expected %0d", expected_id,
                       weight_offset, expected_offset(expected_id));

            // Check representative descriptor semantics, including the two
            // retained 256-to-256 blocks and the terminal GAP/FC stages.
            case (expected_id)
                0: if (op_kind !== OP_CONV || in_channels !== 3 || out_channels !== 16 ||
                        height !== 32 || width !== 32 || stride !== 1)
                       $fatal(1, "invalid stem descriptor");
                1: if (op_kind !== OP_DEPTHWISE || in_channels !== 16 || out_channels !== 16)
                       $fatal(1, "invalid first depthwise descriptor");
                2: if (op_kind !== OP_POINTWISE || in_channels !== 16 || out_channels !== 32)
                       $fatal(1, "invalid first pointwise descriptor");
                13, 15: if (op_kind !== OP_DEPTHWISE || in_channels !== 256 || out_channels !== 256)
                       $fatal(1, "invalid 256-channel depthwise descriptor");
                14, 16: if (op_kind !== OP_POINTWISE || in_channels !== 256 || out_channels !== 256)
                       $fatal(1, "invalid retained 256-to-256 pointwise descriptor");
                21: if (op_kind !== OP_GAP || in_channels !== 512 || out_channels !== 512 ||
                        height !== 2 || width !== 2)
                       $fatal(1, "invalid GAP descriptor");
                22: if (op_kind !== OP_FC || in_channels !== 512 || out_channels !== 10 ||
                        height !== 1 || width !== 1)
                       $fatal(1, "invalid FC descriptor");
            endcase
        end
    endtask

    integer id;
    initial begin
        clk = 1'b0;
        rst_n = 1'b0;
        start = 1'b0;
        layer_done = 1'b0;
        repeat (3) @(posedge clk);
        rst_n = 1'b1;

        // Start launches the stem descriptor.
        @(negedge clk);
        start = 1'b1;
        @(posedge clk); #1;
        start = 1'b0;
        if (!busy) $fatal(1, "sequencer did not enter busy state");
        check_descriptor(0);

        // Each modeled accelerator acknowledgement must issue the next
        // descriptor.  IDs 0..20 cover stem plus ten DW/PW blocks; 21/22
        // are GAP and FC respectively.
        for (id = 0; id < 22; id = id + 1) begin
            @(negedge clk);
            layer_done = 1'b1;
            @(posedge clk); #1;
            layer_done = 1'b0;
            check_descriptor(id + 1);
        end

        // Acknowledge FC and observe the one-cycle done pulse.
        @(negedge clk);
        layer_done = 1'b1;
        @(posedge clk); #1;
        layer_done = 1'b0;
        if (!done || busy) $fatal(1, "sequencer did not complete after FC");
        $display("D5 XSim descriptor sequencer PASS: stem + 10 blocks + GAP + FC");
        $finish;
    end
endmodule
