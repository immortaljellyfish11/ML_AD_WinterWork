// D3 VGG-like INT8 CNN: 16 -> 32 -> 64 -> 128 channels.
// All feature maps use two ping-pong BRAM buffers; conv4 weights are distributed
// ROM to keep the intended XC7Z010 synthesis configuration practical.
module d3_vgg_like_top #(
    parameter integer ADDR_WIDTH=17, parameter integer FEATURE_DEPTH=32768,
    parameter C1_W="conv1_weight.mem", parameter C2_W="conv2_weight.mem",
    parameter C3_W="conv3_weight.mem", parameter C4_W="conv4_weight.mem",
    parameter FC_W="linear_weight.mem", parameter C1_B="conv1_bias.mem",
    parameter C2_B="conv2_bias.mem", parameter C3_B="conv3_bias.mem",
    parameter C4_B="conv4_bias.mem", parameter FC_B="linear_bias.mem",
    parameter integer C1_M=1, parameter integer C1_S=0,
    parameter integer C2_M=1, parameter integer C2_S=0,
    parameter integer C3_M=1, parameter integer C3_S=0,
    parameter integer C4_M=1, parameter integer C4_S=0
)(
    input clk,input rst_n,input start,input input_we,
    input [ADDR_WIDTH-1:0] input_addr,input signed [7:0] input_data,
    output busy,output done,output [3:0] class_id,output signed [31:0] max_logit,
    output argmax_valid,output [3:0] state_dbg
);
    localparam C1=1,C2=2,P1=3,C3=4,C4=5,P2=6,GAP=7,FC=8;
    wire c1_start,c2_start,p1_start,c3_start,c4_start,p2_start,gap_start,fc_start;
    wire c1_done,c2_done,p1_done,c3_done,c4_done,p2_done,gap_done,fc_done;
    wire c1_busy,c2_busy,p1_busy,c3_busy,c4_busy,p2_busy,gap_busy,fc_busy;
    wire layer_busy=c1_busy|c2_busy|p1_busy|c3_busy|c4_busy|p2_busy|gap_busy|fc_busy;
    d3_vgg_like_fsm fsm(.clk(clk),.rst_n(rst_n),.start(start),.conv1_done(c1_done),.conv2_done(c2_done),.pool1_done(p1_done),.conv3_done(c3_done),.conv4_done(c4_done),.pool2_done(p2_done),.gap_done(gap_done),.linear_done(fc_done),.conv1_start(c1_start),.conv2_start(c2_start),.pool1_start(p1_start),.conv3_start(c3_start),.conv4_start(c4_start),.pool2_start(p2_start),.gap_start(gap_start),.linear_start(fc_start),.argmax_valid(argmax_valid),.busy(busy),.done(done),.state_dbg(state_dbg));

    wire [ADDR_WIDTH-1:0] a_ra,b_ra,a_wa,b_wa; wire signed [7:0] a_rd,b_rd,a_wd,b_wd; wire a_we,b_we;
    feature_buffer_sync #(.ADDR_WIDTH(ADDR_WIDTH),.DEPTH(FEATURE_DEPTH)) buffer_a(.clk(clk),.we(a_we),.waddr(a_wa),.wdata(a_wd),.raddr(a_ra),.rdata(a_rd));
    feature_buffer_sync #(.ADDR_WIDTH(ADDR_WIDTH),.DEPTH(FEATURE_DEPTH)) buffer_b(.clk(clk),.we(b_we),.waddr(b_wa),.wdata(b_wd),.raddr(b_ra),.rdata(b_rd));

    wire [ADDR_WIDTH-1:0] c1_fa,c2_fa,c3_fa,c4_fa,c1_wa,c2_wa,c3_wa,c4_wa;
    wire [9:0] c1_ba,c2_ba,c3_ba,c4_ba; wire c1_owe,c2_owe,c3_owe,c4_owe;
    wire [ADDR_WIDTH-1:0] c1_oa,c2_oa,c3_oa,c4_oa; wire signed [7:0] c1_od,c2_od,c3_od,c4_od;
    wire [ADDR_WIDTH-1:0] p1_ia,p2_ia,p1_oa,p2_oa,gap_ia,gap_oa,fc_fa,fc_wa;
    wire p1_owe,p2_owe,gap_owe; wire signed [7:0] p1_od,p2_od,gap_od;
    wire [3:0] fc_ba,logit_index; wire signed [31:0] logit_data; wire logit_valid;

    // A=input/C2/C3/P2; B=C1/P1/C4/GAP.  Reads follow the opposite buffer.
    assign a_ra=(state_dbg==C1)?c1_fa:(state_dbg==P1)?p1_ia:(state_dbg==C4)?c4_fa:(state_dbg==GAP)?gap_ia:0;
    assign b_ra=(state_dbg==C2)?c2_fa:(state_dbg==C3)?c3_fa:(state_dbg==P2)?p2_ia:(state_dbg==FC)?fc_fa:0;
    assign a_we=(!busy&&!layer_busy&&input_we)|c2_owe|c3_owe|p2_owe;
    assign a_wa=(!busy&&!layer_busy&&input_we)?input_addr:c2_owe?c2_oa:c3_owe?c3_oa:p2_oa;
    assign a_wd=(!busy&&!layer_busy&&input_we)?input_data:c2_owe?c2_od:c3_owe?c3_od:p2_od;
    assign b_we=c1_owe|p1_owe|c4_owe|gap_owe;
    assign b_wa=c1_owe?c1_oa:p1_owe?p1_oa:c4_owe?c4_oa:gap_oa;
    assign b_wd=c1_owe?c1_od:p1_owe?p1_od:c4_owe?c4_od:gap_od;

    wire signed [7:0] c1_wd,c2_wd,c3_wd,c4_wd,fc_wd; wire signed [31:0] c1_bd,c2_bd,c3_bd,c4_bd,fc_bd;
    weight_rom #(.ADDR_WIDTH(ADDR_WIDTH),.DEPTH(432),.INIT_FILE(C1_W)) rw1(.clk(clk),.addr(c1_wa),.data(c1_wd));
    weight_rom #(.ADDR_WIDTH(ADDR_WIDTH),.DEPTH(4608),.INIT_FILE(C2_W)) rw2(.clk(clk),.addr(c2_wa),.data(c2_wd));
    weight_rom #(.ADDR_WIDTH(ADDR_WIDTH),.DEPTH(18432),.INIT_FILE(C3_W)) rw3(.clk(clk),.addr(c3_wa),.data(c3_wd));
    weight_rom_distributed #(.ADDR_WIDTH(ADDR_WIDTH),.DEPTH(73728),.INIT_FILE(C4_W)) rw4(.clk(clk),.addr(c4_wa),.data(c4_wd));
    weight_rom #(.ADDR_WIDTH(ADDR_WIDTH),.DEPTH(1280),.INIT_FILE(FC_W)) rwf(.clk(clk),.addr(fc_wa),.data(fc_wd));
    bias_rom #(.ADDR_WIDTH(10),.DEPTH(16),.INIT_FILE(C1_B)) rb1(.addr(c1_ba),.data(c1_bd));
    bias_rom #(.ADDR_WIDTH(10),.DEPTH(32),.INIT_FILE(C2_B)) rb2(.addr(c2_ba),.data(c2_bd));
    bias_rom #(.ADDR_WIDTH(10),.DEPTH(64),.INIT_FILE(C3_B)) rb3(.addr(c3_ba),.data(c3_bd));
    bias_rom #(.ADDR_WIDTH(10),.DEPTH(128),.INIT_FILE(C4_B)) rb4(.addr(c4_ba),.data(c4_bd));
    bias_rom #(.ADDR_WIDTH(4),.DEPTH(10),.INIT_FILE(FC_B)) rbf(.addr(fc_ba),.data(fc_bd));

    conv_layer_engine_sync #(.IN_HEIGHT(32),.IN_WIDTH(32),.IN_CHANNELS(3),.OUT_HEIGHT(32),.OUT_WIDTH(32),.OUT_CHANNELS(16),.REQUANT_MULTIPLIER(C1_M),.REQUANT_SHIFT(C1_S),.ADDR_WIDTH(ADDR_WIDTH)) conv1(.clk(clk),.rst_n(rst_n),.start(c1_start),.feature_data(a_rd),.weight_data(c1_wd),.bias_data(c1_bd),.feature_addr(c1_fa),.weight_addr(c1_wa),.bias_addr(c1_ba),.out_we(c1_owe),.out_addr(c1_oa),.out_data(c1_od),.busy(c1_busy),.done(c1_done));
    conv_layer_engine_sync #(.IN_HEIGHT(32),.IN_WIDTH(32),.IN_CHANNELS(16),.OUT_HEIGHT(32),.OUT_WIDTH(32),.OUT_CHANNELS(32),.REQUANT_MULTIPLIER(C2_M),.REQUANT_SHIFT(C2_S),.ADDR_WIDTH(ADDR_WIDTH)) conv2(.clk(clk),.rst_n(rst_n),.start(c2_start),.feature_data(b_rd),.weight_data(c2_wd),.bias_data(c2_bd),.feature_addr(c2_fa),.weight_addr(c2_wa),.bias_addr(c2_ba),.out_we(c2_owe),.out_addr(c2_oa),.out_data(c2_od),.busy(c2_busy),.done(c2_done));
    pool_layer_2x2_engine_sync #(.IN_HEIGHT(32),.IN_WIDTH(32),.CHANNELS(32),.ADDR_WIDTH(ADDR_WIDTH)) pool1(.clk(clk),.rst_n(rst_n),.start(p1_start),.in_data(a_rd),.in_addr(p1_ia),.out_we(p1_owe),.out_addr(p1_oa),.out_data(p1_od),.busy(p1_busy),.done(p1_done));
    conv_layer_engine_sync #(.IN_HEIGHT(16),.IN_WIDTH(16),.IN_CHANNELS(32),.OUT_HEIGHT(16),.OUT_WIDTH(16),.OUT_CHANNELS(64),.REQUANT_MULTIPLIER(C3_M),.REQUANT_SHIFT(C3_S),.ADDR_WIDTH(ADDR_WIDTH)) conv3(.clk(clk),.rst_n(rst_n),.start(c3_start),.feature_data(b_rd),.weight_data(c3_wd),.bias_data(c3_bd),.feature_addr(c3_fa),.weight_addr(c3_wa),.bias_addr(c3_ba),.out_we(c3_owe),.out_addr(c3_oa),.out_data(c3_od),.busy(c3_busy),.done(c3_done));
    conv_layer_engine_sync #(.IN_HEIGHT(16),.IN_WIDTH(16),.IN_CHANNELS(64),.OUT_HEIGHT(16),.OUT_WIDTH(16),.OUT_CHANNELS(128),.REQUANT_MULTIPLIER(C4_M),.REQUANT_SHIFT(C4_S),.ADDR_WIDTH(ADDR_WIDTH)) conv4(.clk(clk),.rst_n(rst_n),.start(c4_start),.feature_data(a_rd),.weight_data(c4_wd),.bias_data(c4_bd),.feature_addr(c4_fa),.weight_addr(c4_wa),.bias_addr(c4_ba),.out_we(c4_owe),.out_addr(c4_oa),.out_data(c4_od),.busy(c4_busy),.done(c4_done));
    pool_layer_2x2_engine_sync #(.IN_HEIGHT(16),.IN_WIDTH(16),.CHANNELS(128),.ADDR_WIDTH(ADDR_WIDTH)) pool2(.clk(clk),.rst_n(rst_n),.start(p2_start),.in_data(b_rd),.in_addr(p2_ia),.out_we(p2_owe),.out_addr(p2_oa),.out_data(p2_od),.busy(p2_busy),.done(p2_done));
    global_avg_pool_layer_engine_sync #(.IN_HEIGHT(8),.IN_WIDTH(8),.CHANNELS(128),.ADDR_WIDTH(ADDR_WIDTH)) gap(.clk(clk),.rst_n(rst_n),.start(gap_start),.in_data(a_rd),.in_addr(gap_ia),.out_we(gap_owe),.out_addr(gap_oa),.out_data(gap_od),.busy(gap_busy),.done(gap_done));
    linear_nx10_sync #(.IN_FEATURES(128),.INDEX_WIDTH(7),.FEATURE_ADDR_WIDTH(ADDR_WIDTH),.WEIGHT_ADDR_WIDTH(ADDR_WIDTH)) linear(.clk(clk),.rst_n(rst_n),.start(fc_start),.feature_data(b_rd),.weight_data(fc_wd),.bias_data(fc_bd),.feature_addr(fc_fa),.weight_addr(fc_wa),.bias_addr(fc_ba),.logit_index(logit_index),.logit_data(logit_data),.logit_valid(logit_valid),.busy(fc_busy),.done(fc_done));
    reg signed [31:0] l0,l1,l2,l3,l4,l5,l6,l7,l8,l9;
    always @(posedge clk or negedge rst_n) begin if(!rst_n) begin l0<=0;l1<=0;l2<=0;l3<=0;l4<=0;l5<=0;l6<=0;l7<=0;l8<=0;l9<=0; end else if(logit_valid) case(logit_index) 0:l0<=logit_data;1:l1<=logit_data;2:l2<=logit_data;3:l3<=logit_data;4:l4<=logit_data;5:l5<=logit_data;6:l6<=logit_data;7:l7<=logit_data;8:l8<=logit_data;9:l9<=logit_data;endcase end
    argmax_10 argmax(.logit0(l0),.logit1(l1),.logit2(l2),.logit3(l3),.logit4(l4),.logit5(l5),.logit6(l6),.logit7(l7),.logit8(l8),.logit9(l9),.class_id(class_id),.max_logit(max_logit));
endmodule
