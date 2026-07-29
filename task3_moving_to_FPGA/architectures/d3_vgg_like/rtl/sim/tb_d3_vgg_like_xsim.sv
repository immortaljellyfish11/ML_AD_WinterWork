`timescale 1ns/1ps
`include "rtl_params.vh"
// One-vector XSim regression / 单向量 XSim 回归测试。
module tb_d3_vgg_like_xsim;
    localparam INPUT_SIZE=3072, MAX_CYCLES=500000000;
    reg clk=0,rst_n=0,start=0,input_we=0; reg [16:0] input_addr=0; reg signed [7:0] input_data=0;
    wire busy,done,argmax_valid; wire [3:0] class_id,state_dbg; wire signed [31:0] max_logit;
    integer fi,fg,status,value,index,cycles,label,golden_class,golden_max;
    d3_vgg_like_top #(
        .C1_W("conv1_weight.mem"),.C2_W("conv2_weight.mem"),.C3_W("conv3_weight.mem"),.C4_W("conv4_weight.mem"),.FC_W("linear_weight.mem"),
        .C1_B("conv1_bias.mem"),.C2_B("conv2_bias.mem"),.C3_B("conv3_bias.mem"),.C4_B("conv4_bias.mem"),.FC_B("linear_bias.mem"),
        .C1_M(`Q_CONV1_M),.C1_S(`Q_CONV1_S),.C2_M(`Q_CONV2_M),.C2_S(`Q_CONV2_S),
        .C3_M(`Q_CONV3_M),.C3_S(`Q_CONV3_S),.C4_M(`Q_CONV4_M),.C4_S(`Q_CONV4_S)
    ) dut(.clk(clk),.rst_n(rst_n),.start(start),.input_we(input_we),.input_addr(input_addr),.input_data(input_data),
          .busy(busy),.done(done),.class_id(class_id),.max_logit(max_logit),.argmax_valid(argmax_valid),.state_dbg(state_dbg));
    always #5 clk=~clk;
    initial begin
        fi=$fopen("input_int8.txt","r");fg=$fopen("golden_logits.txt","r");
        if(!fi)fi=$fopen("../../../../rtl/sim/data/input_int8.txt","r");if(!fg)fg=$fopen("../../../../rtl/sim/data/golden_logits.txt","r");
        if(!fi||!fg)$fatal(1,"XSIM FAIL: vectors not found");status=$fscanf(fg,"%d %d %d",label,golden_class,golden_max);if(status!=3)$fatal(1,"bad golden header");
        repeat(5)@(posedge clk);@(negedge clk);rst_n=1;
        for(index=0;index<INPUT_SIZE;index=index+1)begin status=$fscanf(fi,"%d",value);if(status!=1)$fatal(1,"input truncated");@(negedge clk);input_we=1;input_addr=index;input_data=value;end
        @(negedge clk);input_we=0;@(negedge clk);start=1;@(negedge clk);start=0;cycles=0;
        while(!done&&cycles<MAX_CYCLES)begin @(posedge clk);#1;cycles=cycles+1;end
        if(!done)$fatal(1,"XSIM timeout state=%0d",state_dbg);if(class_id!==golden_class[3:0]||max_logit!==golden_max)$fatal(1,"XSIM mismatch class=%0d/%0d max=%0d/%0d",class_id,golden_class,max_logit,golden_max);
        $display("XSIM D3 PASS: label=%0d class=%0d max=%0d cycles=%0d",label,class_id,max_logit,cycles);$finish;
    end
endmodule
