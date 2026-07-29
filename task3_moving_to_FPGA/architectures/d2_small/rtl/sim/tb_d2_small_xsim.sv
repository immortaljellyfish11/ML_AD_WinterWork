`timescale 1ns/1ps
`include "rtl_params.vh"
// One-vector XSim regression / 单向量 XSim 回归测试。
module tb_d2_small_xsim;
    localparam INPUT_SIZE=3072, MAX_CYCLES=100000000;
    reg clk=0,rst_n=0,start=0,input_we=0; reg [16:0] input_addr=0; reg signed [7:0] input_data=0;
    wire busy,done; wire [3:0] class_id,state_dbg; wire signed [31:0] max_logit;
    integer fi,fg,status,value,index,cycles,label,golden_class,golden_max;
    d2_small_top #(
        .CONV1_WEIGHT_FILE("conv1_weight.mem"),.CONV2_WEIGHT_FILE("conv2_weight.mem"),
        .CONV3_WEIGHT_FILE("conv3_weight.mem"),.LINEAR_WEIGHT_FILE("linear_weight.mem"),
        .CONV1_BIAS_FILE("conv1_bias.mem"),.CONV2_BIAS_FILE("conv2_bias.mem"),
        .CONV3_BIAS_FILE("conv3_bias.mem"),.LINEAR_BIAS_FILE("linear_bias.mem"),
        .CONV1_REQUANT_MULTIPLIER(`Q_CONV1_M),.CONV1_REQUANT_SHIFT(`Q_CONV1_S),
        .CONV2_REQUANT_MULTIPLIER(`Q_CONV2_M),.CONV2_REQUANT_SHIFT(`Q_CONV2_S),
        .CONV3_REQUANT_MULTIPLIER(`Q_CONV3_M),.CONV3_REQUANT_SHIFT(`Q_CONV3_S)
    ) dut(.clk(clk),.rst_n(rst_n),.start(start),.input_we(input_we),.input_addr(input_addr),
          .input_data(input_data),.busy(busy),.done(done),.class_id(class_id),.max_logit(max_logit),.state_dbg(state_dbg));
    always #5 clk=~clk;
    initial begin
        fi=$fopen("input_int8.txt","r"); fg=$fopen("golden_logits.txt","r");
        if(!fi) fi=$fopen("../../../../rtl/sim/data/input_int8.txt","r");
        if(!fg) fg=$fopen("../../../../rtl/sim/data/golden_logits.txt","r");
        if(!fi||!fg)$fatal(1,"XSIM FAIL: vectors not found");
        status=$fscanf(fg,"%d %d %d",label,golden_class,golden_max); if(status!=3)$fatal(1,"bad golden header");
        repeat(5)@(posedge clk);@(negedge clk);rst_n=1;
        for(index=0;index<INPUT_SIZE;index=index+1)begin status=$fscanf(fi,"%d",value);if(status!=1)$fatal(1,"input truncated");
            @(negedge clk);input_we=1;input_addr=index;input_data=value;end
        @(negedge clk);input_we=0;@(negedge clk);start=1;@(negedge clk);start=0;cycles=0;
        while(!done&&cycles<MAX_CYCLES)begin @(posedge clk);#1;cycles=cycles+1;end
        if(!done)$fatal(1,"XSIM timeout state=%0d",state_dbg);
        if(class_id!==golden_class[3:0]||max_logit!==golden_max)$fatal(1,"XSIM mismatch class=%0d/%0d max=%0d/%0d",class_id,golden_class,max_logit,golden_max);
        $display("XSIM D2 PASS: label=%0d class=%0d max=%0d cycles=%0d",label,class_id,max_logit,cycles);$finish;
    end
endmodule
