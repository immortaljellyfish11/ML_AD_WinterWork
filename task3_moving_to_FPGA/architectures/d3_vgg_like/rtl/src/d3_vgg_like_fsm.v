// D3 controller: Conv16 -> Conv32 -> Pool -> Conv64 -> Conv128 -> Pool -> GAP -> FC.
module d3_vgg_like_fsm(
    input clk, input rst_n, input start,
    input conv1_done, input conv2_done, input pool1_done, input conv3_done,
    input conv4_done, input pool2_done, input gap_done, input linear_done,
    output reg conv1_start, output reg conv2_start, output reg pool1_start,
    output reg conv3_start, output reg conv4_start, output reg pool2_start,
    output reg gap_start, output reg linear_start, output reg argmax_valid,
    output reg busy, output reg done, output reg [3:0] state_dbg
);
    localparam IDLE=0,C1=1,C2=2,P1=3,C3=4,C4=5,P2=6,GAP=7,FC=8,ARG=9,DONE=10;
    reg [3:0] state;
    always @(posedge clk or negedge rst_n) begin
        if(!rst_n) begin
            state<=IDLE; conv1_start<=0;conv2_start<=0;pool1_start<=0;conv3_start<=0;
            conv4_start<=0;pool2_start<=0;gap_start<=0;linear_start<=0;argmax_valid<=0;
            busy<=0;done<=0;state_dbg<=IDLE;
        end else begin
            conv1_start<=0;conv2_start<=0;pool1_start<=0;conv3_start<=0;conv4_start<=0;
            pool2_start<=0;gap_start<=0;linear_start<=0;argmax_valid<=0;done<=0;state_dbg<=state;
            case(state)
                IDLE: begin busy<=0; if(start) begin busy<=1;conv1_start<=1;state<=C1;end end
                C1: if(conv1_done) begin conv2_start<=1;state<=C2;end
                C2: if(conv2_done) begin pool1_start<=1;state<=P1;end
                P1: if(pool1_done) begin conv3_start<=1;state<=C3;end
                C3: if(conv3_done) begin conv4_start<=1;state<=C4;end
                C4: if(conv4_done) begin pool2_start<=1;state<=P2;end
                P2: if(pool2_done) begin gap_start<=1;state<=GAP;end
                GAP: if(gap_done) begin linear_start<=1;state<=FC;end
                FC: if(linear_done) begin argmax_valid<=1;state<=ARG;end
                ARG: state<=DONE;
                DONE: begin busy<=0;done<=1;state<=IDLE;end
                default: state<=IDLE;
            endcase
        end
    end
endmodule
