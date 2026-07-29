`timescale 1ns / 1ps

// D5 MobileNetV1-0.5x external-weight sequencer.
//
// The XC7Z010 cannot store D5 parameters on chip.  This synthesizable top
// therefore controls a streamed/tiled convolution accelerator: each descriptor
// identifies exactly one folded-BN operator and the accelerator returns
// layer_done after consuming its associated external weight tile(s).
// Network: stem + 10 depthwise/pointwise blocks + GAP + FC.
module d5_mobilenet_05_top(
    input clk, input rst_n, input start, input layer_done,
    output reg layer_start, output reg [4:0] layer_id,
    output reg [2:0] op_kind,                 // 0=3x3, 1=depthwise, 2=pointwise, 3=GAP, 4=FC
    output reg [9:0] in_channels, output reg [9:0] out_channels,
    output reg [5:0] height, output reg [5:0] width, output reg [1:0] stride,
    output reg [31:0] weight_offset, output reg busy, output reg done
);
    localparam OP_CONV=0, OP_DEPTHWISE=1, OP_POINTWISE=2, OP_GAP=3, OP_FC=4;
    reg active;
    // All folded weights are INT8 and laid out in exported layer order.
    function [31:0] layer_weight_bytes;
        input [4:0] id;
        begin
            case(id)
                0:layer_weight_bytes=432;  1:layer_weight_bytes=144;  2:layer_weight_bytes=512;
                3:layer_weight_bytes=288;  4:layer_weight_bytes=2048; 5:layer_weight_bytes=576;
                6:layer_weight_bytes=4096; 7:layer_weight_bytes=576;  8:layer_weight_bytes=8192;
                9:layer_weight_bytes=1152; 10:layer_weight_bytes=16384;11:layer_weight_bytes=1152;
                12:layer_weight_bytes=16384;13:layer_weight_bytes=2304;14:layer_weight_bytes=32768;
                15:layer_weight_bytes=2304;16:layer_weight_bytes=65536;17:layer_weight_bytes=2304;
                18:layer_weight_bytes=131072;19:layer_weight_bytes=4608;20:layer_weight_bytes=262144;
                22:layer_weight_bytes=5120;
                default:layer_weight_bytes=0;
            endcase
        end
    endfunction
    task set_descriptor;
        input [4:0] id;
        begin
            layer_id<=id; stride<=0; in_channels<=0; out_channels<=0; height<=0; width<=0; op_kind<=OP_CONV;
            case(id)
                0:begin op_kind<=OP_CONV;in_channels<=3;out_channels<=16;height<=32;width<=32;stride<=1;end
                1:begin op_kind<=OP_DEPTHWISE;in_channels<=16;out_channels<=16;height<=32;width<=32;stride<=1;end
                2:begin op_kind<=OP_POINTWISE;in_channels<=16;out_channels<=32;height<=32;width<=32;stride<=1;end
                3:begin op_kind<=OP_DEPTHWISE;in_channels<=32;out_channels<=32;height<=32;width<=32;stride<=2;end
                4:begin op_kind<=OP_POINTWISE;in_channels<=32;out_channels<=64;height<=16;width<=16;stride<=1;end
                5:begin op_kind<=OP_DEPTHWISE;in_channels<=64;out_channels<=64;height<=16;width<=16;stride<=1;end
                6:begin op_kind<=OP_POINTWISE;in_channels<=64;out_channels<=64;height<=16;width<=16;stride<=1;end
                7:begin op_kind<=OP_DEPTHWISE;in_channels<=64;out_channels<=64;height<=16;width<=16;stride<=2;end
                8:begin op_kind<=OP_POINTWISE;in_channels<=64;out_channels<=128;height<=8;width<=8;stride<=1;end
                9:begin op_kind<=OP_DEPTHWISE;in_channels<=128;out_channels<=128;height<=8;width<=8;stride<=1;end
                10:begin op_kind<=OP_POINTWISE;in_channels<=128;out_channels<=128;height<=8;width<=8;stride<=1;end
                11:begin op_kind<=OP_DEPTHWISE;in_channels<=128;out_channels<=128;height<=8;width<=8;stride<=2;end
                12:begin op_kind<=OP_POINTWISE;in_channels<=128;out_channels<=256;height<=4;width<=4;stride<=1;end
                13,15:begin op_kind<=OP_DEPTHWISE;in_channels<=256;out_channels<=256;height<=4;width<=4;stride<=1;end
                14,16:begin op_kind<=OP_POINTWISE;in_channels<=256;out_channels<=256;height<=4;width<=4;stride<=1;end
                17:begin op_kind<=OP_DEPTHWISE;in_channels<=256;out_channels<=256;height<=4;width<=4;stride<=2;end
                18:begin op_kind<=OP_POINTWISE;in_channels<=256;out_channels<=512;height<=2;width<=2;stride<=1;end
                19:begin op_kind<=OP_DEPTHWISE;in_channels<=512;out_channels<=512;height<=2;width<=2;stride<=1;end
                20:begin op_kind<=OP_POINTWISE;in_channels<=512;out_channels<=512;height<=2;width<=2;stride<=1;end
                21:begin op_kind<=OP_GAP;in_channels<=512;out_channels<=512;height<=2;width<=2;stride<=1;end
                22:begin op_kind<=OP_FC;in_channels<=512;out_channels<=10;height<=1;width<=1;stride<=1;end
            endcase
        end
    endtask
    always @(posedge clk or negedge rst_n) begin
        if(!rst_n) begin active<=0;layer_start<=0;layer_id<=0;op_kind<=0;in_channels<=0;out_channels<=0;height<=0;width<=0;stride<=0;weight_offset<=0;busy<=0;done<=0;end
        else begin
            layer_start<=0; done<=0;
            if(!active) begin
                busy<=0;
                if(start) begin active<=1;busy<=1;weight_offset<=0;set_descriptor(0);layer_start<=1;end
            end else if(layer_done) begin
                if(layer_id==22) begin active<=0;busy<=0;done<=1;end
                else begin weight_offset<=weight_offset+layer_weight_bytes(layer_id);set_descriptor(layer_id+1'b1);layer_start<=1;end
            end
        end
    end
endmodule
