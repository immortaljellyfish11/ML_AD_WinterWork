// argmax_10
// 作用: 从 10 个分类 logit 中找出最大值对应的类别编号。
// 输入 logit 越大，表示该类别越可能。
module argmax_10 (
    input signed [31:0] logit0,
    input signed [31:0] logit1,
    input signed [31:0] logit2,
    input signed [31:0] logit3,
    input signed [31:0] logit4,
    input signed [31:0] logit5,
    input signed [31:0] logit6,
    input signed [31:0] logit7,
    input signed [31:0] logit8,
    input signed [31:0] logit9,
    output reg [3:0] class_id,
    output reg signed [31:0] max_logit
);
    always @(*) begin
        class_id = 4'd0;
        max_logit = logit0;

        if (logit1 > max_logit) begin max_logit = logit1; class_id = 4'd1; end
        if (logit2 > max_logit) begin max_logit = logit2; class_id = 4'd2; end
        if (logit3 > max_logit) begin max_logit = logit3; class_id = 4'd3; end
        if (logit4 > max_logit) begin max_logit = logit4; class_id = 4'd4; end
        if (logit5 > max_logit) begin max_logit = logit5; class_id = 4'd5; end
        if (logit6 > max_logit) begin max_logit = logit6; class_id = 4'd6; end
        if (logit7 > max_logit) begin max_logit = logit7; class_id = 4'd7; end
        if (logit8 > max_logit) begin max_logit = logit8; class_id = 4'd8; end
        if (logit9 > max_logit) begin max_logit = logit9; class_id = 4'd9; end
    end
endmodule
