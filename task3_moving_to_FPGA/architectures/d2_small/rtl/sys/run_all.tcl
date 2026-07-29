# D2 XSim, synthesis and implementation flow / D2 仿真、综合与实现流程。
set script_dir [file dirname [file normalize [info script]]]
set design_dir [file normalize [file join $script_dir ../..]]
set src_dir [file join $design_dir rtl src]
set sim_dir [file join $design_dir rtl sim]
set param_dir [file join $design_dir artifacts int8_params]
set project_dir [file join $script_dir project_d2]
set report_dir [file join $script_dir reports]
file mkdir $report_dir
source [file join $param_dir rtl_params.tcl]
create_project -force d2_small $project_dir -part xc7z010clg400-1
add_files [glob [file join $src_dir *.v]]
add_files [file join $script_dir clock.xdc]
add_files -norecurse [glob [file join $param_dir *.mem]]
set_property include_dirs [list $param_dir] [get_filesets sources_1]
set_property top d2_small_top [get_filesets sources_1]
set_property generic "CONV1_REQUANT_MULTIPLIER=$Q_CONV1_M CONV1_REQUANT_SHIFT=$Q_CONV1_S CONV2_REQUANT_MULTIPLIER=$Q_CONV2_M CONV2_REQUANT_SHIFT=$Q_CONV2_S CONV3_REQUANT_MULTIPLIER=$Q_CONV3_M CONV3_REQUANT_SHIFT=$Q_CONV3_S" [get_filesets sources_1]
add_files -fileset sim_1 [file join $sim_dir tb_d2_small_xsim.sv]
add_files -fileset sim_1 [file join $param_dir rtl_params.vh]
add_files -fileset sim_1 [file join $sim_dir data input_int8.txt]
add_files -fileset sim_1 [file join $sim_dir data golden_logits.txt]
set_property include_dirs [list $param_dir] [get_filesets sim_1]
set_property top tb_d2_small_xsim [get_filesets sim_1]
launch_simulation
run all
close_sim
launch_runs synth_1 -jobs 4
wait_on_run synth_1
open_run synth_1
report_utilization -file [file join $report_dir utilization_synth.rpt]
launch_runs impl_1 -to_step route_design -jobs 4
wait_on_run impl_1
open_run impl_1
report_utilization -file [file join $report_dir utilization_route.rpt]
report_timing_summary -file [file join $report_dir timing_route.rpt]
puts "D2 VIVADO FLOW PASS"
