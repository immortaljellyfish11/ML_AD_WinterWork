# D5 external-weight descriptor-sequencer XSim / synthesis / route flow.
# Scope: only d5_mobilenet_05_top.  This flow does not instantiate external
# memory, tiled convolution engines, or a complete MobileNet inference core.
# It intentionally stops after route_design and never generates a bitstream.
set script_dir [file dirname [file normalize [info script]]]
set design_dir [file normalize [file join $script_dir ../..]]
set src_dir [file join $design_dir rtl src]
set sim_dir [file join $design_dir rtl sim]
set project_dir [file join $script_dir project_d5_descriptor]
set report_dir [file join $script_dir reports]

file mkdir $report_dir
create_project -force d5_descriptor_sequencer $project_dir -part xc7z010clg400-1
set_property target_simulator XSim [current_project]

add_files [file join $src_dir d5_mobilenet_05_top.v]
add_files [file join $script_dir clock.xdc]
set_property top d5_mobilenet_05_top [get_filesets sources_1]

add_files -fileset sim_1 [file join $sim_dir tb_d5_mobilenet_05_xsim.sv]
set_property top tb_d5_mobilenet_05_xsim [get_filesets sim_1]

launch_simulation
close_sim

launch_runs synth_1 -jobs 4
wait_on_run synth_1
open_run synth_1
report_utilization -file [file join $report_dir utilization_synth.rpt]
report_timing_summary -file [file join $report_dir timing_synth.rpt]

launch_runs impl_1 -to_step route_design -jobs 4
wait_on_run impl_1
open_run impl_1
report_utilization -file [file join $report_dir utilization_route.rpt]
report_timing_summary -file [file join $report_dir timing_route.rpt]
report_route_status -file [file join $report_dir route_status.rpt]

puts "D5 descriptor-sequencer Vivado flow PASS (not a complete MobileNet core)"
