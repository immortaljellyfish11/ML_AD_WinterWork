# Descriptor-sequencer timing constraint.  This is the 100 MHz PL timing
# characterization clock, not a board pin assignment and not a bitstream flow.
create_clock -name d5_descriptor_clk -period 10.000 [get_ports clk]
