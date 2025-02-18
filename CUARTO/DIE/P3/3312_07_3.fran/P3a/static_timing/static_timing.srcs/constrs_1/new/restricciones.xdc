create_clock -period 5.000 -name clk -waveform {0.000 2.500} [get_ports clk]

set_input_delay -clock [get_clocks clk] 0.000 [get_ports {{a[0]} {a[1]} {a[2]} {a[3]} {a[4]} {a[5]} {a[6]} {a[7]} {a[8]} {a[9]} {a[10]} {a[11]} {a[12]} {a[13]} {a[14]} {a[15]} {a[16]} {a[17]} {a[18]} {a[19]} {a[20]} {a[21]} {a[22]} {a[23]} {a[24]} {a[25]} {a[26]} {a[27]} {a[28]} {a[29]} {a[30]} {a[31]} {b[0]} {b[1]} {b[2]} {b[3]} {b[4]} {b[5]} {b[6]} {b[7]} {b[8]} {b[9]} {b[10]} {b[11]} {b[12]} {b[13]} {b[14]} {b[15]} {b[16]} {b[17]} {b[18]} {b[19]} {b[20]} {b[21]} {b[22]} {b[23]} {b[24]} {b[25]} {b[26]} {b[27]} {b[28]} {b[29]} {b[30]} {b[31]} cin}]
set_output_delay -clock [get_clocks clk] 0.000 [get_ports -filter { NAME =~  "*" && DIRECTION == "OUT" }]
