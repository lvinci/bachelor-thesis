set style data lines

set xlabel "Threads"
set ylabel "Laufzeit /s"

plot 'comp_all_partitions.dat' index 0 w lp pt 7 ps 0.6 lc rgb "green" lw 2 title "8 Partitionen", \
     ''             index 1 w lp pt 7 ps 0.6 lc rgb "orange" lw 2 title "4 Partitionen", \
     ''             index 2 w lp pt 7 ps 0.6 lc rgb "purple" lw 2 title "2 Partitionen", \
     ''             index 3 w lp pt 7 ps 0.6 lc rgb "blue" lw 2 title "1 Partition"

set term pdf
set output "comp_all_partitions.pdf"
replot
