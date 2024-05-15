set style data lines

set xlabel "Threads"
set ylabel "Laufzeit /s"

plot 'comp_all_partitions.dat' index 0 w lp pt 7 ps 0.6 lc rgb "green" lw 2 title "2377 Datensätze", \
     ''             index 1 w lp pt 7 ps 0.6 lc rgb "orange" lw 2 title "4755 Datensätze", \
     ''             index 2 w lp pt 7 ps 0.6 lc rgb "purple" lw 2 title "9510 Datensätze", \
     ''             index 3 w lp pt 7 ps 0.6 lc rgb "blue" lw 2 title "19020 Datensätze"

set term pdf
set output "comp_all_partitions.pdf"
replot
