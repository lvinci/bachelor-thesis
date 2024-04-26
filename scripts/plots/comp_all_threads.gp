

set xlabel "Partitionen"
set ylabel "Laufzeit /s"

plot 'comp_all_threads.dat' index 0 w lp pt 7 ps 0.6 lc rgb "green" lw 2 title "10 Threads", \
     ''             index 1 w lp pt 7 ps 0.6 lc rgb "blue" lw 2 title "8 Threads", \
     ''             index 2 w lp dt 2 pt 7 ps 0.6 lc rgb "orange" lw 2 title "6 Threads", \
     ''             index 3 w lp pt 7 ps 0.6 lc rgb "purple" lw 2 title "4 Threads", \
     ''             index 4 w lp pt 7 ps 0.6 lc rgb "red" lw 2 title "2 Threads", \
     ''             index 5 w lp pt 7 ps 0.6 lc rgb "grey" lw 2 title "1 Threads"

set term pdf
set output "comp_all_threads.pdf"
replot
