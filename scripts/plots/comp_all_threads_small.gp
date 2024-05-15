

set xlabel "Anzahl der Datensätze"
set ylabel "Laufzeit /s"

set xtics ("4755" 1, "2377" 2)

set yrange [0:8000]

plot 'comp_all_threads.dat' index 0 w lp pt 7 ps 0.6 lc rgb "green" lw 2 title "4 Threads", \
     ''             index 1 w lp pt 7 ps 0.6 lc rgb "red" lw 2 title "2 Threads", \
     ''             index 2 w lp pt 7 ps 0.6 lc rgb "grey" lw 2 title "1 Threads"

set term pdf
set output "comp_all_threads.pdf"
replot
