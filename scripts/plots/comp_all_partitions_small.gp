set style data lines

set xlabel "Threads"
set ylabel "Laufzeit /s"

set xtics ("1" 1, "2" 2, "3" 3, "4" 4)

set yrange [0:8000]

set term pdf
set output "comp_all_partitions.pdf"
plot 'comp_all_partitions.dat' index 0 w lp pt 7 ps 0.6 lc rgb "green" lw 2 title "2377 Datensätze", \
     ''             index 1 w lp pt 7 ps 0.6 lc rgb "orange" lw 2 title "4755 Datensätze"
