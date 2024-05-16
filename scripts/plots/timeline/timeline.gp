set style data histograms
set style histogram columnstacked

set boxwidth 1 relative
set style fill solid 1.0 border -1

set ylabel "Thread"
set xlabel "Laufzeit / %"

set datafile separator "@"
set xrange [0:100]
set yrange [:] reverse
set offsets 0,0,0.5,0.5
set key out

FILE = "timeline_data.txt"
stats FILE u (ColCount=words(strcol(1))-2) nooutput
set datafile separator whitespace

myBoxwidth = 0.8

set linetype 2 lc rgb "#4d88e8"
set linetype 3 lc rgb "#53cf70"
set linetype 4 lc rgb "#ffffff"

set term pdf
set output "barchart.pdf"
plot for [col=2:ColCount+1] FILE u (total=sum [i=2:ColCount+1] column(i), \
    x0=column(col),((sum [i=2:col-1] column(i))+x0/2.)/total*100):0: \
    (x0/2./total*100):(myBoxwidth/2.):ytic(1) \
    w boxxy lt col ti columnhead(col)