# Gnuplot script to generate a NCI interval bar chart
set terminal postscript landscape enhanced color 'Helvetica' 20
#set encoding iso_8859_1
set output 'bar_diagram_nciplot.ps'
#set ylabel 'N (e^{-})' font "Helvetica, 30"
set xlabel 'Interval of sgn({/Symbol l}_2)*{/Symbol r} (a.u.)'
set format y "% .2f"
set format x "% .2f"
set format cb "% -.2f"
set border lw 4

set style fill solid 0.5

#unset key
#set key fixed right top vertical Right noreverse enhanced autotitle nobox
set style data histogram
set style histogram cluster gap 1
set style fill solid border -1
set boxwidth 0.9
set xtic scale 0
plot "int.dat" u 3:xtic(2) ti col ,  '' u 4 ti col, 0 lc "black" title ""

#plot "int.dat" u 3:xtic(2) ti col, '' u 4 ti col, '' u 5 ti col, '' u 6 ti col, 0 lc "black" title ""

