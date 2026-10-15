set terminal pngcairo size 1600,1100 enhanced font "Arial,13"
set output "settling-comparison.png"
set datafile separator ","
set grid
set key top right
set border 3
set multiplot layout 2,2 title "CTRL04: extra settling improves entry, but does not prevent tipping"
set title "Pregrasp position error during the added hold"
set xlabel "Extra hold (seconds)"
set ylabel "Error to unchanged pregrasp target (mm)"
plot "hold-plot.csv" every ::1 using 1:2 with lines lw 3 lc rgb "#2471a3" title "E80"
set title "Motion settles during hold"
set ylabel "Tool linear speed (mm/s)"
plot "hold-plot.csv" every ::1 using 1:4 with lines lw 3 lc rgb "#2471a3" notitle
set key bottom right
set title "Brick tilt aligned at descent entry"
set xlabel "Seconds after descent starts"
set ylabel "Tilt from upright (degrees)"
set xrange [0:6]
plot "E0-plot.csv" every ::1 using 2:4 with lines lw 3 lc rgb "#b03a2e" title "E0 original", "E80-plot.csv" every ::1 using 2:4 with lines lw 3 lc rgb "#2471a3" title "E80 +4s"
set title "Tracking error after descent starts"
set ylabel "Tool error to active target (mm)"
plot "E0-plot.csv" every ::1 using 2:3 with lines lw 3 lc rgb "#b03a2e" title "E0 original", "E80-plot.csv" every ::1 using 2:3 with lines lw 3 lc rgb "#2471a3" title "E80 +4s"
unset multiplot
