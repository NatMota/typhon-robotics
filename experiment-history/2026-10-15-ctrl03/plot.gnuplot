set terminal pngcairo noenhanced size 1500,1350 font 'Sans,17'
set output '/var/tmp/typhon-worktrees-20261015/ctrl03-acquisition/experiment-history/2026-10-15-ctrl03/contact-aligned-trajectories.png'
set multiplot layout 3,1 title 'CTRL03 archived acquisition paths: different entry states; descriptive comparison'
set grid
set xrange [-1.5:2.5]
set arrow 1 from 0, graph 0 to 0, graph 1 nohead dt 2 lc rgb '#777777'
set key outside right
set ylabel 'Tool XY error to initial brick (mm)'
unset xlabel
plot '/var/tmp/typhon-worktrees-20261015/ctrl03-acquisition/experiment-history/2026-10-15-ctrl03/native_B.plotdata' using 1:2 with lines lw 3 title 'native_B', '/var/tmp/typhon-worktrees-20261015/ctrl03-acquisition/experiment-history/2026-10-15-ctrl03/C_pose_then_B.plotdata' using 1:2 with lines lw 3 title 'C_pose_then_B', '/var/tmp/typhon-worktrees-20261015/ctrl03-acquisition/experiment-history/2026-10-15-ctrl03/teacher_from_B.plotdata' using 1:2 with lines lw 3 title 'teacher_from_B', '/var/tmp/typhon-worktrees-20261015/ctrl03-acquisition/experiment-history/2026-10-15-ctrl03/teacher_from_reset.plotdata' using 1:2 with lines lw 3 title 'teacher_from_reset'
set ylabel 'Tool height above initial centre (mm)'
unset xlabel
plot '/var/tmp/typhon-worktrees-20261015/ctrl03-acquisition/experiment-history/2026-10-15-ctrl03/native_B.plotdata' using 1:3 with lines lw 3 title 'native_B', '/var/tmp/typhon-worktrees-20261015/ctrl03-acquisition/experiment-history/2026-10-15-ctrl03/C_pose_then_B.plotdata' using 1:3 with lines lw 3 title 'C_pose_then_B', '/var/tmp/typhon-worktrees-20261015/ctrl03-acquisition/experiment-history/2026-10-15-ctrl03/teacher_from_B.plotdata' using 1:3 with lines lw 3 title 'teacher_from_B', '/var/tmp/typhon-worktrees-20261015/ctrl03-acquisition/experiment-history/2026-10-15-ctrl03/teacher_from_reset.plotdata' using 1:3 with lines lw 3 title 'teacher_from_reset'
set ylabel 'Brick tilt from upright (degrees)'
set xlabel 'Seconds relative to each run first contact'
plot '/var/tmp/typhon-worktrees-20261015/ctrl03-acquisition/experiment-history/2026-10-15-ctrl03/native_B.plotdata' using 1:4 with lines lw 3 title 'native_B', '/var/tmp/typhon-worktrees-20261015/ctrl03-acquisition/experiment-history/2026-10-15-ctrl03/C_pose_then_B.plotdata' using 1:4 with lines lw 3 title 'C_pose_then_B', '/var/tmp/typhon-worktrees-20261015/ctrl03-acquisition/experiment-history/2026-10-15-ctrl03/teacher_from_B.plotdata' using 1:4 with lines lw 3 title 'teacher_from_B', '/var/tmp/typhon-worktrees-20261015/ctrl03-acquisition/experiment-history/2026-10-15-ctrl03/teacher_from_reset.plotdata' using 1:4 with lines lw 3 title 'teacher_from_reset'
unset multiplot
