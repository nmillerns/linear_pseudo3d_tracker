	# Linear Pseudo3d Tracker
A simple example of a linear tracker with synthetic noisy 2d input. Uses size as a hint to do pseudo3d tracking.

A synthetic 3d model of a ball is created and moves through a 3d space in
a repeating helical pattern. This is projected and rendered on a 2d
observer image. The rendered location is taken and artifical noise is 
added and then passed on to a linear 2d tracker.

The 2d tracker includes a measure of the ball's size and that is used
as a cue for a relative estimate of a 3rd dimensional depth. The
track is tracked in 2d + depth dimensions and overlaid on the input
image and plotted in pseud 3d on separate plotting windows.


Runs as an interative program with moving image and plot displays.
Press 'q' to quit.

Written in python3 with numpy and cv2 for interactive display.

