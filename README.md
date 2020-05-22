# whisker_analysis
Tracing mouse whiskers in videos.

Videos (300 Hz) of mice moving their whiskers to touch objects ("pistons") in their vicinity (there is a one-to-one whisker-piston matching). The tracing (labeling, DNN training and prediction) was done using the deeplabcut module for animal pose estimation. Following the tracing, we analyzed the data to extract for each whisker its angle, curvature and touches at every point in time.

See video examples.

![Analysis Example](https://github.com/amirdud/whisker_analysis/blob/master/analysis_example.png)
