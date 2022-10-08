# Non-contact-Heart-Rate-Estimation


Non-contact heart rate estimation is performed by analyzing the color change that occurs in the skin during each cardiovascular cycle.
This color change that is too small to be seen with the naked eye, was converted into a signal of size (fps * video length) by recording the green channel averages of the 3 regions of the face in each frame in the given video. The frequency spectrum of the signal, which was denoised with the signal preprocessing steps, is analyzed and the pulse is detected.


![Screenshot](Workflow.jpg)


**Running a Demo**
