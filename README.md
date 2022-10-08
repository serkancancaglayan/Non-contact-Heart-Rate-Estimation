# Non-contact-Heart-Rate-Estimation

**Contributors**

>Nurullah Calik ncalik.edu@gmail.com

>Semih Demir sdemir1414@gmail.com

>Serkan Can Caglayan serkan.can.caglayan99@gmail.com 

In this study Non-contact heart rate estimation is performed by analyzing the color change that occurs in the skin during each cardiovascular cycle. This color change that is too small to be seen with the naked eye, was converted into a signal of size (fps * video length) by recording the green channel averages of the 3 regions of the face in each frame in the given video. The frequency spectrum of the signal, which was denoised with the signal preprocessing steps, is analyzed and the pulse is detected.


***This study can be used for educational or research purposes only.

![Screenshot](Workflow.jpg)

**Running a Demo**
```
$ git clone https://github.com/serkancancaglayan/Non-contact-Heart-Rate-Estimation.git
$ cd Non-contact-Heart-Rate-Estimation
$ pip install -r requirements.txt
$ python3 runDemo.py --videoSource <Video path or camera index>, --duration <Length of video clip for heart rate detection(15 is recommended), --plot <1 or 0>
```


