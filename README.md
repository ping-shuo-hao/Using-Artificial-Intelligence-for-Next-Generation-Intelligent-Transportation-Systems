# Using-Artificial-Intelligence-for-Next-Generation-Intelligent-Transportation-Systems
This is a group project that aims to detect trespassers in grade-crossing scenarios using computer vision and 
Artificial Intelligence. We process video streams from multiple locations and maintain a database of trespassing 
events from 2019 to 2022. I serve as a back-end engineer developing two Apps to help maintain the database 
and examine the performance of the AI in collaboration with other front-end engineers. I help to develop an 
App filtering out incorrect data in the database and an App to assist the validation of original video streams. My 
responsibility is the following:
## Time Correction
### Algorithm Introduction
 This algorithm is used to correct start & end time for events(trespassing,train,signal). Before running the code, please download both readTime.py and timeCorrection.py. In addition, you need to download PyTesseract library on you local machine. See https://towardsdatascience.com/read-text-from-image-with-one-line-of-python-code-c22ede074cac for downloading the library. 
 
 After download all necessary packages, please change line 7 of readTime.py to the directory where your PyTesseract is installed.
 
 To get start and end time, you need to import timeCorrection.py and call the function datecorrection(path,x1,x2,y1,y2). Input parameters are as follow:
path: video link or the directory of the video.
x1: x coordinate of the upper left point of the bounding box for the timestamp.
x2: x coordinate of the lower right point of the bounding box for the timestamp.
y1: y coordinate of the upper left point of the bounding box for the timestamp.
y2: y coordinate of the lower right point of the bounding box for the timestamp.


#### For Thomasville North/South, Ramsey, and Ashland, the coordinates are:

Thomasville North/South & Ashland: x1=0, x2=43, y1=0, y2=1300

Ramsey: x1=465, x2=488, y1=0, y2=703

### Return value
The function will return two strings. The first one is start time, and the second one is end time.All strings aare in the correct time format: Month/Day/Year Hour:Minute:Second. If both string are incorrect, the function would change the string to "N/A". If only one of the string is invalid, then the program would replace the incorrect value with a calculated the time using video time and the correct timestamp.
For example, the possible format for return value: ('06/04/2021 11:27:17', '06/04/2021 11:27:24'), or ('N/A', 'N/A').


## Legal Occupier
### Algorithm Introduction
This algorithm is used to determine whether a car/truck/person is a legal occupier. Please download both legalOccupier.py and worker_classifier.py. Also please install OpenCV library before running the code.

### Inputs
To find legal Occupier, please call the function findLegalOccupier(type,ROI,trajectory,link) in legalOccupier.py. The inputs are:

type: type of trespassing. Please use the 'type' column in the dataset as the input.

ROI: Coordinate for region of interests. The input should be a list of tuple. Here is the ROI for Thomasville North/South, Ramsey, and Ashland:

Ramsey:[(10,396),(86,574),(581,283),(453,268)]

Ashland:[(424,702),(555,872),(1588,484),(1513,468)]

Thomasville South: [(34,855),(1289,643),(1534,644),(398,1076)]

Thomasvile North: [(301,620),(1391,1055),(1734,761),(535,587)].

trajectory: trajectory of the trespasser. The input of be a list of list. Please use the 'trajectory' column in the dataset as the input.

link: video link or the directory of the video. Please use the 'clip' column in the dataset as the input.

### Output
The output value is either True or False.True if a trespasser is a legal occupier and false otherwise.


## Duplicate_filter
### Algorithm Introduction:
This code is used to eliminate duplicate data in the IGCT database using computer vision. The input of this algorithm is a list of video clips that are processed by IGCT tool. The output is the number of distinct tresspassers (vechile or person) in that video clip. Here is an example:

 ###Input: 
 IGCT video link:
"https://igct.s3.amazonaws.com/5fee675583dbb051a5aa4b16_1626185723.mp4"

Output:

Total number of objects:  3

image name: A_1.jpg  trajectory:  [(457, 293), (457, 293), (457, 293), (443, 294), (438, 294), (436, 294), (436, 294), (436, 294), (436, 295), (436, 295), (416, 292), (409, 290), (407, 290), (405, 291), (406, 291), (406, 291), (405, 291), (406, 291), (406, 291), (406, 291), (397, 289), (393, 287), (392, 287), (392, 287), (392, 287), (392, 287), (371, 285), (364, 284), (360, 284), (359, 284), (359, 284), (359, 284), (346, 283), (343, 282), (341, 282), (340, 282), (340, 283), (340, 283), (334, 280), (332, 279), (331, 279), (332, 278), (332, 278)] Appearance:  0-42 label: car

image name: A_2.jpg  trajectory:  [(236, 262), (236, 262), (236, 262), (236, 262), (246, 264), (250, 265), (250, 264), (251, 265), (251, 265), (251, 265), (251, 265), (261, 266), (265, 267), (266, 267), (267, 267), (267, 267), (267, 266), (267, 266), (267, 266), (279, 267), (285, 267), (286, 268), (287, 268), (287, 268), (287, 268), (286, 268), (305, 270), (313, 270), (315, 270), (317, 270), (317, 271), (316, 271), (316, 271), (328, 273), (333, 274), (334, 275), (335, 275), (334, 275), (334, 275), (377, 277), (392, 276)] Appearance:  65-105 label: truck

image name: A_3.jpg  trajectory:  [(398, 277), (400, 277), (400, 276), (401, 276), (396, 276), (394, 277), (393, 276), (392, 276), (393, 276), (392, 276), (393, 276), (400, 277), (403, 276), (404, 276), (404, 276), (404, 276), (405, 276), (404, 276), (404, 276), (404, 276)] Appearance:  106-125 label: truck

![alt text](https://github.com/ping-shuo-hao/Duplicate_filter/blob/main/images/A_1.jpg)
![alt text](https://github.com/ping-shuo-hao/Duplicate_filter/blob/main/images/A_2.jpg)
![alt text](https://github.com/ping-shuo-hao/Duplicate_filter/blob/main/images/A_3.jpg)

<pre>
    A_1.jpg                   A_2.jpg                       A_3.jpg
</pre>

### Required dependencies:

pip install textdistance

sudo apt-get update

sudo apt-get install tesseract-ocr

sudo apt-get install libtesseract-dev

pip install pytesseract


## trajectory_visulization
### Algorithm Introduction:
This code is used to draw trajectory on specific image using data generated by IGCT tool. Different trajectory will be represented by different color. 

