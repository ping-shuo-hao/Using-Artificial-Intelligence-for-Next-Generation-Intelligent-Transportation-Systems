import cv2
import pytesseract
import datetime
import re
import numpy as np

pytesseract.pytesseract.tesseract_cmd=r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe' #Please change the path to the directory where tesseract.exe is in your local machine

#created by Shuohao Ping
#return a string of timestamp in the following format: mm/dd/yyyy hh:mm:ss
#if the return result is "N/A", then program can't get a valid time string from the frame.

#path: the directory of the video. Could be a link or a path.
#frame_number: the frame number that the program should get the time from
# x: x coordinate of the upper left point of the bounding box for the timestamp
# y: y coordinate of the upper left point of the bounding box
# w: width of the bounding box
# h: height for the bounding box

def text_split(text):
    text=text.split()
    day=None
    time=None
    for element in text:
        if day==None:
            result_day =re.match(r'.*((\d\d\d\d(/|-)\d\d(/|-)\d\d)|(\d\d(/|-)\d\d(/|-)\d\d\d\d)).*',element)
            if result_day!=None:
                day=result_day.group(1)
        if time==None:
           result_time=re.match(r'.*(\d{2}:\d{2}:\d{2}).*',element)
           if result_time!=None:
               time=result_time.group(1)
    if day==None or time==None:
        return "N/A"
    year=None
    month=None
    d=None
    temp=re.search('(\d{4})(/|-)(\d{2})(/|-)(\d{2}).*',day)
    if temp!=None:
        year=temp.group(1)
        month=temp.group(3)
        d=temp.group(5)
    else:
        temp = re.search('(\d{2})(/|-)(\d{2})(/|-)(\d{4}).*', day)
        year=temp.group(5)
        month=temp.group(1)
        d=temp.group(3)
    day='/'.join([month,d,year])
    if time[0]=='9':
        time='0'+time[1:]
    timestamp=' '.join([day,time])
    try:
        datetime.datetime.strptime(timestamp,'%m/%d/%Y %H:%M:%S')
    except ValueError:
        return 'N/A'
    return timestamp

def readTime(path,frame_number,x,y,w,h):
    #change coordinate
    x1=x
    x2=x+w
    y1=y
    y2=y+h
    #load video
    video = cv2.VideoCapture(path)

    #check frame_number<=max(number  of  frame in the video)
    max_length=int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_number>max_length+1 or frame_number<0:
        return 'frame number not valid' # frame number is larger than the length of the video or frame number less than 0

    #jump to target frame
    video.set(1, frame_number)

    #load frame
    ret, frame = video.read()
    if not ret:
        return "N/A"

    #locate timestamp
    frame=cv2.rectangle(frame,(x,x+w),(y,y+h),(255,0,0),2)
    print(x,x+w,y,y+h)
    cv2.imwrite('output.jpg',frame)
    crop_img = frame[y1:y2,x1:x2]

    #convert to gray image
    crop_img=cv2.cvtColor(crop_img,cv2.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(crop_img, config='--psm 6 --dpi 300')
    re=text_split(text)
    if re=='N/A':
        crop_img = np.invert(crop_img)
#        cv2.imwrite("T.jpg", crop_img)
        text = pytesseract.image_to_string(crop_img, config='--psm 6 --dpi 300')
        re = text_split(text)
    return re
