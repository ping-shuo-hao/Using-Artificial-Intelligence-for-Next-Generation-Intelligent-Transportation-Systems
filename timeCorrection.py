import readTime
import cv2
import datetime


def datecorrection(path,x1,x2,y1,y2):
    video = cv2.VideoCapture(path)
    max_length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(video.get(cv2.CAP_PROP_FPS))
    seconds = int(max_length / fps)
    start,end= readTime.readTime(path,0,x1,y1,x2-x1,y2-y1),readTime.readTime(path,max_length-1,x1,y1,x2-x1,y2-y1)
    if start=='N/A' and end=='N/A':
        return 'N/A','N/A'
    elif start=='N/A':
        start=datetime.datetime.strptime(end,"%m/%d/%Y %H:%M:%S")-datetime.timedelta(seconds=seconds)
        start=start.strftime("%m/%d/%Y %H:%M:%S")
    elif end=='N/A':
        end=datetime.datetime.strptime(start,"%m/%d/%Y %H:%M:%S")+datetime.timedelta(seconds=seconds)
        end=end.strftime("%m/%d/%Y %H:%M:%S")
    return start,end
