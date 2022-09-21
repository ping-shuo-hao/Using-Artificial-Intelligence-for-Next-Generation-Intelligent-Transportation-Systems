import cv2
import numpy as np

def getTrajectory(line):
    temp=""
    for char in line:
        if char!=" ":
            temp=temp+char
    arr=[]
    num=0
    po=1
    xValue=""
    yValue=""
    for char in temp:
        if char=='[':
            num+=1
        if char==']':
            num-=1
            po=1
            if num==1:
                x=int(xValue)
                y=int(yValue)
                arr.append([x,y])
            xValue=""
            yValue=""
        if num==2:
            if char == ',':
                po = 2
            if char.isdigit():
                if po==1:
                    xValue=xValue+char
                else:
                    yValue=yValue+char
    return arr

def drawtrajctory(trajectory,image,color):
    if len(trajectory)<2:
        return image
    for i in range(1,len(trajectory)):
        start=trajectory[i-1]
        end=trajectory[i]
        image=cv2.line(image,(start[0],start[1]),(end[0],end[1]),color,3)
    return image

def drawdots(trajectory,image,color):
    for i in range(0,len(trajectory)):
        point=trajectory[i]
        image=cv2.circle(image,point,4,color,-1)
    return image

def randomcolor():
    a=int(np.random.choice(range(256), size=1))
    b=int(np.random.choice(range(256), size=1))
    c=int(np.random.choice(range(256), size=1))
    return [a,b,c]

def draw(filelink,imagelink,outputname,sizex,sizey,mode):
    image = cv2.imread(imagelink)
    if len(image)!=sizey or len(image[0])!=sizex:
        image=cv2.resize(image,(sizex,sizey))
    f=open(filelink,"r")
    lines = f.readlines()
    for line in lines:
        trj = getTrajectory(line)
        if len(trj)>0:
            color=randomcolor()
            if mode:
                image=drawtrajctory(trj,image,color)
            else:
                image=drawdots(trj,image,color)
    cv2.imwrite(outputname,image)


#Ramsey:704*576
#Other:1920*1080

imagefile="Ramsey-test.jpg"
readfile="read_Ramsey.txt"
output="result2.jpg"
#draw(readfile,imagefile,output,704,576,True)
image = cv2.imread(imagefile)
print(len(image))
print(len(image[0]))
