import cv2
import numpy as np


def cordinates(image,line_parameters):
    slop, intercept = line_parameters
    y1=image.shape[0]
    y2=int(y1*0.6)
    x1=int((y1-intercept)/slop)
    x2 = int((y2 - intercept) / slop)
    return np.array([x1,y1,x2,y2])


def average_slope_of_intercept(image,lines):
    left_fit=[]
    right_fit=[]
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters=np.polyfit((x1,x2),(y1,y2),1)
        slope=parameters[0]
        intercept=parameters[1]
        if slope < 0:
            left_fit.append((slope,intercept))
        else:
            right_fit.append((slope,intercept))
    left_fit_avg=np.average(left_fit,axis=0)
    right_fit_avg=np.average(right_fit,axis=0)
    left_line=cordinates(image,left_fit_avg)
    right_line=cordinates(image,right_fit_avg)
    return np.array([left_line,right_line])


def canny(road):
    gray = cv2.cvtColor(road, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    canny = cv2.Canny(blur, 100, 50)
    return canny

def displaylines(image,lines):
   line_img=np.zeros_like(image)
   if line_img is not None:
       for line in lines:
           x1,y1,x2,y2=line.reshape(4)
           cv2.line(line_img,(x1,y1),(x2,y2),(255,0,0),10)
   return line_img



def roi(image):
    h = image.shape[0]
    region = np.array([[(200, h), (1100, h), (550, 250)]])
    mask=np.zeros_like(image)
    cv2.fillPoly(mask, region, 255)
    mask_img = cv2.bitwise_and(mask, image)
    return mask_img


#image = cv2.imread('test_image.jpg')
#lane = np.copy(image)
# = canny(lane)
#cropped_img=roi(c)
#lines=cv2.HoughLinesP(cropped_img, 2,np.pi/180,100,minLineLength=40,maxLineGap=5)
#average_line=average_slope_of_intercept(lane,lines)
#ane_img=displaylines(lane,average_line)
#combo_img=cv2.addWeighted(lane_img,1,lane,0.5,1)
#cv2.imshow('result', combo_img)
#cv2.waitKey(0)
cap=cv2.VideoCapture('test2.mp4')

while(cap.isOpened()):
    check, frame = cap.read()
    lane = np.copy(frame)
    c = canny(frame)
    cropped_img = roi(c)
    lines = cv2.HoughLinesP(cropped_img, 2, np.pi / 180, 100, minLineLength=40, maxLineGap=5)
    average_line = average_slope_of_intercept(lane, lines)
    lane_img = displaylines(lane, average_line)
    combo_img = cv2.addWeighted(lane_img, 1, lane, 0.5, 1)
    cv2.imshow('result', combo_img)
    key= cv2.waitKey(1)
    if key==27:
        break
cap.release()
cv2.destroyAllWindows()


