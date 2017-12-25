import numpy as np
import cv2
from sklearn.externals import joblib
from sklearn.preprocessing import Imputer
clf = joblib.load("./mode/svm.m")
kmeans=joblib.load("./mode/kmeans.pkl")

def SIFTtest(gray,kmeans,i):
    # gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # cv2.imwrite('/pose/data%d.jpg'%i,gray)
    sift=cv2.xfeatures2d.SIFT_create()
    kp,des=sift.detectAndCompute(gray,None)
    if kp == []:
        return None,None

    des= Imputer().fit_transform(des.T/des.sum(axis=1,dtype='float'))
    img=cv2.drawKeypoints(gray,kp,None,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    labels=kmeans.predict(des.T)

    return des,labels

def getTestBoWVector(labels):
    BowVectors=np.zeros((1,210),dtype=int)
    unique,counts=np.unique(labels,return_counts=True)
    index=np.array(range(counts.shape[0]))
    BowVectors[0,unique]=counts[index]
    # print(BowVectors)
    return BowVectors

def skinDetection(img):
    blur = cv2.blur(img,(5,5))
    #Convert to HSV color space
    hsv = cv2.cvtColor(blur,cv2.COLOR_BGR2HSV)
    #Create a binary image with where white will be skin colors and rest is black
    mask2 = cv2.inRange(hsv,np.array([0,70,50]),np.array([20,190,255]))
    #Kernel matrices for morphological transformation
    kernel_square = np.ones((11,11),np.uint8)
    kernel_ellipse= cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    #Perform morphological transformations to filter out the background noise
    #Dilation increase skin color area
    #Erosion increase skin color area
    dilation = cv2.dilate(mask2,kernel_ellipse,iterations = 1)
    erosion = cv2.erode(dilation,kernel_square,iterations = 1)
    dilation2 = cv2.dilate(erosion,kernel_ellipse,iterations = 1)
    filtered = cv2.medianBlur(dilation2,5)
    kernel_ellipse= cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(8,8))
    dilation2 = cv2.dilate(filtered,kernel_ellipse,iterations = 1)
    kernel_ellipse= cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    dilatin3 = cv2.dilate(filtered,kernel_ellipse,iterations = 1)
    median = cv2.medianBlur(dilation2,5)
    ret,thresh = cv2.threshold(median,10,1,cv2.THRESH_BINARY)
    mul = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return mul,thresh

def roiExtraction(img,thresh):
    pic = np.zeros((120,180),np.uint8)
    #Find contours of the filtered frame
    _, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    #Find Max contour area (Assume that hand is in the frame)
    max_area=100
    ci=0
    for i in range(len(contours)):
        cnt=contours[i]
        area = cv2.contourArea(cnt)
        if(area>max_area):
            max_area=area
            ci=i
    #Largest area contour
    if len(contours)== 0:
        return pic
    cnts = contours[ci]
    #Find convex hull
    hull = cv2.convexHull(cnts)
    # Print bounding rectangle
    x,y,w,h = cv2.boundingRect(cnts)
    img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
    roi = mul[y:y+h,x:x+w]

    # roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    cv2.imshow('pic',pic)
    if w/180 < h/120 :
        width = int(120*w/h)
        roi = cv2.resize(roi,(width,120),interpolation=cv2.INTER_CUBIC)
        height = h
        pic[0:120,90-int(width/2):90-int(width/2)+width] = roi
    return pic
