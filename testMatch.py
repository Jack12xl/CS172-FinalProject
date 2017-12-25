import numpy as np
import cv2
from bagOfWordsDownSampled import kmeans
def templateContours():
	template=[]
	poseId=0
	for poseId in range(10):
		templateImg=cv2.imread('./template%d.jpg'%poseId)
		imgray=cv2.cvtColor(templateImg,cv2.COLOR_BGR2GRAY)
		ret,thresh=cv2.threshold(imgray,20,255,0)
		_,contour,hierachy=cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
		if len(contour) != 0:
			cv2.drawContours(templateImg,contour,-1,(0,255,0),3)
			c=max(contour,key=cv2.contourArea)
		template.append(c)
		cv2.imwrite("contour%d.jpg"%poseId,templateImg)
		return template

def SIFTtest(img,kmeans):
	img=cv2.resize(img,(180,120))
	gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	sift=cv2.xfeatures2d.SIFT_create()
	kp,des=sift.detectAndCompute(gray,None)
	des=des.T/des.sum(axis=1,dtype='float')
	print(des.shape)
	labels=kmeans.predict(des.T)
	print(labels)
	return des,labels

def getTestBoWVector(labels):
	BowVectors=np.zeros((1,7),dtype=int)
	unique,counts=np.unique(labels,return_counts=True)
	index=np.array(range(counts.shape[0]))
	BowVectors[0,unique]=counts[index]
	return BowVectors

img=cv2.imread('template1.jpg')
descriptor,labels=SIFTtest(img,kmeans)
BoWVector=getTestBoWVector(labels)
print(BoWVector)
