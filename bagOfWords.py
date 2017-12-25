import cv2
import numpy as np


SIFTfeatures=[]
numEachPic=[]
# from matplotlib import pyplot as plt
#sift
def siftExtrator(poseName,poseId):
	print(poseName)
	featuresEveryPic=[]
	for i in range(10):
		img=cv2.imread('./hand-reader-dataset/%c/0%d00%d.jpg'%(poseName,poseId,i))
		# print('./hand-reader-dataset/%c/0000%d.jpg'%(poseId,i))
		gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
		sift=cv2.xfeatures2d.SIFT_create()
		kp,des=sift.detectAndCompute(gray,None)
		des=des/des.sum(axis=0,dtype='float')
		if i==0:
			descriptors=des
		else:
			descriptors=np.concatenate((descriptors,des))
		print(descriptors.shape)
		featuresEveryPic.append(des.shape[0])
		img=cv2.drawKeypoints(gray,kp,None,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
		cv2.imwrite('./test/sift%c%d.jpg'%(poseName,i),img)
	for i in range(11,50):
		img=cv2.imread('./hand-reader-dataset/%c/0%d0%d.jpg'%(poseName,poseId,i))
		# print('./hand-reader-dataset/%c/000%d.jpg'%(poseId,i))
		gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
		sift=cv2.xfeatures2d.SIFT_create()
		kp,des=sift.detectAndCompute(gray,None)
		des=des/des.sum(axis=0,dtype='float')
		descriptors=np.concatenate((descriptors,des))
		featuresEveryPic.append(des.shape[0])
		img=cv2.drawKeypoints(gray,kp,None,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
		cv2.imwrite('./test/sift%c%d.jpg'%(poseName,i),img)
	print(descriptors.shape)

	return descriptors, featuresEveryPic

def Kmeans(descriptor,num_clusters):
	descriptor = np.float32(descriptor)
	# Define criteria = ( type, max_iter = 10 , epsilon = 1.0 )
	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
	# Set flags use kmeans++ initialization
	flags = cv2.KMEANS_PP_CENTERS
	# Apply KMeans
	compactness,labels,centers = cv2.kmeans(descriptor,num_clusters,None,criteria,10,flags)
	# compactness gives the error, labels gives the cluster each point is in
	# centers gives the value of center
	# NumberOfNodes = np.array([list(labels).count(i) for i in range(50) ])
	# NumberOfNodes gives number of nodes in each cluster.
	# print(compactness)
	# return NumberOfNodes
	return labels,centers

def featureMap(pointer,poseName,poseId):
	global SIFTfeatures,numEachPic
	Descriptor,featuresEveryPic = siftExtrator(poseName,poseId)
	SIFTfeatures.append(Descriptor)
	pointer+=sum(featuresEveryPic)
	numEachPic.append(featuresEveryPic)
	return pointer

def readInAndMatch():
	pointer=0
	FeatureOfPose={}
	for i in range(4):
		FeatureOfPose[chr(ord('A')+i)]=featureMap(pointer,chr(ord('A')+i),i)
		pointer=FeatureOfPose[chr(ord('A')+i)]
	FeatureOfPose['G']=featureMap(pointer,'G',4)
	pointer=FeatureOfPose['G']
	FeatureOfPose['H']=featureMap(pointer,'H',5)
	pointer=FeatureOfPose['H']
	FeatureOfPose['I']=featureMap(pointer,'I',6)
	pointer=FeatureOfPose['I']
	FeatureOfPose['L']=featureMap(pointer,'L',7)
	pointer=FeatureOfPose['L']
	FeatureOfPose['V']=featureMap(pointer,'V',8)
	pointer=FeatureOfPose['V']
	FeatureOfPose['Y']=featureMap(pointer,'Y',9)
	pointer=FeatureOfPose['Y']
	allFeatures=np.concatenate(tuple(SIFTfeatures))
	return FeatureOfPose,allFeatures

def labeledPic(numEachPic,labels):
	ptr=0
	poseId=0
	dictPose={}
	for pose in numEachPic:
		BoWOfPose=[]
		for pic in pose:
			BowVectors=np.zeros((1,750),dtype=int)
			unique,counts=np.unique(labels[ptr:ptr+pic],return_counts=True)
			index=np.array(range(counts.shape[0]))
			BowVectors[0,unique]=counts[index]
			BoWOfPose.append(BowVectors)
			ptr+=pic
		dictPose[poseId]=BoWOfPose
		poseId+=1
	return dictPose


FeatureOfPose,allFeatures=readInAndMatch()
labels, Words =Kmeans(allFeatures,750)
print(labels.size)
print(Words.size)
dictPose=labeledPic(numEachPic,labels)
# print(dictPose)
