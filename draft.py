import cv2
import numpy as np

# from matplotlib import pyplot as plt
#sift
def siftExtrator(poseName,poseId):
	print(poseName)
	descriptors=np.zeros((1,128))
	for i in range(10):
		img=cv2.imread('./hand-reader-dataset/%c/0%d00%d.jpg'%(poseName,poseId,i))
		# print('./hand-reader-dataset/%c/0000%d.jpg'%(poseId,i))
		gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
		sift=cv2.xfeatures2d.SIFT_create()
		kp,des=sift.detectAndCompute(gray,None)
		des=des/des.sum(axis=0,dtype='float')
		descriptors=np.concatenate((descriptors,des))
		# print(descriptors.shape)
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

		img=cv2.drawKeypoints(gray,kp,None,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
		cv2.imwrite('./test/sift%c%d.jpg'%(poseName,i),img)
	# print(descriptors.shape)
	return descriptors

def Kmeans(descriptor):
	descriptor = np.float32(descriptor)
	# Define criteria = ( type, max_iter = 10 , epsilon = 1.0 )
	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
	# Set flags use kmeans++ initialization
	flags = cv2.KMEANS_PP_CENTERS
	# Apply KMeans
	compactness,labels,centers = cv2.kmeans(descriptor,50,None,criteria,10,flags)
	# compactness gives the error, labels gives the cluster each point is in
	# centers gives the value of center
	NumberOfNodes = np.array([list(labels).count(i) for i in range(50) ])
	# NumberOfNodes gives number of nodes in each cluster.
	# print(compactness)
	# return NumberOfNodes
	return centers


bagOfWords=[]
for i in range(4):
	Descriptor = siftExtrator(chr(ord('A')+i),i)
	bagOfWords.append(Kmeans(Descriptor))
Descriptor = siftExtrator('G',4)
bagOfWords.append(Kmeans(Descriptor))
Descriptor = siftExtrator('H',5)
bagOfWords.append(Kmeans(Descriptor))
Descriptor = siftExtrator('I',6)
bagOfWords.append(Kmeans(Descriptor))
Descriptor = siftExtrator('L',7)
bagOfWords.append(Kmeans(Descriptor))
Descriptor = siftExtrator('V',8)
bagOfWords.append(Kmeans(Descriptor))
Descriptor = siftExtrator('Y',9)
bagOfWords.append(Kmeans(Descriptor))

bagOfWords = np.array(bagOfWords)
print bagOfWords.shape



