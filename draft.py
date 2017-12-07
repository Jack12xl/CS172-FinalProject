import cv2
import numpy as np
# from matplotlib import pyplot as plt
#sift
def siftExtrator(poseId):
	descriptors=np.zeros((1,128))
	for i in range(10):
		img=cv2.imread('./hand-reader-dataset/%c/0000%d.jpg'%(poseId,i))
		# print('./hand-reader-dataset/%c/0000%d.jpg'%(poseId,i))
		gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
		sift=cv2.xfeatures2d.SIFT_create()
		kp,des=sift.detectAndCompute(gray,None)
		descriptors=np.concatenate((descriptors,des))
		# print(descriptors.shape)
		img=cv2.drawKeypoints(gray,kp,None,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
		cv2.imwrite('./test/sift%c%d.jpg'%(poseId,i),img)
	for i in range(11,50):
		img=cv2.imread('./hand-reader-dataset/%c/000%d.jpg'%(poseId,i))
		# print('./hand-reader-dataset/%c/000%d.jpg'%(poseId,i))
		gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
		sift=cv2.xfeatures2d.SIFT_create()
		kp,des=sift.detectAndCompute(gray,None)
		descriptors=np.concatenate((descriptors,des))

		img=cv2.drawKeypoints(gray,kp,None,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
		cv2.imwrite('./test/sift%c%d.jpg'%(poseId,i),img)
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
	compactness = np.array(compactness)
	labels = np.array(labels)
	centers = np.array(centers)
	print compactness
	return centers



Descriptor = siftExtrator('A')

Kmeans(Descriptor)


