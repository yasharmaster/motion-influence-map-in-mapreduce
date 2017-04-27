import motionInfuenceGenerator as mig
from pyspark import SparkContext
import createMegaBlocks as cmb
import numpy as np
import cv2
import time

def square(a):
    return (a**2)

def diff(l):
    return (l[0] - l[1])

def showUnusualActivities(unusual, vid, noOfRows, noOfCols, n):

    unusualFrames = unusual.keys()
    unusualFrames.sort()
    print(unusualFrames)
    cap = cv2.VideoCapture(vid)
    ret, frame = cap.read()
    rows, cols = frame.shape[0], frame.shape[1]
    rowLength = rows/(noOfRows/n)
    colLength = cols/(noOfCols/n)
    #print("Block Size ",(rowLength,colLength))
    count = 0
    screen_res = 980, 520
    scale_width = screen_res[0] / 320
    scale_height = screen_res[1] / 240
    scale = min(scale_width, scale_height)
    window_width = int(320 * scale)
    window_height = int(240 * scale)

    cv2.namedWindow('Unusual Frame',cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Unusual Frame',window_width, window_height)
    while (ret == True):
        print(count)
        ret, uFrame = cap.read()
        '''
        if(count <= 475):
            
            count += 1
            continue
        
        elif((count-475) in unusualFrames):
        '''
			
		
        if(count in unusualFrames):
			for blockNum in unusual[count]:
				print(blockNum)
				x1 = blockNum[1] * rowLength
				y1 = blockNum[0] * colLength
				x2 = (blockNum[1]+1) * rowLength
				y2 = (blockNum[0]+1) * colLength
				cv2.rectangle(uFrame,(x1,y1),(x2,y2),(0,0,255),1)
			cv2.imshow('Unusual Frame',uFrame)
			cv2.waitKey(0)
			cv2.destroyAllWindows()
        '''
        #  print("Unusual frame number ",str(count))
        if(count == 622):
            break
        '''
			
        count += 1


def constructMinDistMatrix(megaBlockMotInfVal,codewords, noOfRows, noOfCols, vid):
    threshold = 5.83682407063e-05
    n = 2
    minDistMatrix = np.zeros((len(megaBlockMotInfVal[0][0]),(noOfRows/n),(noOfCols/n)))
    for index,val in np.ndenumerate(megaBlockMotInfVal[...,0]):
        eucledianDist = []
        for codeword in codewords[index[0]][index[1]]:
            #print("haha")
            temp = [list(megaBlockMotInfVal[index[0]][index[1]][index[2]]),list(codeword)]
            #print("Temp",temp)
            dist = np.linalg.norm(megaBlockMotInfVal[index[0]][index[1]][index[2]]-codeword)
            #print("Dist ",dist)
            eucDist = (sum(map(square,map(diff,zip(*temp)))))**0.5
            #eucDist = (sum(map(square,map(diff,zip(*temp)))))
            eucledianDist.append(eucDist)
            #print("My calc ",sum(map(square,map(diff,zip(*temp)))))
        #print(min(eucledianDist))
        minDistMatrix[index[2]][index[0]][index[1]] = min(eucledianDist)
    unusual = {}
    for i in range(len(minDistMatrix)):
        if(np.amax(minDistMatrix[i]) > threshold):
            unusual[i] = []
            for index,val in np.ndenumerate(minDistMatrix[i]):
                #print("MotInfVal_train",val)
                if(val > threshold):
                        unusual[i].append((index[0],index[1]))
    
    #showUnusualActivities(unusual, vid, noOfRows, noOfCols, n)


def test_video(vid, sc):
    '''
        calls all methods to test the given video
        
    '''
    MotionInfOfFrames, rows, cols = mig.getMotionInfuenceMap(vid, sc)
    megaBlockMotInfVal = cmb.createMegaBlocks(MotionInfOfFrames, rows, cols)

    codewords = np.load("/home/yash/work/project/unusual/code/codewords.npy")
	#print("codewords",codewords)
    listOfUnusualFrames = constructMinDistMatrix(megaBlockMotInfVal,codewords,rows, cols, vid)
    return
    
if __name__ == '__main__':
    '''
        defines training set and calls trainFromVideo for every vid
    '''

    testSet = [r"/home/yash/work/project/unusual/code/2_test2.avi"]
    sc = SparkContext("local", "MotionInfluenceMap")
    start_time = time.time()
    for video in testSet:
        test_video(video, sc)
    print("--- %s seconds ---" % (time.time() - start_time))
    sc.stop()
    print "Done"
