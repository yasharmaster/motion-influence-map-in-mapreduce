import math
import itertools
import numpy as np
import cv2



def createMegaBlocks(motionInfoOfFrames,noOfRows,noOfCols):
   
    n = 2
    megaBlockMotInfVal = np.zeros(((noOfRows/n),(noOfCols/n),len(motionInfoOfFrames),8))
    
    frameCounter = 0
    
    for frame in motionInfoOfFrames:
        
        for index,val in np.ndenumerate(frame[...,0]):
            
            temp = [list(megaBlockMotInfVal[index[0]/n][index[1]/n][frameCounter]),list(frame[index[0]][index[1]])]
           
            megaBlockMotInfVal[index[0]/n][index[1]/n][frameCounter] = np.array(map(sum, zip(*temp)))

        frameCounter += 1
    print(((noOfRows/n),(noOfCols/n),len(motionInfoOfFrames)))
    return megaBlockMotInfVal

def kmeans(megaBlockMotInfVal):
    #k-means
    cluster_n = 5
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    flags = cv2.KMEANS_RANDOM_CENTERS
    codewords = np.zeros((len(megaBlockMotInfVal),len(megaBlockMotInfVal[0]),cluster_n,8))
    #codewords = []
    #print("Mega blocks ",megaBlockMotInfVal)
    for row in range(len(megaBlockMotInfVal)):
        for col in range(len(megaBlockMotInfVal[row])):
            #print("megaBlockMotInfVal ",(row,col),"/n/n",megaBlockMotInfVal[row][col])
            
            ret, labels, cw = cv2.kmeans(np.float32(megaBlockMotInfVal[row][col]), cluster_n, None, criteria,10,flags)
            #print(ret)
            #if(ret == False):
            #    print("K-means failed. Please try again")
            codewords[row][col] = cw
            
    return(codewords)


def calcOptFlowOfBlocks(mag,angle,grayImg):
    '''Takes an image (gray scale) and a flow matrix as input. Divides image into blocks and calculates Optical Flow of each block '''
    '''calculate number of rows and columns in the matrix of the image'''
    rows = grayImg.shape[0]
    cols = grayImg.shape[1]
    noOfRowInBlock = 20
    noOfColInBlock = 20
    ''' calculate the number of rows of blocks and columns of blocks in the frame '''
    xBlockSize = rows / noOfRowInBlock
    yBlockSize = cols / noOfColInBlock
    '''To calculate the optical flow of each block'''

    '''declare an array initialized to 0 of the size of the number of blocks'''
    
    opFlowOfBlocks = np.zeros((xBlockSize,yBlockSize,2))
    
    for index,value in np.ndenumerate(mag):
        opFlowOfBlocks[index[0]/noOfRowInBlock][index[1]/noOfColInBlock][0] += mag[index[0]][index[1]]
        opFlowOfBlocks[index[0]/noOfRowInBlock][index[1]/noOfColInBlock][1] += angle[index[0]][index[1]]

    centreOfBlocks = np.zeros((xBlockSize,yBlockSize,2))
    for index,value in np.ndenumerate(opFlowOfBlocks):
        opFlowOfBlocks[index[0]][index[1]][index[2]] = float(value)/(noOfRowInBlock*noOfColInBlock)
        val = opFlowOfBlocks[index[0]][index[1]][index[2]]

        if(index[2] == 1):
            angInDeg = math.degrees(val)
            if(angInDeg > 337.5):
                k = 0
            else:
                q = angInDeg//22.5
                a1 = q*22.5
                q1 = angInDeg - a1
                a2 = (q+2)*22.5
                q2 =  a2 - angInDeg
                if(q1 < q2):
                    k = int(round(a1/45))
                else:
                    k = int(round(a2/45))        
            opFlowOfBlocks[index[0]][index[1]][index[2]] = k
            theta = val
            
        
        
        
        if(index[2] == 0):
            r = val
            x = ((index[0] + 1)*noOfRowInBlock)-(noOfRowInBlock/2)
            y = ((index[1] + 1)*noOfColInBlock)-(noOfColInBlock/2)
            centreOfBlocks[index[0]][index[1]][0] = x
            centreOfBlocks[index[0]][index[1]][1] = y
    return opFlowOfBlocks,noOfRowInBlock,noOfColInBlock,noOfRowInBlock*noOfColInBlock,centreOfBlocks,xBlockSize,yBlockSize


def getThresholdDistance(mag,blockSize):
    return mag*blockSize

def getThresholdAngle(ang):
    tAngle = float(math.pi)/2
    return ang+tAngle,ang-tAngle

def getCentreOfBlock(blck1Indx,blck2Indx,centreOfBlocks):
    x1 = centreOfBlocks[blck1Indx[0]][blck1Indx[1]][0]
    y1 = centreOfBlocks[blck1Indx[0]][blck1Indx[1]][1]
    x2 = centreOfBlocks[blck2Indx[0]][blck2Indx[1]][0]
    y2 = centreOfBlocks[blck2Indx[0]][blck2Indx[1]][1]
    slope = float(y2-y1)/(x2-x1) if (x1 != x2) else float("inf")
    return (x1,y1),(x2,y2),slope


def calcEuclideanDist((x1,y1),(x2,y2)):
    dist = float(((x2-x1)**2 + (y2-y1)**2)**0.5)
    return dist
    
def angleBtw2Blocks(ang1,ang2):
    if(ang1-ang2 < 0):
        ang1InDeg = math.degrees(ang1)
        ang2InDeg = math.degrees(ang2)
        return math.radians(360 - (ang1InDeg-ang2InDeg))
    return ang1 - ang2

def motionInMapGenerator(opFlowOfBlocks,blockSize,centreOfBlocks,xBlockSize,yBlockSize):
    global frameNo
    motionInfVal = np.zeros((xBlockSize,yBlockSize,8))
    for index,value in np.ndenumerate(opFlowOfBlocks[...,0]):
        Td = getThresholdDistance(opFlowOfBlocks[index[0]][index[1]][0],blockSize)
        k = opFlowOfBlocks[index[0]][index[1]][1]
        posFi, negFi =  getThresholdAngle(math.radians(45*(k)))
        
        for ind,val in np.ndenumerate(opFlowOfBlocks[...,0]):
            if(index != ind):
                (x1,y1),(x2,y2), slope = getCentreOfBlock(index,ind,centreOfBlocks)
                euclideanDist = calcEuclideanDist((x1,y1),(x2,y2))
        
                if(euclideanDist < Td):
                    angWithXAxis = math.atan(slope)
                    angBtwTwoBlocks = angleBtw2Blocks(math.radians(45*(k)),angWithXAxis)
        
                    if(negFi < angBtwTwoBlocks and angBtwTwoBlocks < posFi):
                        motionInfVal[ind[0]][ind[1]][int(opFlowOfBlocks[index[0]][index[1]][1])] += math.exp(-1*(float(euclideanDist)/opFlowOfBlocks[index[0]][index[1]][0]))
    #print("Frame number ", frameNo)
    frameNo += 1
    return motionInfVal


def getMotionInfuenceMap(vid):
    global frameNo
    
    frameNo = 0
    cap = cv2.VideoCapture(vid)
    ret, frame1 = cap.read()
    rows, cols = frame1.shape[0], frame1.shape[1]
    print(rows,cols)
    prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    motionInfOfFrames = []
    count = 0
    while 1:
        '''
        #if(count <= 475 or (count > 623 and count <= 1300)):
        if(count < 475):
            ret, frame2 = cap.read()
            prvs = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
            count += 1
            continue
        '''
        
        #if((count < 1451 and count <= 623)):
        '''
        if(count < 475):    
            ret, frame2 = cap.read()
            prvs = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
            count += 1
            continue
        '''
        print(count)
        ret, frame2 = cap.read()
        if (ret == False):
            break
        next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
       
        
        mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
        
        
        prvs = next
        opFlowOfBlocks,noOfRowInBlock,noOfColInBlock,blockSize,centreOfBlocks,xBlockSize,yBlockSize = calcOptFlowOfBlocks(mag,ang,next)
        motionInfVal = motionInMapGenerator(opFlowOfBlocks,blockSize,centreOfBlocks,xBlockSize,yBlockSize)
        motionInfOfFrames.append(motionInfVal)
        
        #if(count == 622):
        #    break
        count += 1
    return motionInfOfFrames, xBlockSize,yBlockSize


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
    print("Block Size ",(rowLength,colLength))
    count = 0
    screen_res = 980, 520
    scale_width = screen_res[0] / 320
    scale_height = screen_res[1] / 240
    scale = min(scale_width, scale_height)
    window_width = int(320 * scale)
    window_height = int(240 * scale)

    cv2.namedWindow('Unusual Frame',cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Unusual Frame',window_width, window_height)
    while 1:
        print(count)
        ret, uFrame = cap.read()
        '''
        if(count <= 475):
            
            count += 1
            continue
        
        elif((count-475) in unusualFrames):
        '''
        if(count in unusualFrames):
            if (ret == False):
                break
            for blockNum in unusual[count]:
                print(blockNum)
                x1 = blockNum[1] * rowLength
                y1 = blockNum[0] * colLength
                x2 = (blockNum[1]+1) * rowLength
                y2 = (blockNum[0]+1) * colLength
                cv2.rectangle(uFrame,(x1,y1),(x2,y2),(0,0,255),1)
            print("Unusual frame number ",str(count))
        cv2.imshow('Unusual Frame',uFrame)
            
        cv2.waitKey(0)
            #cv2.destroyAllWindows()
        '''
        if(count == 622):
            break
        '''
        count += 1
def constructMinDistMatrix(megaBlockMotInfVal,codewords, noOfRows, noOfCols, vid):
    #threshold = 2.1874939946e-21
    #threshold = 0.00196777849633
    #threshold = 9.3985643749758953e-06
    #threshold = 0.439167467697
    #threshold = 0.021305195096797892
    #threshold = 3.35845489394e-07
    #threshold = 1.6586380629e-08
    #threshold = 0.000212282134156
    #threshold = 4.63266766923e-14
    #threshold = 7.29868038369e-06
    #threshold = 8.82926005091e-05
    #threshold = 7.39718222289e-14
    #threshold = 8.82926005091e-05
    #threshold = 0.0080168593265873295
    #threshold = 0.00511863986892
    #------------------------------------#
    threshold = 5.83682407063e-05
    #threshold = 3.37029584538e-07
    #------------------------------------#
    #threshold = 2.63426664698e-06
    #threshold = 1.91130257263e-08
    
    #threshold = 0.0012675861679
    #threshold = 1.01827939172e-05
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
    print(unusual)
    showUnusualActivities(unusual, vid, noOfRows, noOfCols, n)
    
def test_video(vid):
    '''
        calls all methods to test the given video
       
    '''
    print "Test video ", vid
    MotionInfOfFrames, rows, cols = getMotionInfuenceMap(vid)
    #np.save("videos\scene1\rows_cols_set1_p1_test_20-20_k5.npy",np.array([rows,cols]))
    #######print "Motion Inf Map ", len(MotionInfOfFrames)
    #numpy.save("MotionInfluenceMaps", np.array(MotionInfOfFrames), allow_pickle=True, fix_imports=True)
    megaBlockMotInfVal = createMegaBlocks(MotionInfOfFrames, rows, cols)
    ######rows, cols = np.load("rows_cols__set3_p2_test_40_k3.npy")
    #print(megaBlockMotInfVal)
    np.save("/home/yash/work/project/code/megaBlockMotInfVal_set1_p1_test_40-40_k5.npy",megaBlockMotInfVal)
    ######megaBlockMotInfVal = np.load("megaBlockMotInfVal_set3_p2_train_40_k7.npy")
    codewords = np.load("/home/yash/work/project/code/codewords_set1_p1_train_40-40_k5.npy")
    print("codewords",codewords)
    listOfUnusualFrames = constructMinDistMatrix(megaBlockMotInfVal,codewords,rows, cols, vid)
    return
    
if __name__ == '__main__':
    '''
        defines training set and calls trainFromVideo for every vid
    '''
    testSet = [r"/home/yash/work/project/unusual/code/videos/scene2/2_test2.avi"]
    for video in testSet:
        test_video(video)
    print "Done"
