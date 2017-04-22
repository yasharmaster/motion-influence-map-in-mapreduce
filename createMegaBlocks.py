import sys
from pyspark.mllib.clustering import KMeans, KMeansModel
import cv2

import numpy as np

import math
import itertools


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
    cluster_n = 5
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    flags = cv2.KMEANS_RANDOM_CENTERS
    codewords = np.zeros((len(megaBlockMotInfVal),len(megaBlockMotInfVal[0]),cluster_n,8))

    thefile = open('codewords from cv2.txt', 'w')
    for row in range(len(megaBlockMotInfVal)):
        for col in range(len(megaBlockMotInfVal[row])):

            ret, labels, cw = cv2.kmeans(np.float32(megaBlockMotInfVal[row][col]), cluster_n, criteria,10,flags)
            #print (cw)
            codewords[row][col] = cw
            thefile.write("%s\n\n\n" % cw)
    
    thefile.close()
            
    #return(codewords)


def kmeans_map_reduce(megaBlockMotInfVal, sc):

    cluster_n = 5
    codewords = np.zeros((len(megaBlockMotInfVal),len(megaBlockMotInfVal[0]),cluster_n,8))

    #thefile = open('codewords from mapreduce.txt', 'w')
    #myfile = open('codewords from mapreduce array.txt', 'w')

    for row in range(len(megaBlockMotInfVal)):
        for col in range(len(megaBlockMotInfVal[row])):
            rdd = sc.parallelize(megaBlockMotInfVal[row][col])
            cw = KMeans.train(rdd, cluster_n, maxIterations=10, initializationMode="random")
            #thefile.write("%s\n\n" % cw.clusterCenters)
            cluster = cw.clusterCenters
            val = np.asarray(cluster)
            codewords[row][col] = val
            #myfile.write("%s\n\n" % val)
    
    #thefile.close()
    #myfile.close()
    return codewords