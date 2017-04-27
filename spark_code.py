from pyspark import SparkContext
import cv2
import numpy as np
import math
import sys
from operator import add
import mapreduce as mp

sys.path.append('/usr/local/lib/python2.7/site-packages')

def blockshaped(arr, nrows, ncols):
    h, w = arr.shape
    return (arr.reshape(h//nrows, nrows, -1, ncols)
               .swapaxes(1,2)
               .reshape(-1, nrows, ncols))


def mapreduce_to_file(sc, mag, angle, noOfRowInBlock, noOfColInBlock, xBlockSize, yBlockSize):

    # sc = SparkContext("local", "Simple App")
    # convert array to numpy array
    mag = np.asarray(mag)
    angle = np.asarray(angle)

    # divide array into blocks
    mag = blockshaped(mag, 20, 20)
    angle = blockshaped(angle, 20, 20)

    # Parallelize numpy array of blocks
    mag_rdd = sc.parallelize(mag)
    angle_rdd = sc.parallelize(angle)

    R = noOfRowInBlock
    C = noOfColInBlock

    # Execute mapper function (compute averages) on RDDs
    mag_blocks = mag_rdd.map(mp.mapper)
    angle_blocks = angle_rdd.map(mp.mapper)

    opFlowOfBlocks = np.zeros((xBlockSize, yBlockSize, 2))

    i = 0
    j = 0
    for x in mag_blocks.toLocalIterator():
        opFlowOfBlocks[i][j][0] = x
        j = j + 1
        if j >= yBlockSize:
            j = 0
            i = i + 1

    i = 0
    j = 0
    for x in angle_blocks.toLocalIterator():
        opFlowOfBlocks[i][j][1] = x
        j = j + 1
        if j >= yBlockSize:
            j = 0
            i = i + 1
    
    # Stop the Spark Context
    # sc.stop()

    return opFlowOfBlocks


def opflow_mapreduce(sc, mag, angle, noOfRowInBlock, noOfColInBlock, xBlockSize, yBlockSize):

    # convert array to numpy array
    mag = np.asarray(mag)
    angle = np.asarray(angle)

    a, b = mag.shape

    mag_values = np.zeros((a*b,3))
    ang_values = np.zeros((a*b,3))

    i = 0
    for index, value in np.ndenumerate(mag):
    	mag_values[i][0] = int(index[0])
    	mag_values[i][1] = int(index[1])
    	mag_values[i][2] = value
    	i = i + 1

    i = 0
    for index, value in np.ndenumerate(angle):
        ang_values[i][0] = int(index[0])
        ang_values[i][1] = int(index[1])
        ang_values[i][2] = value
        i = i + 1

    rdd_mag_values = sc.parallelize(mag_values)
    rdd_mag_values = rdd_mag_values.map(mp.generate_key_value)
    rdd_mag_values = rdd_mag_values.reduceByKey(add)

    rdd_ang_values = sc.parallelize(ang_values)
    rdd_ang_values = rdd_ang_values.map(mp.generate_key_value)
    rdd_ang_values = rdd_ang_values.reduceByKey(add)

    opFlowOfBlocks = np.zeros((xBlockSize, yBlockSize, 2))

    for item in rdd_mag_values.collect():
        key = item[0]
        val = item[1]
        col = key%100000
        row = key/100000
        opFlowOfBlocks[row][col][0] = val

    for item in rdd_ang_values.collect():
        key = item[0]
        val = item[1]
        col = key%100000
        row = key/100000
        opFlowOfBlocks[row][col][1] = val

    # thefile = open('opflowofblocks mag new approach.txt', 'w')
    # for A in opFlowOfBlocks:
    #     for B in A:
    #         thefile.write("%s\n" % B[0])
    # thefile.close()
    # sc.stop()
    # sys.exit()

    return opFlowOfBlocks
