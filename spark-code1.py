from pyspark import SparkContext
import cv2
import numpy as np
import math
import sys
from operator import add

sys.path.append('/usr/local/lib/python2.7/site-packages')

def blockshaped(arr, nrows, ncols):
    h, w = arr.shape
    return (arr.reshape(h//nrows, nrows, -1, ncols)
               .swapaxes(1,2)
               .reshape(-1, nrows, ncols))

R = 0
C = 0

def mapper(input):
    sum = 0
    for row in range(len(input)):
        for col in range(len(input[0])):
            sum += input[row][col]
    sum /= 400.0
    return sum


def opflow_mapreduce(sc, mag, angle, noOfRowInBlock, noOfColInBlock, xBlockSize, yBlockSize):

    # sc = SparkContext("local", "Simple App")
    # convert array to numpy array
    mag = np.asarray(mag)
    angle = np.asarray(angle)

    a, b = mag.shape

    mag_values = np.zeros((a*b,3))

    i = 0
    for index, value in np.ndenumerate(mag):
    	mag_values[i][0] = index[0]
    	mag_values[i][1] = index[1]
    	mag_values[i][2] = value
    	i = i + 1

    # thefile = open('mag values.txt', 'w')
    # for i in mag_values:
    #     thefile.write("%d " % i[0])
    #     thefile.write("%d " % i[1])
    #     thefile.write("%f \n" % i[2])
    # thefile.close()   

    rdd_mag_values = sc.parallelize(mag_values).map(lambda x: (int(x[0]/noOfRowInBlock)*16 + int(x[1]/noOfColInBlock), x[2])).reduceByKey(add).map(lambda x: x/400)

    count = 10
    for item in rdd_mag_values.collect():
    	print (item)
    	count = count - 1
    	if count < 0:
    		break

    # thefile = open('mag values rdd.txt', 'w')
    # for item in rdd_mag_values.collect():
    #     thefile.write("%s\n" % item)
    # thefile.close()

    sc.stop()
    sys.exit()

    # divide array into blocks
    mag = blockshaped(mag, 20, 20)
    angle = blockshaped(angle, 20, 20)

    # Parallelize numpy array of blocks
    mag_rdd = sc.parallelize(mag)
    angle_rdd = sc.parallelize(angle)

    R = noOfRowInBlock
    C = noOfColInBlock

    # Execute mapper function (compute averages) on RDDs
    mag_blocks = mag_rdd.map(mapper)
    angle_blocks = angle_rdd.map(mapper)

    # print average magnitudes and angles in a file for testing
    # thefile = open('spark mag averages.txt', 'w')
    # for item in mag_blocks.collect():
    #     thefile.write("%s\n" % item)
    # thefile.close()

    opFlowOfBlocks = np.zeros((xBlockSize, yBlockSize, 2))

    # thefile = open('spark mag averages.txt', 'w')
    # for item in mag_blocks.toLocalIterator():
    #     thefile.write("%s\n" % item)
    # thefile.close()


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
    # # print first 20*20 before dividing in blocks
    # thefile = open('mag without blocks.txt', 'w')
    # sum = 0
    # for i in range(20):
    #     for j in range(20):
    #         thefile.write("%s\n" % mag_numpy_array[i][j])
    #         sum += mag_numpy_array[i][j]
    #     thefile.write("\n")
    # thefile.close()
    # print ("avg of first 20*20 is " + str(sum/400.0))

    # # print first block after dividing in blocks 
    # thefile = open('mag with blocks.txt', 'w')
    # for i in array_of_blocks[0]:
    #     for j in i:
    #         thefile.write("%s\n" % j)
    #     thefile.write("\n")
    # thefile.close()

    # parallelize the frame blocks
    # rdd = sc.parallelize(array_of_blocks)
    # # print (rdd.collect())
    # rdd_of_avg = rdd.map(mapper)
    # print (rdd_of_avg.collect())

    # thefile = open('1.txt', 'w')
    # for item in blocks.collect():
    #     thefile.write("%s\n" % item)
    # thefile.close()
    # print ("avg of first block is " + str(blocks.collect()))
