from pyspark import SparkContext
import cv2
import numpy as np
import math
import sys


sys.path.append('/usr/local/lib/python2.7/site-packages')

def blockshaped(arr, nrows, ncols):
    h, w = arr.shape
    return (arr.reshape(h//nrows, nrows, -1, ncols)
               .swapaxes(1,2)
               .reshape(-1, nrows, ncols))

def mapper(input):
    sum = 0
    for row in range(len(input)):
        for col in range(len(input[0])):
            sum += input[row][col]
    sum /= 400
    return sum

def mapreduce_to_file(mag):

    sc = SparkContext("local", "Simple App")
    # convert array to numpy array
    mag = np.asarray(mag)
    
    # divide array in blocks
    mag = blockshaped(mag, 20, 20)

    rdd = sc.parallelize(mag)

    blocks = rdd.map(mapper)

    thefile = open('mag averages.txt', 'w')

    for item in blocks.collect():
        thefile.write("%s\n" % item)
    
    thefile.close()
    sc.stop()

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
