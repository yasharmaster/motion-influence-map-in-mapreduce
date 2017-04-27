import numpy as np
import motionInfuenceGenerator as mig
import createMegaBlocks as cmb
from pyspark import SparkContext
import sys
import time

def reject_outliers(data, m=2):
    return data[abs(data - np.mean(data)) < m * np.std(data)]

def write_mega_block_to_file(megaBlockMotInfVal):
    print ("megaBlockMotInfVal file write starts")
    thefile = open('megaBlockMotInfVal.txt', 'w')
    for x in megaBlockMotInfVal:
        thefile.write("%s\n" % x)
    thefile.close() 
    print ("megaBlockMotInfVal file write ends")

def train_from_video(vid, sc):
    '''
        calls all methods to train from the given video
        May return codewords or store them.
    '''
    print "Training From ", vid
    MotionInfOfFrames, rows, cols = mig.getMotionInfuenceMap(vid, sc)
    #print "Motion Inf Map", len(MotionInfOfFrames)
    #numpy.save("MotionInfluenceMaps", np.array(MotionInfOfFrames), allow_pickle=True, fix_imports=True)
    megaBlockMotInfVal = cmb.createMegaBlocks(MotionInfOfFrames, rows, cols)
    #np.save("/home/yash/work/project/unusual/code/videos/scene1/megaBlockMotInfVal_set1_p1_train_40-40_k5.npy",megaBlockMotInfVal)  
    
    # write_mega_block_to_file(megaBlockMotInfVal)
    codewords = cmb.kmeans_map_reduce(megaBlockMotInfVal, sc)
    #cmb.kmeans(megaBlockMotInfVal)

    #sc.stop()
    #sys.exit()

    np.save("/codewords.npy",codewords)
    return
    
if __name__ == '__main__':
    '''
        defines training set and calls trainFromVideo for every vid
    '''

    trainingSet = [r"/2_train1.avi"]
    sc = SparkContext("local", "Simple App")
    start_time = time.time()
    for video in trainingSet:
        train_from_video(video, sc)
    print("--- %s seconds ---" % (time.time() - start_time))
    sc.stop()
    print "Done"
