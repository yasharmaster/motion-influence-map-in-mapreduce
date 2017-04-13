import numpy as np
import motionInfuenceGenerator as mig
import createMegaBlocks as cmb
from pyspark import SparkContext

def reject_outliers(data, m=2):
    return data[abs(data - np.mean(data)) < m * np.std(data)]

def train_from_video(vid, sc):
    '''
        calls all methods to train from the given video
        May return codewords or store them.
    '''
    print "Training From ", vid
    MotionInfOfFrames, rows, cols = mig.getMotionInfuenceMap(vid, sc)
    print "Motion Inf Map", len(MotionInfOfFrames)
    #numpy.save("MotionInfluenceMaps", np.array(MotionInfOfFrames), allow_pickle=True, fix_imports=True)
    megaBlockMotInfVal = cmb.createMegaBlocks(MotionInfOfFrames, rows, cols)
    np.save("/home/yash/work/project/unusual/code/videos/scene1/megaBlockMotInfVal_set1_p1_train_40-40_k5.npy",megaBlockMotInfVal)
    print(np.amax(megaBlockMotInfVal))
    print(np.amax(reject_outliers(megaBlockMotInfVal)))
    
    codewords = cmb.kmeans(megaBlockMotInfVal)
    np.save("/home/yash/work/project/unusual/code/videos/scene1/codewords_set1_p1_train_40-40_k5.npy",codewords)
    print codewords
    return
    
if __name__ == '__main__':
    '''
        defines training set and calls trainFromVideo for every vid
    '''

    trainingSet = [r"/home/yash/work/project/unusual/code/videos/scene1/train1.avi"]
    sc = SparkContext("local", "Simple App")
    for video in trainingSet:
        train_from_video(video, sc)
    sc.stop()
    print "Done"
