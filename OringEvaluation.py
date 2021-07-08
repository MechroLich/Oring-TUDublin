import cv2 as cv, numpy as np, time
from matplotlib import pyplot as plt
from functions import *

path = './orings/'
savePath = './verdict/'
i=1
while True:
    before = time.time()
    
    #read in an image into memory
    img_name = 'Oring' + str(i) + '.jpg'
    img = cv.imread(path +  img_name,0)

    hist = img_hist(img)
    #smooth_hist(histogram, smoothing_factor), smoothing factor must be odd
    hist = smooth_hist(hist,3)
    thresh = cluster_thresh(hist)    
    threshold(img,thresh)
    
    img = closing(img)
    #connected components
    connected_Array = connected_components(img)

    #defect detection function
    #get center
    center = getCentroid(connected_Array, 1)
    verdict = classify(connected_Array, center, i)

    after = time.time()
    timeTaken = round(after-before, 2)

    finalIng = displayLabelled(connected_Array, center, verdict, img, timeTaken)

    outputToFile(finalIng, i, savePath)
    
    i=(i+1)%16
    if i==0:
        i+=1

    ch = cv.waitKey(100)
    if ch & 0xFF == ord('q'):
        break
