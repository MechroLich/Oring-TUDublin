#author Timmy Dunne
import cv2 as cv, numpy as np, time, queue, math, time, sys
from matplotlib import pyplot as plt

def threshold(img,thresh):
    #original thresholding function, no longer used
    img[img > thresh] = 255
    img[img <= thresh] = 0
    return img

def img_hist(img):
    hist = np.zeros(256)
    for i in range(0,img.shape[0]):#loop through the rows
        for j in range(0,img.shape[1]):#loop through the columns
            hist[img[i,j]]+=1
    return hist

def smooth_hist(hist, smooth_factor):
    #histogram smoothing using an averaging(box) filter, input value allows you to change the smoothing factor(hard coded).
    i=0
    histogram_smooth = np.empty([])
    for x in range (0, int((smooth_factor-1)/2)):
        histogram_smooth = np.append(histogram_smooth, 0)
    while i <(len(hist)-smooth_factor):
        sum = 0
        for y in range(0, (smooth_factor-1)):
            sum = sum + hist[i+y]
        histogram_smooth = np.append(histogram_smooth, int(sum/smooth_factor))
        i+=1
    for x in range (0, int((smooth_factor-1)/2)):
        histogram_smooth = np.append(histogram_smooth, 0)
    return histogram_smooth        

def cluster_thresh(hist):
    #simple clustering algorithm
    #initial estimate for t is 128
    tOld = 128
    tNew = 0
    retry = True
    while(retry == True):
        c1 = 0
        c1tot = 0
        c2 = 0
        c2tot = 0
        #get values for C1
        for j in range (0, int(tOld)):
            c1 = c1 + (j*hist[j])
            c1tot = c1tot + hist[j]
        #get values for C2
        for p in range (int(tOld), len(hist)):
            c2 = c2 + (p*hist[p])
            c2tot = c2tot + hist[p]
        #Compute avg gray levels for u1 and u2
        tNew=((c1/c1tot)+(c2/c2tot))/2
        #if the tNew is within +/-1 of tOld, break the loop. Otherwise repeat until it is
        if(-1<(tOld-tNew)<1):
            retry = False
        else:
            tOld = tNew
    return tNew

def dialation(img):
    #if input is 0 and any its neighbors are 1(255), it becomes 1(255). otherwise it remains 0.
    #Dialation expands forground pixels
    temp = img.copy()   
    for i in range(0,img.shape[0]):#loop through the rows
        for y in range(0,img.shape[1]):#loop through the columns
            #if top row
            if img[i][y] == 0: 
                if i == 0:

                    #and if left column(top left corner)
                    if y == 0:
                        if img[1][0] == 255 or img[0][1] == 255 or img[1][1] == 255:
                            temp[i][y]=255

                    #or if right column(top right corner)
                    elif y == img.shape[1]:                   
                        if img[1][img.shape[1]] == 255 or img[0][img.shape[1]-1] == 255 or img[1][img.shape[1]-1] == 255:
                            temp[i][y]=255

                    #any other column in top row
                    else:                   
                        if img[i][y-1] == 255 or img[i+1][y-1] == 255 or img[i+1][y] == 255 or img[i+1][y+1] == 255 or img[i][y+1] == 255:
                            temp[i][y]=255

                #if bottom row
                elif i == img.shape[0]:

                    #and if left column(bottom left corner)
                    if y == 0:
                        if img[img.shape[0]-1][0] == 255 or img[img.shape[0]][1] == 255 or img[img.shape[0]-1][1] == 255:
                            temp[i][y]=255

                    #or if right column(bottom right corner)
                    elif y == img.shape[1]:
                        if img[img.shape[0]-1][img.shape[1]] == 255 or img[img.shape[0]][img.shape[1]-1] == 255 or img[img.shape[0]-1][img.shape[1]-1] == 255:
                            temp[i][y]=255
                            
                    #any other column in bottom row
                    else:                   
                        if img[image.shape[0]][y-1] == 255 or img[image.shape[0]-1][y-1] == 255 or img[image.shape[0]-1][y] == 255 or img[image.shape[0]-1][y+1] == 255 or img[image.shape[0]][y+1] == 255:
                            temp[i][y]=255

                else:

                    #left column not corners
                    if y == 0:                   
                        if img[i-1][y] == 255 or img[i-1][y+1] == 255 or img[i][y+1] == 255 or img[i+1][y+1] == 255 or img[i+1][y] == 255:
                            temp[i][y]=255
                            
                    #right column not corners
                    elif y == img.shape[1]:                   
                        if img[i-1][img.shape[1]] == 255 or img[i-1][img.shape[1]-1] == 255 or img[i][img.shape[1]-1] == 255 or img[i+1][img.shape[1]-1] == 255 or img[i+1][img.shape[1]] == 255:
                            temp[i][y]=255
                        
                    #middle
                    else:
                        if img[i-1][y-1] == 255 or img[i][y-1] == 255 or img[i+1][y-1] == 255 or img[i+1][y] == 255 or img[i+1][y+1] == 255 or img[i][y+1] == 255 or img[i-1][y+1] == 255 or img[i-1][y] == 255:
                            temp[i][y]=255
    return temp
  
def erosion(img):
    #if input is 1 (255) and any its neighbors are 0, it becomes 0. otherwise it remains 1.
    #erosion shrinks forground pixels in size and holes within the area become larger
    temp = img.copy() 
    for i in range(0,img.shape[0]-1):#loop through the rows
        for y in range(0,img.shape[1]-1):#loop through the columns
            #if top row
            if i == 0:
                #and if left column(top left corner)
                if y == 0:
                    if img[1][0] == 0 or img[0][1] == 0 or img[1][1] == 0:
                        temp[i][y]=0

                #or if right column(top right corner)
                elif y == img.shape[1]:                   
                    if img[1][img.shape[1]] == 0 or img[0][img.shape[1]-1] == 0 or img[1][img.shape[1]-1] == 0:
                        temp[i][y]=0

                #any other column in top row
                else:                   
                    if img[i][y-1] == 0 or img[i+1][y-1] == 0 or img[i+1][y] == 0 or img[i+1][y+1] == 0 or img[i][y+1] == 0:
                        temp[i][y]=0

            #if bottom row
            elif i == img.shape[0]:

                #and if left column(bottom left corner)
                if y == 0:
                    if img[img.shape[0]-1][0] == 0 or img[img.shape[0]][1] == 0 or img[img.shape[0]-1][1] == 0:
                        temp[i][y]=0

                #or if right column(bottom right corner)
                elif y == img.shape[1]:
                    if img[img.shape[0]-1][img.shape[1]] == 0 or img[img.shape[0]][img.shape[1]-1] == 0 or img[img.shape[0]-1][img.shape[1]-1] == 0:
                        temp[i][y]=0
                #any other column in bottom row
                else:                   
                    if img[image.shape[0]][y-1] == 0 or img[image.shape[0]-1][y-1] == 0 or img[image.shape[0]-1][y] == 0 or img[image.shape[0]-1][y+1] == 0 or img[image.shape[0]][y+1] == 0:
                        temp[i][y]=0

            else:

                #left column not corners
                if y == 0:                   
                    if img[i-1][y] == 0 or img[i-1][y+1] == 0 or img[i][y+1] == 0 or img[i+1][y+1] == 0 or img[i+1][y] == 0:
                        temp[i][y]=0
                #right column not corners
                elif y == img.shape[1]:                   
                    if img[i-1][img.shape[1]] == 0 or img[i-1][img.shape[1]-1] == 0 or img[i][img.shape[1]-1] == 0 or img[i+1][img.shape[1]-1] == 0 or img[i+1][img.shape[1]] == 0:
                        temp[i][y]=0
                    
                #middle
                else:
                    if img[i-1][y-1] == 0 or img[i][y-1] == 0 or img[i+1][y-1] == 0 or img[i+1][y] == 0 or img[i+1][y+1] == 0 or img[i][y+1] == 0 or img[i-1][y+1] == 0 or img[i-1][y] == 0:
                         temp[i][y]=0
    return temp

def closing(img):
    #reversed order of closing produced better results than dialation then erosion.
    img = erosion(img)
    img = dialation(img)
    return img

def connected_components(img):
    #returns image with each individual component given its own label (0,1,2,3,4,5...)
    labels = img.copy()
    curlab = 1

    q = queue.Queue()

    #Set all values in new array to 0
    for x in range(0, img.shape[0]):
        for y in range(0, img.shape[1]):
            labels[x,y] = 0

    for x in range(0, img.shape[0]):
        for y in range(0, img.shape[1]):
            # For each pixel
            if labels[x,y] == 0 and img[x,y] == 0:
                
                # If unvisited, and a foreground pixel
                labels[x,y] = curlab
                q.put([x,y])

                # Loop while queue isnt empty, when it is empty we increment curlab
                while q.qsize() != 0:
                    element = q.get()
                    neighbors = [
                        [element[0]+1, element[1]],
                        [element[0]-1, element[1]],
                        [element[0], element[1]+1],
                        [element[0], element[1]-1]
                    ]

                    #For each neighbor
                    for neighbor in neighbors:
                        if img[neighbor[0],neighbor[1]] == 0 and labels[neighbor[0], neighbor[1]] == 0:
                            labels[neighbor[0], [neighbor[1]]] = curlab
                            q.put(neighbor)  
                curlab += 1   
    return labels

def getCentroid(img, label):
    # returns center pixel of oring
    img = img.copy()

    centroid = (0,0)
    x_sum = 0
    y_sum = 0
    labelCount = 0
    x = 0
    y = 0
    #count all cells with given label
    for x in range(0, img.shape[0]):
        for y in range(0, img.shape[1]):
            if(img[x,y] == label):
                x_sum += x
                y_sum += y
                labelCount += 1
            y += 1
        x += 1
    
    #get average x and y values and round to get center
    x_avg = round(x_sum / labelCount)
    y_avg = round(y_sum / labelCount)


    centroid = [x_avg, y_avg]
    return centroid

def getDistance(x1, y1, x2, y2):
    # returns distance between two points
    xDifference = x1-x2
    yDifference = y1-y2
    #pythagoras
    return math.sqrt((xDifference*xDifference) + (yDifference * yDifference))

def getCoV(img, label, centroid, imageNumber):
    # Returns: The overall circularity/roundness value for an O-ring using the
    # coeffecient of variance(CoV). a Perfect circle would have a CoV of 0 therefore any decision
    # system must account for the inherent variance in radius due to the thickness of the oring.
    # used as a secondary metric for flaw detection.
    img = img.copy()
    distances = []

    for x in range(0, img.shape[0]):
        for y in range(0, img.shape[1]):
            if img[x,y] == label:
                distance = getDistance(x,y,centroid[0],centroid[1])
                distances.append(distance)
    
    meanDistances = np.mean(distances)
    standardDeviation = np.std(distances)
    coV = (standardDeviation/meanDistances)

    return coV

def getRadius(img, centroid):
    # returns inner and outer radius as well as thickness
    img = img.copy()
    innerRadiusX = 0
    outerRadiusX = 0
    innerRadiusY = 0
    outerRadiusY = 0
    innerRadiusXY = 0
    outerRadiusXY = 0
    innerRadius = 0
    outerRadius = 0
    thickness = 0

    x = centroid[0]
    y = centroid[1]

    # location:    0:Within O-ring   1: On O-ring   2: Outside Oring
    location = 0

    # examine connected component labels to the right until reach outside the ring
    while location != 2:
        if location == 0 and img[x,y] == 1:
            location = 1
            innerRadiusX = getDistance(x, y, centroid[0], centroid[1]) - 1
        elif location == 1 and img[x,y] == 0:
            location = 2
            outerRadiusX = getDistance(x, y, centroid[0], centroid[1]) - 1
        x += 1

    x = centroid[0]
    y = centroid[1]
    location = 0

    # examine connected component labels to upward until reach outside the ring
    while location != 2:
        if location == 0 and img[x,y] == 1:
            location = 1
            innerRadiusY = getDistance(x, y, centroid[0], centroid[1]) - 1
        elif location == 1 and img[x,y] == 0:
            location = 2
            outerRadiusY = getDistance(x, y, centroid[0], centroid[1]) - 1
        y += 1
    
    outerRadius = (outerRadiusX + outerRadiusY)/2
    innerRadius = (innerRadiusX + innerRadiusY)/2
    
    thickness = outerRadius - innerRadius
    return (outerRadius, innerRadius, thickness)

def isInsideCircle(img, centroid, x, y, radius):
    # Returns: Whether or not a point is within a certain radius of the centroid  (1/0)   
    if (x - centroid[0])**2 + (y - centroid[1])**2 < radius**2:
        return 1
    elif (x - centroid[0])**2 + (y - centroid[1])**2 > radius**2:
        return 0

def getBoundsRatio(img, centroid):
    # returns ratio of pixels that are out of bounds (Black inside inner radius, white in the actual ring) to the total amount
    # used for detecting flaws/chips in the oring as well as breaks
    img = img.copy()

    r = getRadius(img, centroid)
    outerRadius = r[0]
    innerRadius = r[1]

    whiteCounter = 0
    totalOutOfBounds = 0
    total=0
    #totalInBounds = 0


    for x in range(0, img.shape[0]):
        for y in range(0, img.shape[1]):
            insideOuter = isInsideCircle(img, centroid, x, y, outerRadius)
            insideInner = isInsideCircle(img, centroid, x, y, innerRadius)
            if(img[x,y] == 1):
                whiteCounter += 1
            if insideOuter and not isInsideCircle(img, centroid, x, y, innerRadius):
                # Inside outer radius, not inside inner radius
                if img[x,y] == 1:
                    #totalInBounds += 1
                    total += 1
                elif img[x,y] == 0:
                    totalOutOfBounds += 1
                    total += 1
            elif insideInner:
                #Inside inner radius
                if img[x,y] == 1:
                    totalOutOfBounds += 1
                    total += 1
            elif not insideOuter and not insideInner:
                # Outside both radius
                if img[x,y] == 1:
                    totalOutOfBounds +=1
                    total += 1

    ratio = totalOutOfBounds / total
    return (ratio)

def classify(img, centroid, imageNumber):
    # Returns: A verdict of whether the image should pass or fail
    # One false negative in the test set due to its shape being out of the 
    # norm for orings given the image data provided(image 14)

    boundsRatio = round(getBoundsRatio(img, centroid), 3)
    coV = round(getCoV(img, 1, centroid, imageNumber), 3)
    verdict = "UNSURE"
    verdictColor = "\033[42m"

    if(boundsRatio >= 0.17):
        verdict = "FAIL (SNAPPED)"
        verdictColor = "\033[41m"
    elif(coV< 0.09):
        verdict = "PASSED"

    if(verdict == "UNSURE"):
        if boundsRatio < 0.09:
            verdict = "PASSED"            
        else:
            verdict = "FAIL (FAULTY) "
            verdictColor = "\033[41m"
    
    print(verdictColor + "\033[30m" + str(imageNumber)+" - "+str(verdict)+" | "+str(boundsRatio)+" | "+ str(coV)+'\033[0m')
    return verdict

def displayLabelled(img, centroid, verdict, originalImg, timeTaken):
    # return image with labeled text
    img = img.copy()
    r = getRadius(img, centroid)

    mostFrequent = 0
    freq = [0,0,0,0]

    # Get the amount of different labels
    for x in range(0, img.shape[0]):
        for y in range(0, img.shape[1]):
            if img[x,y] > 0:
                freq[img[x,y]] += 1
    # Set the most frequent label so we can identify the O-ring over broken pieces
    mostFrequent = np.argmax(freq)
    
    for x in range(0, img.shape[0]):
        for y in range(0, img.shape[1]):
            if img[x,y] > 0 and img[x,y] != mostFrequent:
                img[x,y] = 100
            elif img[x,y] == mostFrequent:
                img[x,y] = 255

    font = cv.FONT_HERSHEY_SIMPLEX

    img = cv.cvtColor(img,cv.COLOR_GRAY2RGB)
    if verdict == "PASSED":
        cv.putText(img, str(verdict), (5,20), font, 0.4, (0,255,0), 1, cv.LINE_AA)
    else:
        cv.putText(img, str(verdict), (5,20), font, 0.4, (0,0,255), 1, cv.LINE_AA)

    cv.putText(img, 'Time: ' +str(timeTaken)+" sec", (5,210), font, 0.4, (255,255,255), 1, cv.LINE_AA)
    #draw outer circle
    cv.circle(img, (centroid[1], centroid[0]), round(r[0]), (50,50,255), 1)
    #draw inner circle
    cv.circle(img, (centroid[1], centroid[0]), round(r[1]), (20,20,255), 1)
    #draw center point
    cv.circle(img, (centroid[1], centroid[0]), 1, (0,0,255), 2)

    cv.imshow('Final', img)
    cv.waitKey(0)
    cv.destroyAllWindows()

    return img

def outputToFile(img, imageNumber, path):
    #saves image
    cv.imwrite( path+str(imageNumber)+".jpg", img )
