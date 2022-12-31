import cv2
import numpy as np
from sklearn import cluster
import sys

class DiceDetector:

    def __init__(self):
        self.blobparams = cv2.SimpleBlobDetector_Params()

        self.blobparams.filterByInertia
        self.blobparams.minInertiaRatio = 0.7
        self.blobparams.filterByArea = True
        self.blobparams.minArea = 50
        self.blobparams.maxArea = 5000

        self.blobparams.filterByCircularity = True
        self.blobparams.filterByInertia = True
        self.blobparams.minCircularity = 0.3

        self.detector = cv2.SimpleBlobDetector_create(self.blobparams)

        self.thresholdMin = 52
        self.kernel_erode_size = 3
        self.clustering_size = 10
        self.blobMin_size = 10
        self.distortion_amount = 8

        self.kernel_erode = np.ones((self.kernel_erode_size, self.kernel_erode_size),np.uint8) 
    
    def get_blobs(self,frame):

        blobs = self.detector.detect(frame)

        return blobs
    
    def intersects(self, seg1, seg2):
        #check if midpoints within certain distance

        thresh = 10 #threshold distance 2 midpoints are close enough
        p1, q1 = seg1
        p2, q2 = seg2

        m1 = (p1+q1)/2
        m2 = (p2+q2)/2

        dist = np.linalg.norm(m1-m2)
        if dist <= thresh:
            return True
        else:
            return False
    
    def get_dice_from_blobs(self,blobs):
        # Num of each dice:
        numDieWithValue = np.zeros(6) #each indeces is die value + 1

        sweepClusters = [18, 25, 25, 32, 44]
        sweepDie      = [[3],[3],[5],[4],[1,2]] #note the first die is 3 because 6s have the closest spacing of two 3-pip sets; we merge them in post. Then we run same clustering and detect and 6s

        # Get centroids of all blobs
        # remove the blobs from global list that match the indices found in "remove"
        X = []
        for b in blobs:
            pos = b.pt
            if pos != None:
                X.append(pos)

        X = np.asarray(X)
        labeledPips = np.zeros(len(X), dtype=bool)
        diceNumber = np.zeros(6)
        if len(X) > 0:

            #run detector 3 times to detect all number combinations
            # first: die nums 6,5,3
            # 4
            # last: die nums: 2, 1
            #clusters = [25,32,44]
            dice = []
            idx = 0

            for cs in sweepClusters:

                #print(labeledPips)
                if np.all(labeledPips):
                    #print("Labeled all pips!")
                    break

                #dice = get_dice_from_blobs(blobs)
                #diceNumber += countDice(dice, sweepDie[idx])
                
                # Important to set min_sample to 0, as a dice may only have one dot
                #print(X)
                #Create a new X, but with change the value to (-999,-999) so we can easily filter those points
                X_unlabeled = np.copy(X)
                X_unlabeled[labeledPips] = np.ones(2) * -999 #apply mask
                clustering = cluster.DBSCAN(eps=cs, min_samples=0).fit(X_unlabeled)

                # Find the largest label assigned + 1, that's the number of dice found
                num_dice = max(clustering.labels_) + 1
                
                for i in range(num_dice):
                    X_dice = X[clustering.labels_ == i]

                    centroid_dice = np.mean(X_dice, axis=0) # Calculate centroid of each dice, the average between all a dice's dots
                    diceCheckImportantNumbers = sweepDie[idx]
                    # only add dice values from the corresponding clustering threshold, then mark them as found
                    isNumFromCluster = len(X_dice) in diceCheckImportantNumbers
                    hasUnlabeledPips = np.all(~labeledPips[clustering.labels_ == i])
                    isNotOffscreen = X_dice[0][0] >= 0

                    if idx == 3 and len(X_dice) > 4 and hasUnlabeledPips: #at this point in chain of dice checking, 4s can only be miscategorized with 2s or other 4s, therefore we check num pips is even number
                        #TODO: break down clusters of 4 into multiples:
                        #create list of all the line segments within thresholds
                        dice, labeledPips = self.count_4_dice(dice, X_dice, X, labeledPips)
                        #find all the intersections of line segments

                        #use intersections
                        #print("4s are merging")
                    elif idx == 4 and len(X_dice) > 2 and hasUnlabeledPips:
                        dice, labeledPips = self.count_2and1_dice(dice, X_dice, X, labeledPips)
                    elif isNumFromCluster and hasUnlabeledPips and isNotOffscreen:
                        #print(str(cs) + ":" + str(num_dice) +" value: " + str(len(X_dice)))
                        
                        #if sweepDie[idx] == 4 and len(X_dice) > 4: #check if the clustering is picking up more than just 4s (e.g. two 4s flush against each other)
                        #    dice, labeledPips = count_4_dice(dice, X_dice, labeledPips)
                        #elif sweepDie[idx] == 2 and len(X_dice) > 2: #check if the clustering is picking up more than just 2s (e.g. two 2s flush against each other)
                        #    dice, labeledPips = count_2_dice(dice, X_dice, labeledPips)
                        #else:
                        dice.append([len(X_dice), *centroid_dice])
                        labeledPips[clustering.labels_ == i] = True
                #print(numDieWithValue)
                if idx == 0: #combine 3s into 6 dice:
                    #print(dice)
                    dice = self.count_6_dice_from3s(dice)
                idx += 1
                #print(dice)

            diceNumber += self.countDice2(dice)

            
            return diceNumber, dice

        else:
            return diceNumber,[]
        
    def find_segments(self, pips, min, max):
        #find all pips of correct range (i.e. diagonal pips of the 4)
        segments = []
        centroid = np.zeros(2)
        i = 0
        c = 0
        for p in pips: 
            for ji in range(1,len(pips)):
                j = pips[ji]
                dist = np.linalg.norm(p-j)
                if dist <= max and dist >= min and not [i,ji] in segments and not [ji,i] in segments:
                    segments.append([i,ji]) #store the indices of pips
            i += 1
            centroid += np.array([p[0],p[1]])
            c+=1
        
        centroid = centroid / c #center from average of all pips

        return segments, centroid
    
    def count_2and1_dice(self, dice, pips, allpips, pipMask):
        global frame
        #find all diagonal pips, group as 2s, left overs are 1s
        min = 40
        max = 44
        # Green color in BGR
        color = (0, 0, 255)
        if len(pips) > 2:
            segments, centroid = self.find_segments(pips, min, max)

        #label each segment as a 2:
        for s in segments:
            pip1  = pips[s[0]]
            pip1i = self.indexWherePosition(allpips, pip1[0], pip1[1])
            pip2 = pips[s[1]]
            pip2i = self.indexWherePosition(allpips, pip2[0], pip2[1])
            center = (pip1 + pip2)/2
            if(False):#debug
                frame = cv2.line(frame, (int(pip1[0]),int(pip1[1])), (int(pip2[0]),int(pip2[1])) , color, 1)
            if np.all(~pipMask[[pip1i,pip2i]]):
                dice.append( [ 2, center[0], center[1] ])
                pipMask[pip1i] = True
                pipMask[pip2i] = True

        #check there are any floating single pips, those are 1s:
        for p in pips:
            pi = self.indexWherePosition(allpips, p[0], p[1])
            if(~pipMask[pi]):
                dice.append( [ 1, p[0], p[1] ])
                pipMask[pi] = True

            

        return dice, pipMask
    
    def count_4_dice(self, dice, pips, allpips, pipMask):
        global frame
        #find all diagonal pips, then intersecting lines, which are the 4s
        min = 32
        max = 44

        segments, centroid = self.find_segments(pips, min, max)

        #cv2.circle(frame, (int(centroid[0]), int(centroid[1])),
        #               int(9), (0, 0, 255), 1)

        # Green color in BGR
        color = (0, 0, 255)
        intersections = []    
        #find segments that intersect:
        
        counted = np.zeros(len(segments), dtype=bool)
        ki = 0
        d = 0
        start1 = np.array([0, 0])
        end1   = np.array([0, 0])
        start2 = np.array([0, 0])
        end2   = np.array([0, 0])

        for k in segments: #iterate over all segments (nested for loop) and find intersections using intersect function above
            for li in range(1,len(segments)):
                l = segments[li]
                
                start1 = np.array([int(pips[k[0]][0]), int(pips[k[0]][1])])
                end1   = np.array([int(pips[k[1]][0]), int(pips[k[1]][1])])
                start2 = np.array([int(pips[l[0]][0]), int(pips[l[0]][1])])
                end2   = np.array([int(pips[l[1]][0]), int(pips[l[1]][1])])
                if self.intersects((start1, end1), (start2, end2)) and not l == k and not counted[ki]:
                    intersections.append([start1,end1,start2,end2])
                    counted[li] = True
                    break
            if(False):#debug
                frame = cv2.line(frame, (start1[0],start1[1]), (end1[0],end1[1]) , color, 1)
                d += 1

            ki += 1
        
        #print("intrsectns: " +str(len(intersections)))
        #print("  segments: " +str(len(segments)))
        #print("line draws: " +str(d))

        if len(intersections) == 0:
            return dice, pipMask

        num4s = int(len(pips) / 4)

        sorted_intersections = []
        #if len(pips) % 4 == 0: #and (len(pips) / 4) - 1 == len(intersections): #num pips and intersections suggest a multiple of 4
        #sort the intersections by distance to center and use the top num4s as the centers

        maxDist    = 99999999
        foundValid = 0
        while( foundValid < num4s ):
            dist   = 0
            maxIdx = -1
            furthestPoint = np.zeros(2)
            for i in range(len(intersections)): #Find the furthest intersection from centroid:
                itr     = intersections[i]
                xSum    = itr[0][0] + itr[1][0]+ itr[2][0] + itr[3][0]
                ySum    = itr[0][1] + itr[1][1]+ itr[2][1] + itr[3][1]
                intrsctnPoint = np.array([xSum,ySum]) / 4
                i_dist = np.linalg.norm(intrsctnPoint - centroid)
                pip1 = self.indexWherePosition(allpips, itr[0][0], itr[0][1])
                pip2 = self.indexWherePosition(allpips, itr[1][0], itr[1][1])
                pip3 = self.indexWherePosition(allpips, itr[2][0], itr[2][1])
                pip4 = self.indexWherePosition(allpips, itr[3][0], itr[3][1])

                if i_dist > dist and i_dist < maxDist and np.all(~pipMask[[pip1,pip2,pip3,pip4]]): #and check pips haven't been counted
                    maxIdx = i
                    dist = i_dist
                    furthestPoint = intrsctnPoint
            if maxIdx >= 0:
                sorted_intersections.append([furthestPoint, intersections[maxIdx], dist])
                maxDist = dist
                foundValid += 1
            else:
                return dice, pipMask

            #add dice from furthest intersection
            #for i in range(num4s):
            pos = [int(sorted_intersections[0][0][0]), int(sorted_intersections[0][0][1])]
            dice.append( [ 4, pos[0], pos[1] ])
            #print([centroid,pos])
            #cv2.circle(frame, (int(pos[0]), int(pos[1])),
            #        int(25), (0, 0, 255), 1)
            #TODO: update pipMask (labeledPips array) from the pips that touch intersections
            for k in range(4):
                pipIndex = self.indexWherePosition(allpips, sorted_intersections[0][1][k][0], sorted_intersections[0][1][k][1])
                if pipIndex >= 0:
                    pipMask[pipIndex] = True
            sorted_intersections = []


        #TODO: return the new list with found dice, and return the updated pipMask with the pips used 
        return dice, pipMask

    def indexWherePosition(self, pips, posX, posY):
        i = 0
        for p in pips:
            if int(p[0]) == int(posX) and int(p[1]) == int(posY):
                return i
            i += 1
        
        return -1
    
    def count_6_dice_from3s(self, dice):
        #The 6 dice has the closest pip spacing of 3 in a row with matching set for a total of 6
        #      So the first clustering locates any pairs of 3 and merges them into 6s

        maxDistance6Pips = 32

        #check there's even number of 3s
        #if len(dice) % 2 > 0:
        #    return dice

        mergedDice = []
        matchingPairsMask = np.zeros(len(dice))
        #TODO: foreach set of 3, find the shortest distance to matching 3 set and update dice array
        #while(~np.all(matchingPairsMask)):
        minDistance = 99999
        matchingPairIndex = -1
        newCentroid = np.zeros(2)
        i = 0
        for d in dice:
            for k in range(0,len(dice)): #iterate over every remaining dice
                a = np.array([d[1],d[2]])
                b = np.array([dice[k][1],dice[k][2]])
                dist = np.linalg.norm(a-b)
                if dist < minDistance and i != k:
                    minDistance = dist
                    matchingPairIndex = k
                    newCentroid = (a+b)/2
            if not matchingPairsMask[matchingPairIndex] and minDistance < maxDistance6Pips:
                #merge smallest distance:
                mergedDice.append([6,newCentroid[0],newCentroid[1]])
                matchingPairsMask[[i,k]] = True
                matchingPairsMask[matchingPairIndex] = True
            i += 1
            minDistance = 99999

        if len(mergedDice) == 0:
            return dice

        #check if any remaining:
        if(~np.all(matchingPairsMask)):
            unmasked = np.where(matchingPairsMask < 1)[0][0]
            unpaired3 = dice[unmasked]
            mergedDice.append(unpaired3)

        #merged centroids and updated dice list:
        return mergedDice
    
    def undistort(self, width, height, amount, src):
        distCoeff = np.zeros((4,1),np.float64)

        # TODO: add your coefficients here!
        k1 = -1.0e-5 * amount # negative to remove barrel distortion
        k2 = 0.0
        p1 = 0.0
        p2 = 0.0

        distCoeff[0,0] = k1
        distCoeff[1,0] = k2
        distCoeff[2,0] = p1
        distCoeff[3,0] = p2

        # assume unit matrix for camera
        cam = np.eye(3,dtype=np.float32)

        cam[0,2] = width/2.0  # define center x
        cam[1,2] = height/2.0 # define center y
        cam[0,0] = 10.        # define focal length x
        cam[1,1] = 10.        # define focal length y

        # here the undistortion will be computed
        dst = cv2.undistort(src,cam,distCoeff)
        return dst
        #cv2.imshow('dst',dst)

    def overlay_info(self, frame, dice, blobs):
        # Overlay blobs
        for b in blobs:
            pos = b.pt
            r = b.size / 2

            #cv2.circle(frame, (int(pos[0]), int(pos[1])),
            #           int(r), (255, 100, 0), 2)

        # Overlay dice number
        for d in dice:
            # Get textsize for text centering
            textsize = cv2.getTextSize(
                str(d[0]), cv2.FONT_HERSHEY_PLAIN, 3, 2)[0]

            #Render text twice for black border:
            cv2.putText(frame, str(d[0]),
                        (int(d[1] - textsize[0] / 2),
                        int(d[2] + textsize[1] / 2)),
                        cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 0), 4)
            cv2.putText(frame, str(d[0]),
                        (int(d[1] - textsize[0] / 2),
                        int(d[2] + textsize[1] / 2)),
                        cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)
    
    def countDice(self, dice, numbers):
        counted = np.zeros(6)

        for d in dice:
            for n in numbers:
                if d[0] == n:
                    counted[n-1] += 1

        return counted
    
    def countDice2(self, dice):
        counted = np.zeros(6)

        for d in dice:
            n = d[0]-1 #first element of dice array is total pips, convert it to index of array:
            if n < 6:
                counted[n] += 1

        return counted
