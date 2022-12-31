import cv2
import numpy as np
from sklearn import cluster
import sys
import dicedetector as dd

# ================Initialize a video feed -or- input an image: ===========

frame = cv2.imread("exampledice.png")

imgWidth  = frame.shape[1] #600
imgHeight = frame.shape[0] #600

#scale the image:
#NOTE: there are hard coded clustering values in get_dice_from_blobs(), it's just easier to scale the image but those values can be tweaked in variable "sweepClusters"
scale_percent = 60 # percent of original size
width = int(frame.shape[1] * scale_percent / 100)
height = int(frame.shape[0] * scale_percent / 100)
dim = (width, height)
print(dim)
frame = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
imgHeight = height
imgWidth  = width

# cap = cv2.VideoCapture(0)
# cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)
# cap.set(cv2.CAP_PROP_EXPOSURE, -7)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, imgWidth)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, imgHeight)

# fps = cap.get(cv2.CAP_PROP_FPS) # Gets the frames per second
# print("camera fps: " + str(fps))

#================ Initialize Dice Detector:

dicedetector = dd.DiceDetector() #bunch of hard coded values in init to tune for your application
diceDebounced = np.zeros(6)
#frameCount = 0

#================ Initialize UI:

haveCreatedUI = False

def on_change(value):
    #cap.set(cv2.CAP_PROP_EXPOSURE, -value)
    print("change cam exposure to: ", str(value))

def on_change_threshold(value):
    #global thresholdMin
    dicedetector.thresholdMin = value

def on_change_erode(value):
    #global kernel_erode_size
    dicedetector.kernel_erode_size = value
    dicedetector.kernel_erode = np.ones((dicedetector.kernel_erode_size, dicedetector.kernel_erode_size),np.uint8)

def on_change_distortion(value):
    #global distortion_amount
    dicedetector.distortion_amount = value

#def on_change_clustering(value):
#    global clustering_size
#    clustering_size = value

def on_change_blobsize(value):
    # global blobMin_sizeq
    # global params
    # global detector
    dicedetector.blobparams.minArea = value
    dicedetector.detector = cv2.SimpleBlobDetector_create(dicedetector.blobparams)

def on_change_blobshape(value):
    # global blobMin_size
    # global params
    # global detector
    dicedetector.blobparams.minInertiaRatio = value/10.0
    dicedetector.detector = cv2.SimpleBlobDetector_create(dicedetector.blobparams)


while(True):

    dice = [] #each frame we want to fill this with detect dice locations

    #Alternative idea that could be useful:
    #https://docs.opencv.org/4.x/d3/db4/tutorial_py_watershed.html

    # Grab the latest image from the video feed, or comment out and use image
    #ret, frame = cap.read()

    #frameCount += 1

    #adjust for barrel distortion from camera lens:
    #frame = dicedetector.undistort(imgWidth,imgHeight, dicedetector.distortion_amount, frame)

    # #crop:
    # h=480
    # w=480
    # y=130 #round((imgHeight-h)/2)
    # x=round((imgWidth-w)/2)

    # frame = frame[y:y+h, x:x+w]

    g = frame.copy()
    # set blue and red channels to 0 because we want to look very brightly at pips
    g[:, :, 0] = 0
    g[:, :, 2] = 0

    gray = cv2.cvtColor(g,cv2.COLOR_BGR2GRAY)
    #frame_gray = cv2.bitwise_not(gray)
    
    frame_gray = cv2.erode(gray, dicedetector.kernel_erode)
    ret2, frame2 = cv2.threshold(frame_gray,dicedetector.thresholdMin,255,cv2.THRESH_BINARY)
    #frame2 = cv2.bitwise_not(frame2)

    blobs = dicedetector.get_blobs(frame2)

    #TODO: Update the blobs array after detecting numbers
    #From the blobs, cluster them at varying thresholds to detect the die sides. In a successive fashion, remove the detected clusters from blob list.
    # eg 6,5,3 have the closest pip spacings, so remove corresponding blobs, then repeat for die side 4 and remove blobs, etc
    
    #TODO: fit all the following functionality into get_dice_from_blobs(?)
    
    summary, dice = dicedetector.get_dice_from_blobs(blobs)

    #handles detection being affected by camera image noise:
    diceDebounced = (diceDebounced*0.9) + (summary*0.1)
    print(np.around(diceDebounced)) #Evenly round to the given number of decimals

    #Render detected die values onto frame:
    detectedFrame = frame.copy()
    dicedetector.overlay_info(detectedFrame, dice, blobs)
    #cv2.imshow("frame_blurred", frame_blurred)
    cv2.imshow("frame2", frame2)
    cv2.imshow("frame", detectedFrame)

    if not haveCreatedUI:
        haveCreatedUI = True
        cv2.createTrackbar('exposure', 'frame', 0, 8, on_change)
        cv2.createTrackbar('threshold', 'frame2', dicedetector.thresholdMin, 250, on_change_threshold)
        cv2.createTrackbar('erode', 'frame2', dicedetector.kernel_erode_size, 48, on_change_erode)
        #cv2.createTrackbar('clustering', 'frame', clustering_size, 128, on_change_clustering)
        cv2.createTrackbar('distortion', 'frame', 1, 128, on_change_distortion)
        cv2.createTrackbar('blob size', 'frame', 10, 128, on_change_blobsize)
        cv2.createTrackbar('circularity', 'frame', 0, 10, on_change_blobshape)
        

    res = cv2.waitKey(1)

    # Stop if the user presses "q"
    if res & 0xFF == ord('q'):
        break

# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()
