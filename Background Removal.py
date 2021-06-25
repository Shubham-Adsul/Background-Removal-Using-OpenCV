import cv2
import cvzone
from cvzone.SelfiSegmentationModule import SelfiSegmentation

cap = cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)
cap.set(cv2.CAP_PROP_FPS, 68)
segmentor = SelfiSegmentation()
fpsreader = cvzone.FPS()

# background img path
# img should be in 640x480 size
imgbg = cv2.imread("BGimages1.jpg")


while True:
    success, img = cap.read()
    imgout = segmentor.removeBG(img, imgbg, threshold=0.95)


    imgstack = cvzone.stackImages([img,imgout],2,1)
    _, imgstack = fpsreader.update(imgstack,color=(0,0,255))

    cv2.imshow("Image out", imgstack)
    cv2.waitKey(1)
