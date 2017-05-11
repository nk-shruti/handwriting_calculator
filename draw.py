import cv2
import numpy as np

drawing = False 
ix,iy = -1,-1

def draw_circle(event,x,y,flags,param):
    global ix,iy,drawing,mode,px,py
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix,iy = x,y
        px,py=x,y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            cv2.circle(img,(x,y),5,(255,255,255),-1)
            cv2.line(img,(x,y),(px,py),(255,255,255),10)
            px,py=x,y


    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        px, py = x,y

def main():
    global img
    img = np.zeros((700,700,3), np.uint8)
    cv2.namedWindow('image')
    cv2.setMouseCallback('image',draw_circle)

    while(1):
        cv2.imshow('image',img)
        k = cv2.waitKey(1) & 0xFF
        if k==115:
            #save the image
            cv2.imwrite('pic.png',img)
        if k==32:
            break

    cv2.destroyAllWindows()


main()