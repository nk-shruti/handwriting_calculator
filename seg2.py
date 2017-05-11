import cv2
import numpy as np

def main():
    folder = './data/test/'

    im = cv2.imread("pic.png")

    im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    im_th = cv2.GaussianBlur(im_gray, (5, 5), 0)
    ctrs, hier = cv2.findContours(im_th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rects = [cv2.boundingRect(ctr) for ctr in ctrs]
    i = 0
    rects.sort(key=lambda x:x[0])
    # print rects
    imlist = []
    for r in rects:
        i = i+1
        # print r
        im1 = im[r[1]:r[1]+r[3],r[0]:r[0]+r[2]]
        imlist.append(im1)
        # cv2.imwrite(folder+str(i)+'.png',im1)
        # while(1):
        #     cv2.imshow('image',im1)
        #     k = cv2.waitKey(1) & 0xFF
        #     if k==32:
        #         #save the image
        #         cv2.imwrite(folder+str(i)+'.png',im1)
        #         break;
    # print imlist
    return imlist

    # cv2.waitKey()


main()