import cnn as c
import cv2
import numpy as np
import seg2 as seg
import draw as d

d.main()
# seg.main()
# return
imlist = seg.main()
# print imlist
model = c.init_model('bestval.h5')
l = ''
# i=0
pred2digit = dict({
		11 : '0',
		5 : '1',
		10 : '2',
		9 : '3',
		2 : '4',
		1 : '5',
		8 : '6',
		7 : '7',
		0 : '8',
		4 : '9',
		6 : '+',
		3 : '-'
	})
for im1 in imlist:
	# i=i+1
	# im2 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
	im1 = cv2.resize(im1, (96, 96), interpolation=cv2.INTER_AREA)
	# while(1):
	# 	cv2.imshow('image',im1)
	# 	k = cv2.waitKey(1) & 0xFF
	# 	if k==32:
	# 			#save the image
	# 			cv2.imwrite('./'+str(i)+'.png',im1)
	# 			break	

	im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2RGB)

	pclass =  np.argmax(model.predict(im1.reshape(1, 96, 96, 3)))
	l += pred2digit[pclass]
print eval(l)
#     while(1):
#             cv2.imshow('image',im1)
#             k = cv2.waitKey(1) & 0xFF
#             if k==32:
#                 #save the image
#                 # cv2.imwrite(folder+str(i)+'.png',im1)
#                 break;

#     print model.predict(image)	
