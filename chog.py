import numpy as np

def calc_hog(im, numorient=9):
	 """
    calculate integral HOG (Histogram of Orientation Gradient) image (w,h,numorient)
     
    calc_hog(im, numorient=9)
     
    returns 
        Integral HOG image
 
    params 
        im : color image
        numorient : number of orientation bins, default is 9 (-4..4)
   
    """
    height, width, channels = im.shape

    # if color image convert to grayscale
    if channels > 1:
    	im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)

    # calc gradient using sobel
    gx = cv2.Sobel(im, cv2.CV_64F, 1, 0, ksize=3)
	gy = cv2.Sobel(im, cv2.CV_64F, 0, 1, ksize=3)

	# calc initial result
	hog = np.zeros((height, width, numorient))
	mid = numorient / 2
	for y in range(height - 1):
		for x in range(width - 1):
			angle = int(round(mid * np.arctan2 (gy[y, x], gx[y, x]) / np.pi)) + mid
			magnitude = np.sqrt(gx[y, x] ** 2 + gy[y, x] ** 2)
			hog[y, x, angle] += magnitude

def calc_chog():
	pass