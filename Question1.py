import cv2 as cv2
import numpy as np

image = cv2.imread(r'C:\Users\sahru\OneDrive\Desktop\ENPM673 - code\Midterm\Q1image.png')
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(47,47))
erosion = cv2.erode(image,kernel,iterations = 1)
dilation = cv2.dilate(erosion,kernel,iterations = 1)
image_edge = cv2.Canny(dilation,100,200)

contours, hierarchy = cv2.findContours(image_edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) #boundaries
print("Number of Contours found = " + str(len(contours)))

cv2.drawContours(dilation, contours, -1, (0, 255, 0), 7)

cv2.imshow('Contour Image',dilation)
cv2.waitKey(0)

# =============================================================================
#  counting number of circles with cv2.HoughCircles()
# =============================================================================

# finding number of circles in opened_image
circles = cv2.HoughCircles(dilation,cv2.HOUGH_GRADIENT,1.2,25,param1=30,param2=25,minRadius=0,maxRadius=35)

#gray image to BGR image
displayImage=cv2.cvtColor(dilation, cv2.COLOR_GRAY2BGR)

disIm=dilation.copy()

# drawing circles on to  the image
for circle in circles[0]:
    
    centerX=int(circle[0])
    centerY=int(circle[1])
    r=int(circle[2])
    cv2.circle(disIm,(centerX,centerY),r,(0,255,0),2)
    cv2.circle(disIm,(centerX,centerY),2,(0,0,255),5)

cv2.imshow('circles detected by cv2.HoughCircles',disIm)
# cv2.imwrite('../Results/CirclesDetected_with_HoughCircle.png',disIm)
cv2.waitKey(0)
cv2.destroyAllWindows()

print('Number of circles detected  by Hough Circles:',len(circles[0]))

# =============================================================================
# option2-Count of circles through CCA(Connected Component Analysis)
# =============================================================================

ret,im_labels=cv2.connectedComponents(dilation)

print(f"Number of circles detected by Connected Compont Analysis: {im_labels.max()}")

#visulazing CCA image

minVal=im_labels.min()
maxVal=im_labels.max()

# normalizing pixel values in between 0 and 255
normalized_image=255*(im_labels-minVal)/(maxVal-minVal)

normalized_image=np.uint8(normalized_image)
# gray_normalized_image=
colormapped_image=cv2.applyColorMap(normalized_image,cv2.COLORMAP_JET)

cv2.imshow('CCA image after colormapping', colormapped_image)
# cv2.imwrite('../Results/CirclesDetected_with_CCA.png',colormapped_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.destroyAllWindows()
