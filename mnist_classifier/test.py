import cv2 as cv
import matplotlib.pyplot as plt

image = cv.imread("/home/cmodi/cmodi/htw/sem_1/v_fzg_entwicklung/groh/vibing_cat.jpg")
gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
image = image[:,:,2]

plt.imshow(gray)
plt.show()