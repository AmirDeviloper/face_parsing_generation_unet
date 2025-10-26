import cv2
from skimage import io

img_bg = cv2.imread('__new_ex1.png', 0)
img = cv2.imread('3898_nose.jpg', 0)

height_a, width_a = img_bg.shape
height_b, width_b = img.shape

diff_height = abs(height_a - height_b) // 2
diff_width = abs(width_a - width_b) // 2

img = cv2.copyMakeBorder(img, diff_height, diff_height, diff_width, diff_width, cv2.BORDER_CONSTANT, value=0)

backSub = cv2.createBackgroundSubtractorMOG2()
_ = backSub.apply(img_bg)
mask = backSub.apply(img)

mask_morph = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)))
mask_morph = cv2.morphologyEx(mask_morph, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)))

_, binary_image = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(25, 25))
closing = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(17, 17))
dilation = cv2.dilate(closing, kernel, iterations = 1)

and_img = cv2.bitwise_and(img, dilation)

cv2.imshow('mask', and_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
