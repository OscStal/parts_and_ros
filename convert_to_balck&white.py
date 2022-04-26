import cv2
import numpy

frame = cv2.imread('/Users/ludvigtyden/Desktop/279073618_2732017530264808_3911679520841650677_n.png')
frame = frame[80:580, 1:330]    #cropped = img[start_row:end_row, start_col:end_col]
img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

cv2.imshow("show", img)

hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

hsv_channels = cv2.split(hsv)

rows = frame.shape[0]
cols = frame.shape[1]

for i in range(0, rows):
    for j in range(0, cols):
        h = hsv_channels[0][i][j]

        if h > 117 and h < 130:
            hsv_channels[2][i][j] = 255
        else:
            hsv_channels[2][i][j] = 0

#cv2.imshow("show", frame)
#cv2.imshow("show", hsv)
cv2.imshow("show2", hsv_channels[2])

cv2.waitKey(0)
