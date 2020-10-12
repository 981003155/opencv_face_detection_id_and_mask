import cv2
capture = cv2.VideoCapture(0)
imgcount = 1
while True:
    ret, frame = capture.read()
    frame = cv2.flip(frame,1)   #镜像操作
    cv2.imshow("video", frame)
    cv2.imwrite("/home/cheng/my_code/face/Face-Detection-and-Identification-master/images/shiren/"+str(imgcount)+".jpg",frame)
    print('image coutn:',imgcount)
    imgcount = imgcount+1
    key = cv2.waitKey(50)
    #print(key)
    if key  == ord('q'):  #判断是哪一个键按下
        break
cv2.destroyAllWindows()