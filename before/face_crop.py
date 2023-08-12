import glob
import numpy as np
import cv2
import os

face_cascade=cv2.CascadeClassifier('e:/datasets/train_face/edit/haarcascades/haarcascade_frontalface_default.xml')
eye_casecade=cv2.CascadeClassifier('e:/datasets/train_face/edit/haarcascades/haarcascade_eye.xml')

# mask_list=glob.glob('e:/datasets/train_face/train_mask/*.jpg')
# nomask_list=glob.glob('e:/datasets/train_face/train_nomask/*.jpg')
# filepath='e:/datasets/train_face/train_mask/*.jpg'
# filepath2='e:/datasets/train_face/train_nomask/*.jpg'

# for filepath in glob.glob(filepath):
#     filepath_r=filepath.replace('(', '')
#     filepath_r=filepath.replace(')', '')
#     os.rename(filepath, filepath_r)

# for filepath2 in glob.glob(filepath2):
#     filepath2_r=filepath2.replace('(', '')
#     filepath2_r=filepath2.replace(')', '')
#     os.rename(filepath2, filepath2_r)

# count=1
# # img=cv2.imread('e:/datasets/train_face/train_mask/mask_man34.jpg')

# # for i in nomask_list:
# img=cv2.imread('e:/datasets/train_face/train_nomask/no_mask1.jpg')
# img_gray=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# faces=face_cascade.detectMultiScale(img_gray, 1.3, 5)

# import matplotlib.pyplot as plt

# plt.imshow(img_gray)
# plt.show()

# for (x,y,w,h) in faces:
#     cropped=img[y-int(h/4):y+h+int(h/4), x-int(w/4):x+w+int(w/4)]
#     cv2.imwrite('e:/datasets/train_face/train_nomask/no_mask_cropped1.jpg', cropped)
#     count+=1
#         # cv2.rectangle(img, (x,y), (x+w, y+h), (255, 0, 0), 2)
#         # roi_gray=img_gray[y:y+h, x:x+w]
#         # roi_color=img[y:y+h, x:x+w]
#         # eyes=eye_casecade.detectMultiScale(roi_gray)
#         # for (ex, ey, ew, eh) in eyes:
#         #     cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

# # for i in mask_list:
# # img=cv2.imread(i)
# # img_gray=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# # faces=face_cascade.detectMultiScale(img, 1.3, 5)
# # for (x,y,w,h) in faces:
# #     cv2.rectangle(img, (x,y), (x+w, y+h), (255, 0, 0), 2)
# #     cropped=img[y-int(h/4):y+h+int(h/4), x-int(w/4):x+w+int(w/4)]
# #     cv2.imwrite('e:/datasets/train_face/train_mask/mask_cropped'+str(count)+'.jpg', cropped)
# #     roi_gray=img_gray[y:y+h, x:x+w]
# #     roi_color=img[y:y+h, x:x+w]
# #     eyes=eye_casecade.detectMultiScale(roi_gray)
# #     for (ex, ey, ew, eh) in eyes:
# #         cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

