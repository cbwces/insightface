import cv2
import numpy as np
import datetime
import glob
from retinaface import RetinaFace

img_path = "/home/chenbw/project/datasets/"
imgs = glob.glob(img_path+"*.jpg")  
thresh = 0.8
scales = [1024, 1980]
flip = False
# count = 1
count = len(imgs)
gpuid = 0
detector = RetinaFace('./model/R50', 0, gpuid, 'net3')

def load_img(path=imgs):    #返回图片和缩放因子
    
    scaling_factors = []
    img_set = []
    
    for single_img in path:
        img = cv2.imread(single_img)
    #     print(img.shape)
        im_shape = img.shape
        target_size = scales[0]
        max_size = scales[1]
        im_size_min = np.min(im_shape[0:2])
        im_size_max = np.max(im_shape[0:2])

    #     im_scale = 1.0
    #     if im_size_min>target_size or im_size_max>max_size:
        im_scale = float(target_size) / float(im_size_min)  #图片scale
    # prevent bigger axis from being more than max_size:
        if np.round(im_scale * im_size_max) > max_size:
            im_scale = float(max_size) / float(im_size_max)
        scaling_factors.append(im_scale)
        img_set.append(img)
        
    return  img_set, scaling_factors

def train_data(img_set, scaling_factors, flip=flip)
    for img_num, single_img in enumerate(img_set):
        faces, landmarks = detector.detect(single_img, thresh, scales=[scaling_factors[img_num]], do_flip=flip)
#         print(c, faces.shape, landmarks.shape)

        if faces is not None:   #检测到有人脸
          print('find', faces.shape[0], 'faces')    #人脸数量
          for i in range(faces.shape[0]):
            #print('score', faces[i][4])
            box = faces[i].astype(np.int)
            #color = (255,0,0)
            color = (0,0,255)
            cv2.rectangle(single_img, (box[0], box[1]), (box[2], box[3]), color, 2)    #构建人脸框
            if landmarks is not None:   #找到五官点
              landmark5 = landmarks[i].astype(np.int)
              #print(landmark.shape)
              for l in range(landmark5.shape[0]):
                color = (0,0,255)
                if l==0 or l==3:
                  color = (0,255,0)
                cv2.circle(single_img, (landmark5[l][0], landmark5[l][1]), 1, color, 2)

      filename = './detector_test' + img_num + '.jpg'
      print('writing', filename)
      cv2.imwrite(filename, single_img)
   
if __name__ == "__main__":
    
    img_set, scaling_factors = load_img()
    train_data(img_set=img_set, scaling_factors=scaling_factors)
