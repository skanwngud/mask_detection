from glob import glob # 여러장의 이미지 파일을 로드하는 라이브러리
import torch

img_list=glob('c:/data/dataset/pistols/export/images/*.jpg') # 이미지 로드
print(len(img_list)) # 이미지 갯수 2971 

from sklearn.model_selection import train_test_split

train_img_list, val_img_list=train_test_split(
    img_list,
    train_size=0.8,
    random_state=23
)

with open('c:/data/dataset/pistols/train.txt', 'w') as f:
    f.write('\n'.join(train_img_list) + '\n') # 해당 데이터를 txt 화 시킴

with open('c:/data/dataset/pistols/val.txt', 'w') as f:
    f.write('\n'.join(val_img_list) + '\n')

import yaml # yaml 파일 로드
with open('c:/data/dataset/pistols/data.yaml', 'r', encoding='utf-8') as f:
    data=yaml.safe_load(f)
# cp949 인코딩은 오류가 뜰 수 있으므로 utf-8 로 지정해준다

print(data) # {'train': '../train/images', 'val': '../valid/images', 'nc': 1, 'names': ['pistol']}

data['train']='c:/data/dataset/pistols/train.txt'
data['val']='c:/data/dataset/pistols/val.txt'

with open('c:/data/dataset/pistols/data.yaml', 'w') as f:
    yaml.dump(data, f)
# 기존에 있던 yaml 파일의 train, val 부분을 해당 경로로 바꿔줌
# {'train': 'c:/data/dataset/pistols/train.txt', 'val': 'c:/data/dataset/pistols/val.txt', 'nc': 1, 'names': ['pistol']

print(data)

# $ python train.py --img 640 --batch 16 --epochs 5 --data coco128.yaml --weights yolov5s.pt
# 해당 내용은 터미널에서 실행, ' $ ' 는 터미널에서 실행시킨다는 의미이다. ($ 를 뺀 뒤의 내용을 붙여넣기)

from IPython.display import Image
import os