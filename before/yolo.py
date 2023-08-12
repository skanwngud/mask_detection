from glob import glob

img_list=glob('c:/data/dataset/mask/export/images/*.jpg') # 이미지로드
print(len(img_list)) # 1294

from sklearn.model_selection import train_test_split

train_img_list, val_img_list=train_test_split(
    img_list,
    train_size=0.8,
    random_state=23
)

with open('c:/data/dataset/mask/train.txt', 'w') as f:
    f.write('\n'.join(train_img_list)+'\n')

with open('c:/data/dataset/mask/val.txt', 'w') as f:
    f.write('\n'.join(val_img_list)+'\n')

import yaml
with open('c:/data/dataset/pistols/data.yaml', 'r', encoding='utf-8') as f:
    data=yaml.safe_load(f)

print(data)

data['train']='c:/data/dataset/mask/train.txt'
data['val']='c:/data/dataset/mask/val.txt'
data['names']=['pareidolia', 'mask', 'face']
data['nc']=str(3)

with open('c:/data/dataset/mask/data.yaml', 'w') as f:
    yaml.dump(data, f)

print(data)