# Lib
import cv2
import glob
import tqdm
import torch
import numpy as np

def visualize(images, classes, batch_idx=0):
    '''
    batch data visualize
    if you use Linux OS(Docker):
        save image file in docs dir and check
    '''
    img = images[batch_idx].numpy()
    img = (np.transpose(img, (1, 2, 0)) * 255.).astype(np.uint8).copy()
    label = classes[batch_idx].numpy()

    # cv2.imshow('img', img)
    # cv2.waitKey(0)
    cv2.imwrite('/home/torch/Classification/docs/class'+str(label)+'.JPEG', img)


def create_annot(label_info, mode):
    txt_path = os.getcwd() + f'/generator/{mode}.txt'
    for label in tqdm(label_info, desc='Converting txt file'):
        with open(txt_path, 'a') as f:
            info = f'{label} \n'
            f.write(info)


if __name__ == '__main__':
    train_label = [os.basename(file) for file in glob(os.getcwd() + '/imagenet-ini/train/*')]
    val_label = [os.basename(file) for file in glob(os.getcwd() + '/imagenet-mini/val/*')]

    create_annot(train_label, mode='train')
    create_annot(valid_label, mode='val')