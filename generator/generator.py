import os
from glob import glob
from tqdm import tqdm


def create_anno(label_info, mode):
    txt_path = os.getcwd() + f'/generator/{mode}.txt'
    for label in tqdm(label_info, desc='Converting txt file'):
        with open(txt_path, 'a') as f:
            info = f'{label} \n'
            f.write(info)

if __name__ == '__main__':
    tr_label = [os.path.basename(file) for file in glob(os.getcwd() + '/imagenet-mini/train/*')]
    val_label = [os.path.basename(file) for file in glob(os.getcwd() + '/imagenet-mini/val/*')]

    create_anno(tr_label, mode='train')
    create_anno(val_label, mode='val')

