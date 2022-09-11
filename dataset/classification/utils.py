# Lib
import cv2
import torch
import numpy as np

def imagenet_collate_fn(data):
    batch_size = len(data)
    img_size = [batch_size, * data[0]['image'].shape]
    img_dtype = data[0]['image'].dtype

    images = torch.zeros(size=img_size, dtype=img_dtype)
    labels = torch.full(size=[batch_size], fill_value=-1, dtype=torch.long)
    sample_ids = torch.zeros(size=[batch_size], dtype=torch.long)
    valid_indexes = []

    for i, batch_i in enumerate(data):
        label_i = batch_i.pop("label")
        images[i] = batch_i.pop("image")
        labels[i] = label_i
        sample_ids[i] = batch_i.pop("sample_id")
        if label_i != -1:
            valid_indexes.append(i)
    
    valid_indexes = torch.tensor(valid_indexes, dtype=torch.long)
    images = torch.index_select(images, dim=0, index=valid_indexes)
    labels = torch.index_select(labels, dim=0, index=valid_indexes)
    sample_id = torch.index_select(sample_ids, dim=0, index=valid_indexes)

    channels_last = getattr(opts, "common.channels_last", False)
    if channels_last:
        images = images.to(memory_format=torch.channel_last)

    return {"images": images, "label": labels, "sample_id": sample_id, "on_gpu": images.is_cuda}