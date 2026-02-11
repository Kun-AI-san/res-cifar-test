import sys
import os
sys.path.insert(1, os.getcwd())
import torch
import torch.nn as nn
from datasets import load_dataset
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from model.res14 import Res14
import argparse
from tqdm import tqdm
from functools import partial
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def provide_samples(batch):
    # transforms.
    transform = v2.Compose([
        v2.ToImage(), 
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean = [0.49139968, 0.48215827 ,0.44653124], std = [0.24703233, 0.24348505, 0.26158768])
    ])
    tensor_list = []
    tensor_label_list = []
    for sample in batch:
        tensor_list.append(transform(sample['img']))
        tensor_label_list.append(sample['label'])
    image_batch = torch.stack(tensor_list)
    label_batch = torch.tensor(tensor_label_list).view(len(tensor_label_list))
    if torch.cuda.is_available():
        return image_batch.cuda(), label_batch.cuda()
    return image_batch, label_batch


def test(model, args):
    
    criterion = nn.CrossEntropyLoss()
    test_dataset = load_dataset('uoft-cs/cifar10', split='test', streaming=True).take(args.batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=partial(provide_samples), drop_last=True)
    epoch_loss = 0.0
    with torch.no_grad():
        model.eval()
        for image_batch, outputs in test_dataloader:
            out = model(image_batch)
            cm = confusion_matrix(outputs.cpu(), y_pred=torch.argmax(out.cpu(), dim=-1))
            ConfusionMatrixDisplay(cm).plot().figure_.savefig('confusion.png')

#sample run python ./test/test.py --batch-size 1024
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch-size",
        type=int,
        required=True
    )
    args = parser.parse_args()
    if torch.cuda.is_available():
        model = Res14().cuda()
    else:
        model = Res14()
    try:
        state = torch.load('./Res14Mod')
        model.load_state_dict(state)
    except Exception as e:
        print("no saved file. continue training from scratch.")

    test(model, args)
