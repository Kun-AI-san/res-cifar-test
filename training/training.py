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
import matplotlib.pyplot as plt

def provide_samples(batch):
    # transforms.
    transform = v2.Compose([
        v2.ToTensor(),
        v2.RandomHorizontalFlip(p=0.25),
        v2.Normalize(mean = [0.49139968, 0.48215827 ,0.44653124], std = [0.24703233, 0.24348505, 0.26158768])
    ])
    tensor_list = []
    tensor_label_list = []
    for sample in batch:
        tensor_list.append(transform(sample['img']))
        tensor_label_list.append(sample['label'])
    image_batch = torch.stack(tensor_list)
    label_batch = torch.tensor(tensor_label_list).view(len(tensor_label_list))
    return image_batch.cuda(), label_batch.cuda()

def train(model, args, optimizer):
    
    criterion = nn.CrossEntropyLoss()
    train_loss_list = []
    test_loss_list = []
    accuracy_list = []
    epoch_list = []
    for i in tqdm(range(args.epochs), "Epoch status"):
        train_dataset = load_dataset('uoft-cs/cifar10', split='train', streaming=True)
        test_dataset = load_dataset('uoft-cs/cifar10', split='test', streaming=True)

        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=provide_samples, drop_last=True)
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=provide_samples, drop_last=True)
        epoch_loss = 0.0
        test_epoch_loss = 0.0
        model.train()
        for image_batch, outputs in train_dataloader:
            optimizer.zero_grad()
            out = model(image_batch)
            loss = criterion(out, outputs)
            loss.backward()
            epoch_loss+=loss.item()/(50000//args.batch_size)
            optimizer.step()
            optimizer.zero_grad()

        accurate = 0
        total = 0
        with torch.no_grad():
            model.eval()
            for image_batch, outputs in test_dataloader:
                out = model(image_batch)
                loss = criterion(out, outputs)
                # print(out.shape, outputs.shape)
                for j, ans in enumerate(torch.argmax(out, dim=1)):
                    if ans == outputs[j]:
                        accurate+=1
                    total+=1
                test_epoch_loss+=loss.item()/(10000//args.batch_size)
        
        train_loss_list.append(epoch_loss)
        test_loss_list.append(test_epoch_loss)
        if len(accuracy_list)>0 and (accurate/total) > max(accuracy_list):
            torch.save(model.state_dict(), 'Res14Mod')
        else:
            torch.save(model.state_dict(), 'Res14Mod')
        accuracy_list.append(accurate/total)
        epoch_list.append(i)
        plt.plot(epoch_list, train_loss_list, label='Training Loss')
        plt.plot(epoch_list, test_loss_list, label='Test Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Train and Test losses')
        plt.legend()
        plt.savefig('losses.png')
        plt.close()
        plt.plot(epoch_list, accuracy_list)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Test set accuracy')
        plt.legend()
        plt.savefig('accuracy.png')
        plt.close()
        
        tqdm.write(f"Epoch {i+1} Train loss: {epoch_loss}, Test loss: {test_epoch_loss}, Accuracy: {accurate/total}")


# Sample run: python ./training/training.py --batch-size 256 --epochs 20
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch-size",
        type=int,
        required=True
    )
    parser.add_argument(
        "--epochs",
        type=int,
        required=True
    )
    args = parser.parse_args()
    model = Res14().cuda()
    try:
        state = torch.load('./Res14Mod')
        model.load_state_dict(state)
    except Exception as e:
        print("no saved file. continue training from scratch.")
    optimizer = torch.optim.Adam(model.parameters())
    train(model, args, optimizer)