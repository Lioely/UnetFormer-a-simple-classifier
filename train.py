from torch.utils.data import Dataset, DataLoader
from make_data_loader import make_annotation,MyDataset
import torch
import math
import torch.nn as nn
from tqdm import tqdm
import sys
from torch.optim.lr_scheduler import LambdaLR
from Unetformer import Unetformer
import torch.optim.lr_scheduler as lr_scheduler

train_dataset = MyDataset("Fruit360", "data/Fruit360/train_annotation.csv", "train")
test_dataset = MyDataset("Fruit360", "data/Fruit360/test_annotation.csv", "test")
train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)
val_num = len(test_dataloader)
train_steps = len(train_dataloader)

device = "cuda" if torch.cuda.is_available() else "cpu"

print(device)
#self, image_size, input_channels, model_channels, num_classes, dropout=.0, resnetBlock=None
model = Unetformer((100,100),3,64,num_classes=11)
model.to(device)
epochs = 6
loss = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=5e-4,momentum=0.9,weight_decay=0.0005)
lf = lambda x: ((1 + math.cos(x * math.pi / 30)) / 2) * (1 - 0.1) + 0.1 # cosine
scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
lloss =[]
best_acc = 0
accc = []
all_iters = 0
for epoch in range(epochs):
  running_loss = 0.0
  cnt = 0
  iters=0
  train_bar = tqdm(train_dataloader, file=sys.stdout)
  for i, images in enumerate(train_bar):
    model.train()
    image, label = images["rgbd"], images["label"]
    image = image.to(device)
    optimizer.zero_grad()
    cls_pro = model(image).to(device).float()
    #softmax = nn.Softmax(dim=1)
    #pred = torch.argmax(softmax(cls_pro), dim=1).float()
    #label = label.to(device).float()
    label = label.type(torch.LongTensor).to(device)
    total_loss = loss(cls_pro, label)
    running_loss+=total_loss.item()
    #total_loss.requires_grad = True
    total_loss.backward()
    optimizer.step()
    train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,epochs,total_loss)
    iters+=1
    all_iters +=1
    lloss.append(total_loss.item())
  scheduler.step()
  print(f"epoch:{epoch} Loss:{running_loss/iters}")
  
  
  # validate
  model.eval()
  acc = 0.0  # accumulate accurate number / epoch
  with torch.no_grad():
      val_bar = tqdm(test_dataloader, file=sys.stdout)
      for val_data in val_bar:
          val_images, val_labels = val_data["rgbd"], val_data["label"]
          outputs = model(val_images.to(device))
          # loss = loss_function(outputs, test_labels)
          #softmax = nn.Softmax(dim=1)
          #predict_y = torch.argmax(softmax(outputs), dim=1).float()
          predict_y = torch.max(outputs, dim=1)[1]
          acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

          val_bar.desc = "valid epoch[{}/{}]".format(epoch + 1,epochs)

  val_accurate = acc / val_num
  print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
        (epoch + 1, running_loss / train_steps, val_accurate))
  accc.append(val_accurate)
  if val_accurate > best_acc:
      best_acc = val_accurate
      torch.save(model.state_dict(), "Unetformer.pth")

print('Finished Training')
