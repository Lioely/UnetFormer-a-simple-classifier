# UnetFormer-a-simple-classifier


![UnetFormer](https://github.com/Lioely/UnetFormer-a-simple-classifier/blob/main/unetformer.png)


首先下载数据集：\
Friut360，Flower\
https://www.kaggle.com/datasets/moltean/fruits \
https://www.kaggle.com/datasets/batoolabbas91/flower-photos-by-the-tensorflow-team \
分别放到data文件夹里面的Flower，Fruit文件夹里，运行data文件夹里面的make_data_loader.py,生成数据的annotation

再安装必要的库
! pip install xformers
! pip install einops


然后运行Trian.py,数据集路径之类的需要在train.py里面自行调整，只是一个简单的小项目，就没有用param了。\
trian.py里面训练的是Fruit360数据集，如果你想要训练Flower数据集或者其他数据集，稍微改一下路径就可以了。\
还有就是，由于文件路径不同，可能会出现需要微调的Bug，但是模型代码是绝对不存在问题！

UnetFormer在Fruit360数据集上，batchsize=128大概需要的显存是58G。在Flower数据集上，batchsize=32大概需要的显存是64G，根据自己的设备选择合适的batchsize吧。我用的A100 80G的显卡，6个epoches在两个数据集上分布train了1h和4h。

