# UnetFormer-a-simple-classifier

https://github.com/Lioely/UnetFormer-a-simple-classifier/blob/main/unetformer.png

首先下载数据集：
Friut100，Flower
https://www.kaggle.com/datasets/moltean/fruits
https://www.kaggle.com/datasets/batoolabbas91/flower-photos-by-the-tensorflow-team
分别放到Flower，Fruit文件夹里面，运行文件及里面的make_data_loader.py,生成数据的annotation

再安装必要的库
! pip install xformers
! pip install einops


然后运行Trian.py,数据集路径之类的需要在train.py里面自行调整，只是一个简单的小项目，就没有用param了。
