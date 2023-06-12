import os
from torchvision import transforms
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from PIL import Image


def choose_transform(datasets_name):
    data_transform = {transforms.ToTensor()}
    if datasets_name == "Fruit100":
        data_transform = {
            "train": transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
            "test": transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}
    elif datasets_name == "flower_photos":
        data_transform = {
            "train": transforms.Compose([transforms.Resize(256),
                                       transforms.CenterCrop(224),
                                         transforms.RandomHorizontalFlip(),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
            "test": transforms.Compose([transforms.Resize(256),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}
    return data_transform


def make_annotation(split, datasets_name):
    root_dir = os.getcwd()
    annotation_path = None
    if datasets_name == "Fruit100":
        df = pd.DataFrame()
        path = []
        label = []
        annotation_path = os.path.join(root_dir, "Fruit360")
        datasets_path = os.path.join(annotation_path, split)
        fruit_class = list(set(cla.split()[0] for cla in os.listdir(datasets_path)
                               if os.path.isdir(os.path.join(datasets_path, cla))))
        fruit_path = [cla for cla in os.listdir(datasets_path)
                      if os.path.isdir(os.path.join(datasets_path, cla))]
        for cla in fruit_path:
            cla_path = os.path.join(datasets_path, cla)
            for img_name in os.listdir(cla_path):
                img_path = os.path.join(cla_path, img_name)
                path.append(img_path)
                label.append(fruit_class.index(
                    cla.split()[0]
                )
                )
        df['data'] = path
        df['label'] = label
        print(os.path.join(annotation_path, f"{split}_annotation.csv"))
        df.to_csv(os.path.join(annotation_path, f"{split}_annotation.csv"), index=False)

    elif datasets_name == "flower_photos":
        df = pd.DataFrame()
        path = []
        label = []
        annotation_path = os.path.join(root_dir, "flower_photos")
        datasets_path = os.path.join(annotation_path, split)
        flower_class = [cla for cla in os.listdir(datasets_path)
                        if os.path.isdir(os.path.join(datasets_path, cla))]

        for i, cla in enumerate(flower_class):
            cla_path = os.path.join(datasets_path, cla)
            for img_name in os.listdir(cla_path):
                img_path = os.path.join(cla_path, img_name)
                path.append(img_path)
                label.append(i)
        df['data'] = path
        df['label'] = label
        print(os.path.join(annotation_path, f"{split}_annotation.csv"))
        df.to_csv(os.path.join(annotation_path, f"{split}_annotation.csv"), index=False)


"""
make_annotation("train", "Fruit360")
make_annotation("test", "Fruit360")
make_annotation("train", "flower_photos")
make_annotation("test", "flower_photos")
"""




class MyDataset(Dataset):
    # nyu depth dataset
    def __init__(self, dataset_name, csv_file, split):
        """
        Args:
            dataset_name(string): Fruit100 or flower_photos
            csv_file (string): Path to the csv file with annotations.
        """
        self.rgb_frame = pd.read_csv(csv_file)
        self.split = split
        self.dataset_name = dataset_name

    def __len__(self):
        return len(self.rgb_frame)

    def __getitem__(self, idx):
        data_transform = choose_transform(self.dataset_name)
        rgb_name = self.rgb_frame.iloc[idx, 0]
        rgb_label = self.rgb_frame.iloc[idx, 1]
        with open(rgb_name, 'rb') as fRgb:
            rgb_image = Image.open(rgb_name).convert('RGB')

        rgb_image = data_transform[self.split](rgb_image)

        sample = {'rgbd': rgb_image, 'label': rgb_label}
        return sample
