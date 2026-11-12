import os
from torch.utils.data import Dataset
from PIL import Image
import json
import random
from collections import defaultdict
'''
https://towardsdatascience.com/downloading-and-using-the-imagenet-dataset-with-pytorch-f0908437c4be
'''


class ImageNet(Dataset):
    def __init__(self, root, split, transform=None):
        self.samples = []
        self.targets = []
        self.transform = transform
        self.syn_to_class = {}
        with open(os.path.join(root, "imagenet_class_index.json"), "rb") as f:
                    json_file = json.load(f)
                    for class_id, v in json_file.items():
                        self.syn_to_class[v[0]] = int(class_id)
        with open(os.path.join(root, "ILSVRC2012_val_labels.json"), "rb") as f:
                    self.val_to_syn = json.load(f)
        samples_dir = os.path.join(root, "ILSVRC/Data/CLS-LOC", split)

        class_samples = defaultdict(list)
        for entry in os.listdir(samples_dir):
            if split == "train":
                syn_id = entry
                target = self.syn_to_class[syn_id]
                syn_folder = os.path.join(samples_dir, syn_id)
                for sample in os.listdir(syn_folder):
                    sample_path = os.path.join(syn_folder, sample)
                    # self.samples.append(sample_path)
                    # self.targets.append(target)
                    class_samples[target].append(sample_path) ####
            elif split == "val":
                syn_id = self.val_to_syn[entry]
                target = self.syn_to_class[syn_id]
                sample_path = os.path.join(samples_dir, entry)
                self.samples.append(sample_path)
                self.targets.append(target)

        # Select only 130 samples per class #이건 원래 없었음
        for target, paths in class_samples.items():
            selected_paths = random.sample(paths, 130)  # Randomly sample 130 paths
            self.samples.extend(selected_paths)
            self.targets.extend([target] * 130)

    def __len__(self):
            return len(self.samples)
    def __getitem__(self, idx):
            x = Image.open(self.samples[idx]).convert("RGB")
            if self.transform:
                x = self.transform(x)
            return x, self.targets[idx]




if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from torchvision import transforms
    import torch
    import torchvision
    from tqdm import tqdm

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    val_transform = transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),
                ]
            )
    dataset = ImageNet("C:/TransUnrecon/data/ImageNet/imagenet-object-localization-challenge/", "val", val_transform)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False,pin_memory=True)

    for x, y in tqdm(dataloader):
        print(x.size)