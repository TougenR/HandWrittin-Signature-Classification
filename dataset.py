# Description: This file is used to create a dataset class for the signature
from torch.utils.data import Dataset, DataLoader
import os
from torchvision.transforms import Compose, ToTensor, Resize, Normalize
from PIL import Image

class SignatureDataset(Dataset):
    def __init__(self, root = "Dataset_Signature_Final/Dataset", train=True, transform=None):
        super().__init__()
        self.transform = transform
        self.images = []
        self.labels = []
        if train:
            # 0 is forge handwriting signature, 1 is real handwriting signature
            self.data_paths = ["dataset{}".format(i) for i in range(1, 5)]
            for filename in self.data_paths:
                data_file = os.path.join(root, filename)
                forge_images_folder = os.path.join(data_file, "forge")
                real_images_folder = os.path.join(data_file, "real")

                # get every item in forge images folder
                forge_files = [f for f in os.listdir(forge_images_folder) if f.endswith(".png")]
                for filename in forge_files:
                    image_path = os.path.join(forge_images_folder, filename)
                    self.images.append(image_path)
                    self.labels.append(0)
                
                # get every item in real images folder
                real_files = [f for f in os.listdir(real_images_folder) if f.endswith(".png")]
                for filename in real_files:
                    image_path = os.path.join(real_images_folder, filename)
                    self.images.append(image_path)
                    self.labels.append(1)
        else:
            self.data_paths = "sample_Signature/sample_Signature"
            forge_images_folder = os.path.join(self.data_paths, "forged")
            genuine_images_folder = os.path.join(self.data_paths, "genuine")
            
            # get every item in forged folder
            forge_files = [f for f in os.listdir(forge_images_folder) if f.endswith(".png")]
            for filename in forge_files:
                image_path = os.path.join(forge_images_folder, filename)
                self.images.append(image_path)
                self.labels.append(0)
            
            # get every item in genuine folder
            genuine_files = [f for f in os.listdir(genuine_images_folder) if f.endswith(".png")]
            for filename in genuine_files:
                image_path = os.path.join(genuine_images_folder, filename)
                self.images.append(image_path)
                self.labels.append(1)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        image_path = self.images[index]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = self.labels[index]
        return image, label

if __name__ == "__main__":
    transform = Compose([
        Resize((224, 224)),
        ToTensor(),
        Normalize([0.48235, 0.45882, 0.40784], [0.00392156862745098, 0.00392156862745098, 0.00392156862745098])
    ])
    dataset = SignatureDataset(train=True, transform=transform)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=8,
        num_workers=0,
        shuffle=False,
        drop_last=False
    )

    for images, labels in dataloader:
        print(images.shape, labels.shape)



    