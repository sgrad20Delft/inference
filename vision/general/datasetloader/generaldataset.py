import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import transforms
import tensorflow_datasets as tfds
from datasets import load_dataset
from pycocotools.coco import COCO

class GeneralDataset(Dataset):
    def __init__(self, dataset_type, source,task_type, transform=None):
        self.task_type = task_type
        self.dataset_type = dataset_type
        self.source = source  # path or dataset name
        self.transform = transform or transforms.ToTensor()
        self.data = []

        if dataset_type == "huggingface":
            self.dataset = load_dataset(source)

        elif dataset_type == "pytorch":
            self.image_dir = source["image_dir"]
            self.annotation_file = source.get("annotation_file")  # optional
            self.image_paths = sorted([
                os.path.join(self.image_dir, f) for f in os.listdir(self.image_dir)
                if f.lower().endswith((".jpg", ".png", ".jpeg"))
            ])

            if self.annotation_file and os.path.exists(self.annotation_file):
                import json
                with open(self.annotation_file, 'r') as f:
                    self.labels_dict = json.load(f)
            else:
                self.labels_dict = {}


        elif dataset_type == "tensorflow":
            self.dataset = tfds.load(name=source, as_supervised=True)
            self.iterator = iter(tfds.as_numpy(self.dataset))

        elif dataset_type == "coco":
            self.coco = COCO(source["annotation_file"])
            self.image_dir = source["image_dir"]
            self.image_ids = self.coco.getImgIds()

        else:
            raise ValueError("Unsupported dataset_type")

    def __len__(self):
        if self.dataset_type in ["huggingface", "coco"]:
            return len(self.dataset)
        elif self.dataset_type == "tensorflow":
            return len(list(tfds.load(name=self.source)))

    def __getitem__(self, idx):
        if self.dataset_type == "huggingface":
            sample = self.dataset[idx]
            image = sample["image"]
            label = sample.get("label", -1)
            if self.transform:
                image = self.transform(image)
            return image, label
        elif self.dataset_type == "pytorch":
            img_path = self.image_paths[idx]
            image = Image.open(img_path).convert("RGB")
            label = self.labels_dict.get(os.path.basename(img_path), -1)
            if self.transform:
                image = self.transform(image)
            return image, label


        elif self.dataset_type == "tensorflow":
            sample = next(self.iterator)
            image, label = sample
            image = Image.fromarray(image)
            image = self.transform(image)
            return image, label

        elif self.dataset_type == "coco":
            img_id = self.image_ids[idx]
            img_info = self.coco.loadImgs(img_id)[0]
            image = Image.open(os.path.join(self.image_dir, img_info["file_name"])).convert("RGB")
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            anns = self.coco.loadAnns(ann_ids)
            boxes = [ann["bbox"] for ann in anns]
            labels = [ann["category_id"] for ann in anns]
            return self.transform(image), {"boxes": boxes, "labels": labels}
