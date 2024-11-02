import os
from os import path
import tarfile
from xml.etree import ElementTree
from typing import Optional, List

import torch

from torch.utils.data import Dataset

from torchvision.io import decode_image
from torchvision.datasets.utils import download_url
from torchvision.tv_tensors import BoundingBoxes

from torchvision.transforms.v2 import Transform

import scipy.io


class DogeDataset(Dataset):
    DOWNLOAD_PATH = "http://vision.stanford.edu/aditya86/ImageNetDogs"

    README_URL = f"{DOWNLOAD_PATH}/README.txt"
    IMAGES_URL = f"{DOWNLOAD_PATH}/images.tar"
    ANNOTATIONS_URL = f"{DOWNLOAD_PATH}/annotation.tar"
    LISTS_URL = f"{DOWNLOAD_PATH}/lists.tar"

    IMAGES_TAR = path.basename(IMAGES_URL)
    ANNOTATIONS_TAR = path.basename(ANNOTATIONS_URL)
    LISTS_TAR = path.basename(LISTS_URL)

    README_PATH = "README.txt"
    IMAGES_FOLDER_NAME = "Images"
    ANNOTATIONS_FOLDER_NAME = "Annotation"

    TRAIN_LIST = "train_list.mat"
    TEST_LIST = "test_list.mat"

    IMAGE_MODE = "RGB"
    BOX_FORMAT = "XYXY"

    CLASS_MAPPING = [
        "Chihuaha",
        "Japanese Spaniel",
        "Maltese Dog",
        "Pekinese",
        "Shih-Tzu",
        "Blenheim Spaniel",
        "Papillon",
        "Toy Terrier",
        "Rhodesian Ridgeback",
        "Afghan Hound",
        "Basset Hound",
        "Beagle",
        "Bloodhound",
        "Bluetick",
        "Black-and-tan Coonhound",
        "Walker Hound",
        "English Foxhound",
        "Redbone",
        "Borzoi",
        "Irish Wolfhound",
        "Italian Greyhound",
        "Whippet",
        "Ibizian Hound",
        "Norwegian Elkhound",
        "Otterhound",
        "Saluki",
        "Scottish Deerhound",
        "Weimaraner",
        "Staffordshire Bullterrier",
        "American Staffordshire Terrier",
        "Bedlington Terrier",
        "Border Terrier",
        "Kerry Blue Terrier",
        "Irish Terrier",
        "Norfolk Terrier",
        "Norwich Terrier",
        "Yorkshire Terrier",
        "Wirehaired Fox Terrier",
        "Lakeland Terrier",
        "Sealyham Terrier",
        "Airedale",
        "Cairn",
        "Australian Terrier",
        "Dandi Dinmont",
        "Boston Bull",
        "Miniature Schnauzer",
        "Giant Schnauzer",
        "Standard Schnauzer",
        "Scotch Terrier",
        "Tibetan Terrier",
        "Silky Terrier",
        "Soft-coated Wheaten Terrier",
        "West Highland White Terrier",
        "Lhasa",
        "Flat-coated Retriever",
        "Curly-coater Retriever",
        "Golden Retriever",
        "Labrador Retriever",
        "Chesapeake Bay Retriever",
        "German Short-haired Pointer",
        "Vizsla",
        "English Setter",
        "Irish Setter",
        "Gordon Setter",
        "Brittany",
        "Clumber",
        "English Springer Spaniel",
        "Welsh Springer Spaniel",
        "Cocker Spaniel",
        "Sussex Spaniel",
        "Irish Water Spaniel",
        "Kuvasz",
        "Schipperke",
        "Groenendael",
        "Malinois",
        "Briard",
        "Kelpie",
        "Komondor",
        "Old English Sheepdog",
        "Shetland Sheepdog",
        "Collie",
        "Border Collie",
        "Bouvier des Flandres",
        "Rottweiler",
        "German Shepard",
        "Doberman",
        "Miniature Pinscher",
        "Greater Swiss Mountain Dog",
        "Bernese Mountain Dog",
        "Appenzeller",
        "EntleBucher",
        "Boxer",
        "Bull Mastiff",
        "Tibetan Mastiff",
        "French Bulldog",
        "Great Dane",
        "Saint Bernard",
        "Eskimo Dog",
        "Malamute",
        "Siberian Husky",
        "Affenpinscher",
        "Basenji",
        "Pug",
        "Leonberg",
        "Newfoundland",
        "Great Pyrenees",
        "Samoyed",
        "Pomeranian",
        "Chow",
        "Keeshond",
        "Brabancon Griffon",
        "Pembroke",
        "Cardigan",
        "Toy Poodle",
        "Miniature Poodle",
        "Standard Poodle",
        "Mexican Hairless",
        "Dingo",
        "Dhole",
        "African Hunting Dog",
    ]

    def __init__(
        self,
        root_path: str,
        train: bool = True,
        transformer: Optional[Transform] = None,
    ):
        self._check_dataset_or_download(root_path)

        list_path = path.join(root_path, self.TRAIN_LIST if train else self.TEST_LIST)

        list_file = scipy.io.loadmat(list_path)

        annotations = [item[0][0] for item in list_file["annotation_list"]]

        annotations_path = path.join(root_path, self.ANNOTATIONS_FOLDER_NAME)

        image_filenames = [f"{annotation}.jpg" for annotation in annotations]
        bounding_boxes = [
            self._get_boxes(path.join(annotations_path, annotation))
            for annotation in annotations
        ]

        labels = [item[0] - 1 for item in list_file["labels"]]

        images_path = path.join(root_path, self.IMAGES_FOLDER_NAME)

        self.image_filenames = image_filenames
        self.labels = labels
        self.bounding_boxes = bounding_boxes
        self.transformer = transformer
        self.images_path = images_path

    @property
    def num_classes(self) -> int:
        return len(self.CLASS_MAPPING)

    def count_labels(self) -> dict[int, int]:
        counts = {}

        for label in self.labels:
            if label not in counts:
                counts[label] = 1

            else:
                counts[label] += 1

        return counts

    def _check_dataset_or_download(self, root_path: str) -> None:
        readme_path = path.join(root_path, self.README_PATH)
        images_path = path.join(root_path, self.IMAGES_FOLDER_NAME)
        annotations_path = path.join(root_path, self.ANNOTATIONS_FOLDER_NAME)
        train_list = path.join(root_path, self.TRAIN_LIST)
        test_list = path.join(root_path, self.TEST_LIST)

        if not path.exists(readme_path):
            download_url(self.README_URL, root_path)

        if not path.exists(images_path):
            download_url(self.IMAGES_URL, root_path)

            file_path = path.join(root_path, self.IMAGES_TAR)

            with tarfile.open(file_path, "r") as file:
                file.extractall(root_path)

            os.remove(file_path)

        if not path.exists(annotations_path):
            download_url(self.ANNOTATIONS_URL, root_path)

            file_path = path.join(root_path, self.ANNOTATIONS_TAR)

            with tarfile.open(file_path, "r") as file:
                file.extractall(root_path)

            os.remove(file_path)

        if not path.exists(train_list) or not path.exists(test_list):
            download_url(self.LISTS_URL, root_path)

            file_path = path.join(root_path, self.LISTS_TAR)

            with tarfile.open(file_path, "r") as file:
                file.extractall(root_path)

            os.remove(file_path)

    def _get_boxes(self, annotation_path: str) -> List[List[float]]:
        tree = ElementTree.parse(annotation_path)

        boxes = []

        for obj in tree.iter("object"):
            bndbox = obj.find("bndbox")

            if not bndbox:
                continue

            xmin = float(bndbox.find("xmin").text)
            ymin = float(bndbox.find("ymin").text)
            xmax = float(bndbox.find("xmax").text)
            ymax = float(bndbox.find("ymax").text)

            boxes.append([xmin, ymin, xmax, ymax])

        return boxes

    def __getitem__(self, index: int):
        filename, label, boxes = (
            self.image_filenames[index],
            self.labels[index],
            self.bounding_boxes[index],
        )

        image_path = path.join(self.images_path, filename)

        image = decode_image(image_path, mode=self.IMAGE_MODE)

        boxes = BoundingBoxes(
            boxes, format=self.BOX_FORMAT, canvas_size=image.shape[-2:]
        )

        if self.transformer:
            image, boxes = self.transformer(image, boxes)

        return image, label, boxes[0]

    def __len__(self):
        return len(self.image_filenames)
