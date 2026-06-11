from glob import glob as gg
import os
import torch
from torch.utils.data import Dataset, WeightedRandomSampler, DataLoader
import numpy as np
from torchvision import transforms
from scipy import ndimage
import random


class Loader(Dataset):
    def __init__(self, dataset, labels=np.empty(0), online_aug=False, oversampling=False, undersampling=False,
                 normalization="mean", nchannels=1, size=128):
        """
        A class for dataset loading, preprocessing, and balancing.
        :param dataset: List of file paths for the dataset.
        :param labels: List of corresponding labels.
        :param train: Whether the loader is for training or not.
        :param normalization: Normalization method, choices are "max" or "mean".
        """
        self.dataset = dataset
        self.labels = labels
        self.normalization = normalization
        self.nchannels = nchannels
        self.oversampling = oversampling
        self.undersampling = undersampling

        if self.undersampling and self.oversampling:
            raise('Undersampling and oversampling cannot both performed.')

        if undersampling:
            self.dataset, self.labels = self.undersample_dataset()

        if online_aug:
            self.transform = transforms.Compose([
                self.normalize_intensity,
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=(-5, 5)),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.Resize((size, size)),
            ])
        else:
            self.transform = transforms.Compose([
                self.normalize_intensity,
                transforms.Resize((size, size)),
            ])

        if self.oversampling:
            self.weights = self.calculate_class_weights()  # Class weights for balancing

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        """
        Retrieves an item from the dataset and processes it.

        :param index: Index of the item to retrieve.
        :return: A tuple (processed image tensor, label).
        """
        x_batch = self.dataset[index]
        h, w, c = x_batch.shape
        x_batch = np.reshape(x_batch, (c, h, w))
        x_batch = self.transform(x_batch)
        if self.nchannels== 1:
            x_batch = torch.mean(x_batch, -3)
        elif self.nchannels == 3 and c == 1:
            x_batch = x_batch.repeat(3, 1, 1)
        if self.labels.size:
            label = self.labels[index]
            return x_batch, label
        else:
            return x_batch

    def normalize_intensity(self, img_tensor):
        """
        Normalizes an image tensor based on the selected method.

        :param img_tensor: Input image tensor.
        :return: Normalized image tensor.
        """
        img_tensor = torch.tensor(img_tensor, dtype=torch.float32)
        if self.normalization == "mean":
            mask = img_tensor.ne(0.0)
            desired = img_tensor[mask]
            mean_val, std_val = desired.mean(), desired.std()
            img_tensor = (img_tensor - mean_val) / std_val
        elif self.normalization == "minmax":
            max_val, min_val = img_tensor.max(), img_tensor.min()
            img_tensor = (img_tensor - min_val) / (max_val - min_val)
        return img_tensor

    def calculate_class_weights(self):
        """
        Calculates weights for each sample based on class frequency.
        :return: List of weights for each sample.
        """
        class_counts = np.bincount(self.labels)  # Count occurrences of each class
        total_samples = len(self.labels)
        class_weights = total_samples / (len(class_counts) * class_counts)  # Inverse frequency
        sample_weights = [class_weights[label] for label in self.labels]
        return sample_weights

    def undersample_dataset(self):
        """
        Performs random undersampling to balance the dataset.
        Reduces majority class samples to match the minority class count.

        :return: A tuple (undersampled dataset, undersampled labels).
        """
        unique_classes, class_counts = np.unique(self.labels, return_counts=True)
        min_samples = np.min(class_counts)  # Find the minority class count

        undersampled_indices = []
        for cls in unique_classes:
            cls_indices = np.where(self.labels == cls)[0]  # Get indices for the class
            undersampled_indices.extend(random.sample(list(cls_indices), min_samples))  # Random selection

        undersampled_indices = np.array(undersampled_indices)

        undersampled_dataset = self.dataset[undersampled_indices]
        undersampled_labels = self.labels[undersampled_indices]

        return undersampled_dataset, undersampled_labels

class Loader3D(Dataset):
    def __init__(self, dataset, labels, rescale_resize=False, normalization="max",
                 undersampling=False, oversampling=False, nchannels=1):
        """
        A class for dataset loading, preprocessing, and balancing.
        :param dataset: List of file paths for the dataset.
        :param labels: List of corresponding labels.
        :param rescale_resize: Whether the images are rescaled and resized.
        :param normalization: Normalization method, choices are "max" or "mean".
        """
        self.dataset = dataset
        self.labels = labels
        self.normalization = normalization
        self.rescale_resize = rescale_resize
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            self.normalize_intensity
        ])
        self.oversampling = oversampling
        self.undersampling = undersampling
        self.nchannels = nchannels

        if self.undersampling and self.oversampling:
            raise('Undersampling and oversampling cannot both performed.')

        if undersampling:
            self.dataset, self.labels = self.undersample_dataset()

        if self.oversampling:
            self.weights = self.calculate_class_weights()  # Class weights for balancing

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        """
        Retrieves an item from the dataset and processes it.

        :param index: Index of the item to retrieve.
        :return: A tuple (processed image tensor, label).
        """
        x_batch = self.dataset[index]
        label = self.labels[index]
        if self.rescale_resize:
            x_batch = self.img_processing(x_batch)
        x_batch = self.transform(x_batch)
        if self.nchannels == 3:
            x_batch = x_batch.unsqueeze(0)
            x_batch = x_batch.repeat(3, 1, 1, 1)
        return x_batch, label


    def resize_data_volume_by_scale(self, data, scale):
        """
        Resize the data based on the provided scale
        :param scale: float between 0 and 1
        """
        if isinstance(scale, float):
            scale_list = [scale, scale, scale]
        else:
            scale_list = scale
        return ndimage.interpolation.zoom(data, scale_list, order=0)

    def img_processing(self, image, final_size=[64, 64, 64]):
        new_scaling = [final_size[i] / image.shape[i] for i in range(3)]
        final_image = self.resize_data_volume_by_scale(image, scale=new_scaling)
        return final_image

    def normalize_intensity(self, img_tensor):
        """
        Normalizes an image tensor based on the selected method.

        :param img_tensor: Input image tensor.
        :return: Normalized image tensor.
        """
        img_tensor = torch.tensor(img_tensor, dtype=torch.float32)
        if self.normalization == "mean":
            mask = img_tensor.ne(0.0)
            desired = img_tensor[mask]
            mean_val, std_val = desired.mean(), desired.std()
            img_tensor = (img_tensor - mean_val) / std_val
        elif self.normalization == "minmax":
            max_val, min_val = img_tensor.max(), img_tensor.min()
            img_tensor = (img_tensor - min_val) / (max_val - min_val)
        return img_tensor


    def calculate_class_weights(self):
        """
        Calculates weights for each sample based on class frequency.
        :return: List of weights for each sample.
        """
        class_counts = np.bincount(self.labels)  # Count occurrences of each class
        total_samples = len(self.labels)
        class_weights = total_samples / (len(class_counts) * class_counts)  # Inverse frequency
        sample_weights = [class_weights[label] for label in self.labels]
        return sample_weights

    def undersample_dataset(self):
        """
        Performs random undersampling to balance the dataset.
        Reduces majority class samples to match the minority class count.

        :return: A tuple (undersampled dataset, undersampled labels).
        """
        unique_classes, class_counts = np.unique(self.labels, return_counts=True)
        min_samples = np.min(class_counts)  # Find the minority class count

        undersampled_indices = []
        for cls in unique_classes:
            cls_indices = np.where(self.labels == cls)[0]  # Get indices for the class
            undersampled_indices.extend(random.sample(list(cls_indices), min_samples))  # Random selection

        undersampled_indices = np.array(undersampled_indices)

        undersampled_dataset = self.dataset[undersampled_indices]
        undersampled_labels = self.labels[undersampled_indices]

        return undersampled_dataset, undersampled_labels