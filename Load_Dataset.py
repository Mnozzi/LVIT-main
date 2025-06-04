# -*- coding: utf-8 -*-
import numpy as np
import torch
import random
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.transforms import functional as F
from typing import Callable
import os
import cv2
from scipy import ndimage
# from bert_embedding import BertEmbedding
from transformers import BertTokenizer, BertModel
import torch.nn.functional as F

def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label, text = sample['image'], sample['label'], sample['text']
        image, label = image.astype(np.uint8), label.astype(np.uint8)
        image, label = F.to_pil_image(image), F.to_pil_image(label)
        x, y = image.size
        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)

        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)  # why not 3?
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = F.to_tensor(image)
        label = to_long_tensor(label)
        text = torch.Tensor(text)
        sample = {'image': image, 'label': label, 'text': text}
        return sample


class ValGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label, text = sample['image'], sample['label'], sample['text']
        image, label = image.astype(np.uint8), label.astype(np.uint8)  # OSIC
        image, label = F.to_pil_image(image), F.to_pil_image(label)
        x, y = image.size
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)  # why not 3?
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = F.to_tensor(image)
        label = to_long_tensor(label)
        text = torch.Tensor(text)
        sample = {'image': image, 'label': label, 'text': text}
        return sample


def to_long_tensor(pic):
    # handle numpy array
    img = torch.from_numpy(np.array(pic, np.uint8))
    # backward compatibility
    return img.long()


def correct_dims(*images):
    corr_images = []
    for img in images:
        if len(img.shape) == 2:
            corr_images.append(np.expand_dims(img, axis=2))
        else:
            corr_images.append(img)

    if len(corr_images) == 1:
        return corr_images[0]
    else:
        return corr_images


class LV2D(Dataset):
    def __init__(self, dataset_path: str, task_name: str, row_text: str, joint_transform: Callable = None,
                 one_hot_mask: int = False,
                 image_size: int = 224) -> None:
        self.dataset_path = dataset_path
        self.image_size = image_size
        self.output_path = os.path.join(dataset_path)
        self.mask_list = os.listdir(self.output_path)
        self.one_hot_mask = one_hot_mask
        self.rowtext = row_text
        self.task_name = task_name
        # self.bert_embedding = BertEmbedding()
        bert_local_path = r"E:\Learning_Documents\Computer_Vision_and_Intelligent_Medical_Image_Analysis\BIG_HOMEWORK\LVIT-main\bert-base-uncased"  # 您的本地路径
        self.tokenizer = BertTokenizer.from_pretrained(
            bert_local_path,
            local_files_only=True
        )
        self.bert_model = BertModel.from_pretrained(
            bert_local_path,
            local_files_only=True,
            use_safetensors=True
        )
        self.bert_model.eval()

        if joint_transform:
            self.joint_transform = joint_transform
        else:
            to_tensor = T.ToTensor()
            self.joint_transform = lambda x, y: (to_tensor(x), to_tensor(y))

    def __len__(self):
        return len(os.listdir(self.output_path))

    def __getitem__(self, idx):

        mask_filename = self.mask_list[idx]  # Co
        mask = cv2.imread(os.path.join(self.output_path, mask_filename), 0)
        mask = cv2.resize(mask, (self.image_size, self.image_size))
        mask[mask <= 0] = 0
        mask[mask > 0] = 1
        mask = correct_dims(mask)

        # text = self.rowtext[mask_filename]
        # text = text.split('\n')
        # text_token = self.bert_embedding(text)
        # text = np.array(text_token[0][1])
        # if text.shape[0] > 14:
        #     text = text[:14, :]

        text = self.rowtext.get(mask_filename, "")
        # inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        # with torch.no_grad():
        #     outputs = self.bert_model(**inputs)

        try:
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding='max_length', max_length=14)
            with torch.no_grad():
                outputs = self.bert_model(**inputs)
            embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
        except Exception as e:
            print(f"文本处理出错: {str(e)}，使用零向量代替")
            embedding = np.random.normal(0, 0.01, (768,))  # 保持维度一致

        # embedding = outputs.last_hidden_state.squeeze(0).cpu().numpy()
        # max_len = 14
        # if embedding.shape[0] > max_len:
        #     embedding = embedding[:max_len]
        # else:
        #     embedding = np.pad(embedding, ((0, max_len - embedding.shape[0]), (0, 0)), mode='constant')
        # text = embedding.astype(np.float32)
        if len(embedding.shape) == 1:
            embedding = np.tile(embedding, (14, 1))

        text_embedding = embedding.astype(np.float32)

        if self.one_hot_mask:
            assert self.one_hot_mask > 0, 'one_hot_mask must be nonnegative'
            mask = torch.zeros((self.one_hot_mask, mask.shape[1], mask.shape[2])).scatter_(0, mask.long(), 1)

        sample = {'label': mask, 'text': text_embedding}

        return sample, mask_filename


class ImageToImage2D(Dataset):

    def __init__(self, dataset_path: str, task_name: str, row_text: str, joint_transform: Callable = None,
                 one_hot_mask: int = False,
                 image_size: int = 224) -> None:
        self.dataset_path = dataset_path
        self.image_size = image_size
        self.input_path = os.path.join(dataset_path, 'img')
        self.output_path = os.path.join(dataset_path, 'labelcol')
        self.images_list = os.listdir(self.input_path)
        self.mask_list = os.listdir(self.output_path)
        self.one_hot_mask = one_hot_mask
        self.rowtext = row_text
        self.task_name = task_name
        # self.bert_embedding = BertEmbedding()

        bert_local_path = r"E:\Learning_Documents\Computer_Vision_and_Intelligent_Medical_Image_Analysis\BIG_HOMEWORK\LVIT-main\bert-base-uncased"  # 您的本地路径
        self.tokenizer = BertTokenizer.from_pretrained(
            bert_local_path,
            local_files_only=True
        )
        self.bert_model = BertModel.from_pretrained(
            bert_local_path,
            local_files_only=True,
            use_safetensors=True
        )
        self.bert_model.eval()

        if joint_transform:
            self.joint_transform = joint_transform
        else:
            to_tensor = T.ToTensor()
            self.joint_transform = lambda x, y: (to_tensor(x), to_tensor(y))

    def __len__(self):
        return len(os.listdir(self.input_path))

    def __getitem__(self, idx):
        max_attempts = 3  # 最大尝试次数
        attempt = 0

        while attempt < max_attempts:
            try:
                # 获取文件名
                image_filename = self.images_list[idx]
                mask_filename = image_filename[:-3] + "png"

                # 读取图像
                image_path = os.path.join(self.input_path, image_filename)
                image = cv2.imread(image_path)
                if image is None:
                    raise ValueError(f"无法读取图像文件: {image_path}")

                # 读取掩码
                mask_path = os.path.join(self.output_path, mask_filename)
                mask = cv2.imread(mask_path, 0)
                if mask is None:
                    raise ValueError(f"无法读取掩码文件: {mask_path}")

                # 调整尺寸
                image = cv2.resize(image, (self.image_size, self.image_size))
                mask = cv2.resize(mask, (self.image_size, self.image_size))

                # 后续处理
                mask[mask <= 0] = 0
                mask[mask > 0] = 1
                image, mask = correct_dims(image, mask)

                # 文本处理（添加容错）
                text = self.rowtext.get(mask_filename, "").strip()  # 使用get避免KeyError
                try:
                    inputs = self.tokenizer(text,
                                            return_tensors="pt",
                                            truncation=True,
                                            padding='max_length',
                                            max_length=10)  # 与max_len一致
                    with torch.no_grad():
                        outputs = self.bert_model(**inputs)
                    # 使用[CLS]标记作为embedding
                    embedding = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()
                except Exception as e:
                    print(f"文本处理出错: {str(e)}，使用随机向量代替")
                    embedding = np.random.normal(0, 0.01, (768,))
                # if text:
                #     # inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
                #     # with torch.no_grad():
                #     #     outputs = self.bert_model(**inputs)
                #     try:
                #         inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True,
                #                                 max_length=512)
                #         with torch.no_grad():
                #             outputs = self.bert_model(**inputs)
                #         embedding = outputs.last_hidden_state.squeeze(0).cpu().numpy()
                #     except Exception as e:
                #         print(f"文本处理出错: {str(e)}，使用零向量代替")
                #         embedding = np.zeros((10, 768), dtype=np.float32)  # 保持维度一致
                #     embedding = outputs.last_hidden_state.squeeze(0).cpu().numpy()
                #     max_len = 10
                #     if embedding.shape[0] > max_len:
                #         embedding = embedding[:max_len]
                #     else:
                #         embedding = np.pad(embedding, ((0, max_len - embedding.shape[0]), (0, 0)), mode='constant')
                # else:
                #     embedding = np.zeros((10, 768), dtype=np.float32)
                # text_embedding = embedding.astype(np.float32)
                # # text = text.split('\n')
                # # text_token = self.bert_embedding(text) if text else [None]
                # # text_embedding = np.array(text_token[0][1]) if text_token and text_token[0][
                # #     1] is not None else np.zeros((10, 768))
                # # if text_embedding.shape[0] > 10:
                # #     text_embedding = text_embedding[:10, :]

                # 准备样本
                if len(embedding.shape) == 1:
                    embedding = np.tile(embedding, (10, 1))

                text_embedding = embedding.astype(np.float32)
                sample = {'image': image, 'label': mask, 'text': text_embedding}
                if self.joint_transform:
                    sample = self.joint_transform(sample)

                return sample, image_filename

            except Exception as e:
                print(f"警告: 处理 {image_filename} 失败 (尝试 {attempt + 1}/{max_attempts}): {str(e)}")
                attempt += 1
                idx = (idx + 1) % len(self.images_list)  # 尝试下一个文件
                continue

        # 如果多次尝试都失败，返回空数据（需确保模型能处理）
        print(f"错误: 无法加载有效数据，已跳过索引 {idx}")
        empty_image = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
        empty_mask = np.zeros((self.image_size, self.image_size), dtype=np.uint8)
        empty_text = np.zeros((10, 768), dtype=np.float32)

        sample = {'image': empty_image, 'label': empty_mask, 'text': empty_text}
        if self.joint_transform:
            sample = self.joint_transform(sample)

        return sample, "invalid_image"