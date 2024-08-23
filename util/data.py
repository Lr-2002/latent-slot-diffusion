import os
import glob
import torch
import random
import numpy

from PIL import Image, ImageFile
from torchvision import transforms
from torch.utils.data import Dataset
import re
import torchvision.transforms.functional as TF
from collections import deque
ImageFile.LOAD_TRUNCATED_IMAGES = True

def extract_number(item):
    match = re.findall(r'(\d+)', item)
    if match:
        return int(match[-1])
    else:
        return float('inf')  # Return infinity if no number found


class GlobVideoDataset(Dataset):
    def __init__(self, root, phase, img_size, ep_len=3, img_glob='*.png', random_crop=False, crop_ratio=0.8,step_size=1, is_random_order=False):
        self.root = root
        self.img_size = img_size
        self.random_crop = random_crop
        self.crop_size = int(img_size * crop_ratio)
        self.total_dirs = sorted(glob.glob(root, recursive=True))
        self.ep_len = ep_len
        # print('total_dirs', self.total_dirs)
        if phase == 'train':
            self.total_dirs = self.total_dirs[:int(len(self.total_dirs) * 0.9)]
        elif phase == 'val':
            self.total_dirs = self.total_dirs[int(len(self.total_dirs) * 0.9):]
        elif phase == 'test':
            self.total_dirs = self.total_dirs[int(len(self.total_dirs) * 0.95):]
        else:
            pass
        # self.total_dirs = sorted(self.total_dirs, key=extract_number)
        # chunk into episodes
        self.episodes = []
        for dir in self.total_dirs:
            # print('dir!! ', dir)
            frame_buffer = deque(maxlen=self.ep_len)
            image_paths = glob.glob(os.path.join(dir, img_glob))
            if is_random_order:
                random.shuffle(image_paths)
            else:
                image_paths = sorted(image_paths, key=extract_number)
            
            # print('img path ', image_paths, glob.glob(os.path.join(dir, img_glob)))
            for start in range(0,step_size):
                for i in range(start,len(image_paths),step_size):
                    frame_buffer.append(image_paths[i])
                    if len(frame_buffer) == self.ep_len:
                        self.episodes.append(list(frame_buffer))
                        # frame_buffer = []

        self.transform = transforms.Compose(
            [
                # * ( [transforms.RandomCrop(self.crop_size)] if random_crop else [] ) ,
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ToTensor()
            ]
        )

    def __len__(self):
        return len(self.episodes)

    def __getitem__(self, idx):
        video = []
        t,l,h,w = None, None, None, None
        for img_loc in self.episodes[idx]:
            image = Image.open(img_loc).convert("RGB")
            image = image.resize((self.img_size, self.img_size))
            if self.random_crop:
                if t == None:
                    # print(transforms.RandomCrop.get_params(image, output_size=(self.crop_size, self.crop_size)))
                    t,l,h,w = transforms.RandomCrop.get_params(image, output_size=(self.crop_size, self.crop_size))
                image = TF.crop(image, t, l, h, w)
                # print('use crop, image size: ', image.size)
            # print('image_size is {}'.format(image.size))
            tensor_image = self.transform(image)
            video += [tensor_image]
        video = torch.stack(video, dim=0)
        return video
class GlobVideoDataset_Mask(Dataset):
    def __init__(self, root, phase, img_size, ep_len=3, img_glob='*.png', random_crop=False, crop_ratio=0.8,step_size=1, is_random_order=False):
        self.root = root
        self.img_size = img_size
        self.random_crop = random_crop
        self.crop_size = int(img_size * crop_ratio)
        self.total_dirs = sorted(glob.glob(root, recursive=True))
        self.ep_len = ep_len
        # print('total_dirs', self.total_dirs)
        if phase == 'train':
            self.total_dirs = self.total_dirs[:int(len(self.total_dirs) * 0.9)]
        elif phase == 'val':
            self.total_dirs = self.total_dirs[int(len(self.total_dirs) * 0.9):]
        elif phase == 'test':
            self.total_dirs = self.total_dirs[int(len(self.total_dirs) * 0.95):]
        else:
            pass
        # self.total_dirs = sorted(self.total_dirs, key=extract_number)
        # chunk into episodes
        self.episodes = []
        for dir in self.total_dirs:
            # print('dir!! ', dir)
            frame_buffer = deque(maxlen=self.ep_len)
            image_paths = glob.glob(os.path.join(dir, img_glob))
            if is_random_order:
                random.shuffle(image_paths)
            else:
                image_paths = sorted(image_paths, key=extract_number)
            
            # print('img path ', image_paths, glob.glob(os.path.join(dir, img_glob)))
            for start in range(0,step_size):
                for i in range(start,len(image_paths),step_size):
                    frame_buffer.append(image_paths[i])
                    if len(frame_buffer) == self.ep_len:
                        self.episodes.append(list(frame_buffer))
                        # frame_buffer = []
        # img_loc = self.episodes[0]
        # mask_loc = img_loc.replace('features', 'masks').replace('png', 'npy')
        # print(img_loc, mask_loc)

        self.transform = transforms.Compose(
            [
                # * ( [transforms.RandomCrop(self.crop_size)] if random_crop else [] ) ,
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ToTensor()
            ]
        )

    def __len__(self):
        return len(self.episodes)

    def __getitem__(self, idx):
        video = []
        video_mask = []
        t,l,h,w = None, None, None, None
        for img_loc in self.episodes[idx]:
            image = Image.open(img_loc).convert("RGB")
            mask_loc = img_loc.replace('features', 'masks').replace('png', 'npy')
            

            image = image.resize((self.img_size, self.img_size))
            mask = Image.fromarray(numpy.load(mask_loc))
            if self.random_crop:
                if t == None:
                    # print(transforms.RandomCrop.get_params(image, output_size=(self.crop_size, self.crop_size)))
                    t,l,h,w = transforms.RandomCrop.get_params(image, output_size=(self.crop_size, self.crop_size))
                image = TF.crop(image, t, l, h, w)
                mask = TF.crop(mask, t, l, h, w)
                # print('use crop, image size: ', image.size)
            # print('image_size is {}'.format(image.size))
            tensor_image = self.transform(image)
            # mask = transforms.Resize((self.img_size, self.img_size))(mask)
            # mask = torch.Tensor(mask)
            mask = self.transform(mask)
            video_mask += [mask]
            video += [tensor_image]
        video = torch.stack(video, dim=0)
        video_mask = torch.stack(video_mask, dim=0)
        return video, video_mask
 
class GlobImageDataset(Dataset):
    def __init__(self, root, phase, img_size):
        self.root = root
        self.img_size = img_size
        self.total_imgs = sorted(glob.glob(root))

        if phase == 'train':
            self.total_imgs = self.total_imgs[:int(len(self.total_imgs) * 0.7)]
        elif phase == 'val':
            self.total_imgs = self.total_imgs[int(len(self.total_imgs) * 0.7):int(len(self.total_imgs) * 0.85)]
        elif phase == 'test':
            self.total_imgs = self.total_imgs[int(len(self.total_imgs) * 0.85):]
        else:
            pass

        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.total_imgs)

    def __getitem__(self, idx):
        img_loc = self.total_imgs[idx]
        image = Image.open(img_loc).convert("RGB")
        image = image.resize((self.img_size, self.img_size))
        tensor_image = self.transform(image)
        return tensor_image

class GlobImageDataset_Mask_Movi(Dataset):
    def __init__(self, root, phase, img_size):
        self.root = root
        self.img_size = img_size
        self.total_imgs = sorted(glob.glob(root))

        if phase == 'train':
            self.total_imgs = self.total_imgs[:int(len(self.total_imgs) * 0.7)]
        elif phase == 'val':
            self.total_imgs = self.total_imgs[int(len(self.total_imgs) * 0.7):int(len(self.total_imgs) * 0.85)]
        elif phase == 'test':
            self.total_imgs = self.total_imgs[int(len(self.total_imgs) * 0.85):]
        else:
            pass

        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.total_imgs)

    def __getitem__(self, idx):
        img_loc = self.total_imgs[idx]
        mask_loc = img_loc.replace('features', 'masks').replace('.png', '.npy')
        image = Image.open(img_loc).convert("RGB")
        image = image.resize((self.img_size, self.img_size))
        tensor_image = self.transform(image)
        mask = torch.Tensor(numpy.load(mask_loc))
        return tensor_image, mask.unsqueeze(0)



class GlobVideoDataset_Mask_Movi(Dataset):
    def __init__(self, root, phase, img_size, ep_len=3, img_glob='*.npy', random_crop=False, crop_ratio=0.8,
                 step_size=1, is_random_order=False, mode='image', target_shape=None):
        self.root = root
        self.img_size = img_size
        self.random_crop = random_crop
        self.target_shape = target_shape if target_shape is not None else self.img_size
        self.crop_size = int(img_size * crop_ratio)
        self.total_paths = sorted(glob.glob(root, recursive=True))
        self.ep_len = ep_len
        self.is_random_order = is_random_order
        self.mode = mode
        if is_random_order:
            raise NotImplementedError
        if phase == 'train':
            self.total_paths = self.total_paths[:int(len(self.total_paths) * 0.9)]
        elif phase == 'val':
            self.total_paths = self.total_paths[int(len(self.total_paths) * 0.9):int(len(self.total_paths) * 0.95)]
        elif phase == 'test':
            self.total_paths = self.total_paths[int(len(self.total_paths) * 0.95):]
        else:
            pass
        self.total_frame_num = numpy.load(self.total_paths[0]).shape[0]
        self.episodes = []
        for file in self.total_paths:
            frame_buffer = deque(maxlen=self.ep_len)
            # image_paths = file

            for start in range(0, step_size):
                for i in range(start, self.total_frame_num, step_size):
                    frame_buffer.append(i)
                    if len(frame_buffer) == self.ep_len:
                        self.episodes.append((file, list(frame_buffer)))

        self.transform = transforms.Compose(
            [
                transforms.Resize((self.target_shape, self.target_shape)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])
            ]
        )
        self.only_resize = transforms.Compose(
            [
                transforms.Resize((self.target_shape, self.target_shape)),
            ]
        )

    def __len__(self):
        return len(self.episodes)

    def __getitem__(self, idx):
        video = []
        video_mask = []
        t, l, h, w = None, None, None, None
        (img_loc, frame_idxs) = self.episodes[idx]
        data = numpy.load(img_loc)
        images = data  # T,H,W,
        mask_loc = img_loc.replace('image', 'mask')
        masks = numpy.load(mask_loc)  # T,H,W,1

        for frame_idx in frame_idxs:

            image = Image.fromarray(images[frame_idx])
            # mask_loc = img_loc.replace('features', 'masks').replace('png', 'npy')

            image = image.resize((self.img_size, self.img_size))
            mask = Image.fromarray(masks[frame_idx].reshape(self.img_size, self.img_size)).resize(
                (self.img_size, self.img_size))
            if self.random_crop:
                if t == None:
                    # print(transforms.RandomCrop.get_params(image, output_size=(self.crop_size, self.crop_size)))
                    t, l, h, w = transforms.RandomCrop.get_params(image, output_size=(self.crop_size, self.crop_size))
                image = TF.crop(image, t, l, h, w)
                mask = TF.crop(mask, t, l, h, w)
                # print('use crop, image size: ', image.size)
            # print('image_size is {}'.format(image.size))
            tensor_image = self.transform(image)
            # mask = transforms.Resize((self.img_size, self.img_size))(mask)
            # mask = torch.Tensor(mask)
            # mask = self.transform(mask)
            mask = transforms.functional.pil_to_tensor(self.only_resize(mask))
            video_mask += [mask]
            video += [tensor_image]

        video = torch.stack(video, dim=0)
        video_mask = torch.stack(video_mask, dim=0)
        if self.ep_len == 1 and self.mode == 'image':  # image mode, reshape from B,1,C,H,W to B,H,W,C
            video = video.flatten(0, 1)
            video_mask = video_mask.flatten(0, 1)
        return {'pixel_values': video, 'mask': video_mask}

class GlobImageDataset_Mask(Dataset):
    def __init__(self, root, phase, img_size):
        self.root = root
        self.img_size = img_size
        self.total_imgs = sorted(glob.glob(root))

        if phase == 'train':
            self.total_imgs = self.total_imgs[:int(len(self.total_imgs) * 0.7)]
        elif phase == 'val':
            self.total_imgs = self.total_imgs[int(len(self.total_imgs) * 0.7):int(len(self.total_imgs) * 0.85)]
        elif phase == 'test':
            self.total_imgs = self.total_imgs[int(len(self.total_imgs) * 0.85):]
        else:
            pass

        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.total_imgs)

    def __getitem__(self, idx):
        img_loc = self.total_imgs[idx]
        mask_loc = img_loc.replace('features', 'masks').replace('.png', '.npy')
        image = Image.open(img_loc).convert("RGB")
        image = image.resize((self.img_size, self.img_size))
        tensor_image = self.transform(image)
        mask = torch.Tensor(numpy.load(mask_loc))
        return tensor_image, mask.unsqueeze(0)


if __name__=='__main__':
    dataset = GlobVideoDataset_Mask_Movi('/home/lr-2002/movi_c/train/image/**.npy', 'train', img_size=128 , target_shape=256)
    for a in dataset:
        print(a[0].shape, a[1].shape)