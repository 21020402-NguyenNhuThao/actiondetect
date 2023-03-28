from typing import Any, Dict, Optional, Tuple

import torch
from lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
import recog_utils.video_transforms as video_transforms 
import recog_utils.volume_transforms as volume_transforms
from torchvision.transforms import transforms
import os
import pandas as pd
from torchvision.io import read_video

# class RegDataset(Dataset):
#     def __init__(self, annotations_file, data_dir, transform=None, target_transform=None):
#         self.img_labels = pd.read_csv(annotations_file)
#         self.img_dir = data_dir
#         self.transform = transform
#         self.target_transform = target_transform

#     def __len__(self):
#         return len(self.img_labels)

#     def __getitem__(self, idx):
#         img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
#         image = read_video(img_path)
#         label = self.img_labels.iloc[idx, 1]
#         if self.transform:
#             image = self.transform(image)
#         if self.target_transform:
#             label = self.target_transform(label)
#         return image, label

class RegDataset(Dataset):
    def __init__(
            self,
            anno_path,
            data_dir,
            mode='train',
            new_height=256, 
            new_width=340, 
            keep_aspect_ratio=True,
            clip_len=7):
        self.anno_path = anno_path
        self.data_dir = data_dir
        self.mode = mode
        self.clip_len = clip_len
        self.new_height = new_height
        self.new_width = new_width
        self.keep_aspect_ratio = keep_aspect_ratio
        self.aug = False
        samples = pd.read_csv(self.anno_path, header=None, delimiter=' ')
        self.dataset_samples = list(samples.values[:, 1])
        self.label_array = list(samples.values[:, 0])

        if (mode == 'train'):
            pass

        elif (mode == 'validation'):
            self.data_transform = video_transforms.Compose([
                video_transforms.Resize(self.short_side_size, interpolation='bilinear'),
                video_transforms.CenterCrop(size=(self.crop_size, self.crop_size)),
                volume_transforms.ClipToTensor(),
                video_transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                           std=[0.229, 0.224, 0.225])
            ])
        elif mode == 'test':
            self.data_resize = video_transforms.Compose([
                video_transforms.Resize(size=(short_side_size), interpolation='bilinear')
            ])
            self.data_transform = video_transforms.Compose([
                volume_transforms.ClipToTensor(),
                video_transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                           std=[0.229, 0.224, 0.225])
            ])
            self.test_seg = []
            self.test_dataset = []
            self.test_label_array = []
            for ck in range(self.test_num_segment):
                for cp in range(self.test_num_crop):
                    for idx in range(len(self.label_array)):
                        sample_label = self.label_array[idx]
                        self.test_label_array.append(sample_label)
                        self.test_dataset.append(self.dataset_samples[idx])
                        self.test_seg.append((ck, cp))
    
    def __len__(self):
        if self.mode != 'test':
            return len(self.dataset_samples)
        else:
            return len(self.test_dataset)
        
    def __getitem__(self, index):
        if self.mode == 'train':
            print('wait...')
        elif self.mode == 'validation':
            print('wait me...')
        elif self.mode == 'test':
            sample = self.test_dataset[index]
            temp_split, spac_split = self.test_seg[index]
            buffer = self.loadvideo_decord(sample) #np arrays num_of_frames x H x W x num_of_channels

            while len(buffer) == 0:
                #handle error
                print('loading video failed')
            buffer = self.data_resize(buffer)
            #compute buffer

        return buffer, self.test_label_array[index], sample.split("/")[-1].split(".")[0], temp_split, spac_split #return name of video


class RegDataModule(LightningDataModule):
    """Example of LightningDataModule for MNIST dataset.

    A DataModule implements 6 key methods:
        def prepare_data(self):
            # things to do on 1 GPU/TPU (not on every GPU/TPU in DDP)
            # download data, pre-process, split, save to disk, etc...
        def setup(self, stage):
            # things to do on every process in DDP
            # load data, set variables, etc...
        def train_dataloader(self):
            # return train dataloader
        def val_dataloader(self):
            # return validation dataloader
        def test_dataloader(self):
            # return test dataloader
        def teardown(self):
            # called on every process in DDP
            # clean up after fit or test

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://lightning.ai/docs/pytorch/latest/data/datamodule.html
    """

    def __init__(
        self,
        data_dir: str = "data/",
        train_val_test_split: Tuple[int, int, int] = (55_000, 5_000, 10_000),
        data_train : RegDataset = None,
        data_test : RegDataset = None,
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        # data transformations
        self.transforms = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    @property
    def num_classes(self):
        pass

    def prepare_data(self):
        """Download data if needed.

        Do not use it to assign state (self.x = y).
        """
        pass

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning with both `trainer.fit()` and `trainer.test()`, so be
        careful not to execute things like random split twice!
        """
        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            # trainset = MNIST(self.hparams.data_dir, train=True, transform=self.transforms)
            # testset = MNIST(self.hparams.data_dir, train=False, transform=self.transforms)
            # dataset = ConcatDataset(datasets=[trainset, testset])
            data_train = self.hparams.data_train(
                data_dir=self.hparams.data_dir)
            self.data_test = self.hparams.data_test(
                data_dir=self.hparams.data_dir)
            self.data_train, self.data_val = random_split(
                dataset=data_train,
                lengths=self.hparams.train_val_test_split,
                generator=torch.Generator().manual_seed(42),
            )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
        pass

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Things to do when loading checkpoint."""
        pass



if __name__ == "__main__":
    import pyrootutils
    from omegaconf import DictConfig
    import hydra
    import numpy as np
    from PIL import Image, ImageDraw
    from tqdm import tqdm

    path = pyrootutils.find_root(
        search_from=__file__, indicator=".project-root")
    config_path = str(path / "configs" / "data")
    output_path = path / "outputs"
    print("root", path, config_path)
    # pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


    def test_datamodule(cfg: DictConfig):
        datamodule: LightningDataModule = hydra.utils.instantiate(cfg)
        datamodule.prepare_data()
        datamodule.setup()
        loader = datamodule.train_dataloader()
        bx, by = next(iter(loader))
        print("n_batch", len(loader), bx.shape, by.shape, type(by))
        
        
        for bx, by in tqdm(datamodule.train_dataloader()):
            pass
        print("training data passed")

        for bx, by in tqdm(datamodule.val_dataloader()):
            pass
        print("validation data passed")

        for bx, by in tqdm(datamodule.test_dataloader()):
            pass
        print("test data passed")

    @hydra.main(version_base="1.3", config_path=config_path, config_name="recog.yaml")
    def main(cfg: DictConfig):
        print(cfg)
        test_datamodule(cfg)

    main()
