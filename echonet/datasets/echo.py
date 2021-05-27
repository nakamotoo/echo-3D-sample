import os
import collections

import numpy as np
import torch.utils.data
import echonet
import sys


class Echo(torch.utils.data.Dataset):
    def __init__(self,
                 root="./data",
                 filelist_name = "FileList.csv",
                 split="train",
                 mean=0., std=1.,
                 length=16, period=2,
                 max_length=250,
                 clips=1,
                 pad=None,
                 noise=None,
                 target_transform=None,
                 external_test_location=None):

        if root is None:
            root = os.getcwd()


        self.folder = os.path.join(root, "videos")
        self.split = split
        self.mean = mean
        self.std = std
        self.length = length
        self.max_length = max_length
        self.period = period
        self.clips = clips
        self.pad = pad
        self.noise = noise
        self.target_transform = target_transform
        self.external_test_location = external_test_location

        self.normal_fnames = []
        self.sarco_fnames = []

        self.fnames = []
        self.targets = []

        with open(os.path.join(root, filelist_name)) as f:
            self.header = f.readline().strip().split(",")
            filenameIndex = self.header.index("filename")
            splitIndex = self.header.index("split")
            targetIndex = self.header.index("target")

            for line in f:
                lineSplit = line.strip().split(',')
                fileName = lineSplit[filenameIndex]
                fileMode = lineSplit[splitIndex].lower()
                target = int(lineSplit[targetIndex])

                if split in ["all", fileMode] :
                    for dirname in os.listdir(self.folder):
                        fileName = os.path.join(self.folder, dirname, fileName)
                        self.fnames.append(fileName)
                        self.targets.append(target)



    def __getitem__(self, index):
        filename = self.fnames[index]

        # Load video into np.array
        video = echonet.utils.loadvideo(filename).astype(np.float32)

        # Add simulated noise (black out random pixels)
        # 0 represents black at this point (video has not been normalized yet)
        if self.noise is not None:
            n = video.shape[1] * video.shape[2] * video.shape[3]
            ind = np.random.choice(n, round(self.noise * n), replace=False)
            f = ind % video.shape[1]
            ind //= video.shape[1]
            i = ind % video.shape[2]
            ind //= video.shape[2]
            j = ind
            video[:, f, i, j] = 0

        # Apply normalization
        if isinstance(self.mean, (float, int)):
            video -= self.mean
        else:
            video -= self.mean.reshape(3, 1, 1, 1)

        if isinstance(self.std, (float, int)):
            video /= self.std
        else:
            video /= self.std.reshape(3, 1, 1, 1)


        # Set number of frames
        c, f, h, w = video.shape
        if self.length is None:
            # Take as many frames as possible
            length = f // self.period
        else:
            # Take specified number of frames
            length = self.length


        if self.max_length is not None:
            # Shorten videos to max_length
            length = min(length, self.max_length)

        if f < length * self.period:
            # Pad video with frames filled with zeros if too short
            # 0 represents the mean color (dark grey), since this is after normalization
            video = np.concatenate((video, np.zeros((c, length * self.period - f, h, w), video.dtype)), axis=1)
            c, f, h, w = video.shape  # pylint: disable=E0633

        if self.clips == "all":
            # Take all possible clips of desired length
            start = np.arange(f - (length - 1) * self.period)
        else:
            # Take random clips from video
            start = np.random.choice(f - (length - 1) * self.period, self.clips)
            if self.split == "test":
                start = [0]


        # Select random clips
        video = tuple(video[:, s + self.period * np.arange(length), :, :] for s in start)

        if self.clips == 1:
            video = video[0]
        else:
            video = np.stack(video)


        if self.pad is not None:
            # Add padding of zeros (mean color of videos)
            # Crop of original size is taken out
            # (Used as augmentation)
            f, c, l, h, w = video.shape
            temp = np.zeros((f, c, l, h + 2 * self.pad, w + 2 * self.pad), dtype=video.dtype)
            temp[:, :, :, self.pad:-self.pad, self.pad:-self.pad] = video  # pylint: disable=E1130
            i, j = np.random.randint(0, 2 * self.pad, 2)
            video = temp[:, :, :, i:(i + h), j:(j + w)]


        target = self.targets[index]

        return video, target

    def __len__(self):
        return len(self.fnames)