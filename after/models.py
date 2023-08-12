# 맥북 환경에서는 CUDA, NVIDIA 사용 불가

import numpy as np
import glob
import time
import cv2
import torch
import torchvision

from PIL import Image

from sklearn.model_selection import train_test_split, KFold

from make_dataset import MakeDataset

class Models:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.eff = torch.hub.load(
            "NVIDIA/DeepLearningExamples:torchhub", "nvidia_efficientnet_b0", pretrained=True
        )

        self.eff_utils = torch.hub.load(
            'NVIDIA/DeepLearningExamples:torchhub', 'nvidia_convnets_processing_utils'
        )

        self.eff.eval().to(self.device)

        self.md = MakeDataset()


if __name__ == "__main__":
    md = Models()

