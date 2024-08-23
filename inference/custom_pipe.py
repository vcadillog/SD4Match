import os

import torch
import torchvision.transforms as T

import numpy as np

from utils.misc import *
from utils.matching import *
from utils.geometry import *
from config import get_default_defaults

from src.stable_diffusion.sd_feature_extractor import SDFeatureExtraction
from src.stable_diffusion.prompt import PromptManager
from inference.hybrid_captioner import HybridCaptioner

from transformers import AutoModel, AutoImageProcessor

np.random.seed(0)
torch.manual_seed(0)


class SD4MatchPipe:
    def __init__(self, device):
        cfg = get_default_defaults()
        config_file = 'config/learnedToken.py'
        cfg.merge_from_file(config_file)

        dataset = 'spair'
        prompt_type = 'CPM_spair_sd2-1_Pair-DINO-Feat-G25-C50'
        timestep = 50
        layer = 1

        cfg.DATASET.NAME = dataset
        cfg.FEATURE_EXTRACTOR.PROMPT_TYPE = prompt_type
        cfg.FEATURE_EXTRACTOR.SELECT_TIMESTEP = timestep
        cfg.FEATURE_EXTRACTOR.SELECT_LAYER = layer
        cfg.FEATURE_EXTRACTOR.FUSE_DINO = True
        cfg.FEATURE_EXTRACTOR.PROMPT_CACHE_ROOT = 'pretrained_weights'

        dino_cfg = get_default_defaults()
        dino_cfg.merge_from_file("config/fuseDino.py")
        self.transforms = T.Compose([
            T.ToTensor(),
            T.Resize((dino_cfg.DATASET.IMG_SIZE, dino_cfg.DATASET.IMG_SIZE)),
            T.Normalize(mean=dino_cfg.DATASET.MEAN, std=dino_cfg.DATASET.STD)
        ])

        vision_encoder = AutoModel.from_pretrained('facebook/dinov2-base')
        vision_processor = AutoImageProcessor.from_pretrained(
            'facebook/dinov2-base')
        for param in vision_encoder.parameters():
            param.requires_grad = False

        self.vision_encoder = vision_encoder.to(device)
        self.vision_processor = vision_processor

        feature_extractor = SDFeatureExtraction(cfg)
        self.feature_extractor = feature_extractor.to(device)

        if "CPM" in prompt_type:
            ckpt = torch.load(os.path.join(
                cfg.FEATURE_EXTRACTOR.PROMPT_CACHE_ROOT, prompt_type, "ckpt.pt"))
            captioner = HybridCaptioner(prompt_type, device)
            captioner.load_state_dict(ckpt, strict=False)
            self.captioner = captioner.to(device)
        else:
            prompter = PromptManager(cfg)

        self.enable_l2_norm = True
        self.softmax_temp = 0.04
        self.gaussian_suppression_sigma = 7
        self.device = device

    def process_features(self, src_img, trg_img):

        with torch.no_grad():
            src_tens = self.vision_processor(src_img, return_tensors="pt")[
                "pixel_values"]
            src_feat = self.vision_encoder(
                pixel_values=src_tens.to(self.vision_encoder.device))

            trg_tens = self.vision_processor(trg_img, return_tensors="pt")[
                "pixel_values"]
            trg_feat = self.vision_encoder(
                pixel_values=trg_tens.to(self.vision_encoder.device))

            prompt = self.captioner(
                src_feature=src_feat, trg_feature=trg_feat)

            src_tens = self.transforms(src_img).to(self.device)
            src_tens = src_tens.unsqueeze(0)

            trg_tens = self.transforms(trg_img).to(self.device)
            trg_tens = trg_tens.unsqueeze(0)

            src_featmaps = self.feature_extractor(
                image=src_tens, prompt=prompt).float()
            trg_featmaps = self.feature_extractor(
                image=trg_tens, prompt=prompt).float()
        return src_featmaps, trg_featmaps

    def get_pair_points(self, src_mask, src_featmaps, trg_featmaps, src_size, trg_size):
        W1, H1 = src_size
        W2, H2 = trg_size
        h1, w1 = src_featmaps.shape[2:]
        h2, w2 = trg_featmaps.shape[2:]
        x, y = np.where(src_mask > 0)
        idx = np.random.choice(len(x), len(y)//256)
        input_point = np.concatenate([[y[idx]], [x[idx]]], axis=0).transpose()
        src_kps = torch.tensor([input_point]).to(self.device).float()
        src_kps = scaling_coordinates(src_kps, (H1, W1), (h1, w1))
        matches = kernel_softargmax_get_matches(
            src_featmaps, trg_featmaps, src_kps, self.softmax_temp, self.gaussian_suppression_sigma, l2_norm=self.enable_l2_norm)
        matches = scaling_coordinates(matches, (h2, w2), (H2, W2))
        matches = matches.detach().cpu().numpy()

        _, idx = np.unique(matches, axis=1, return_index=True)
        trg_points = matches[0][idx]
        src_points = input_point[idx]
        return src_points, trg_points
