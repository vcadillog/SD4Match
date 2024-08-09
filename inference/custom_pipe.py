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
from PIL import Image

np.random.seed(0)
torch.manual_seed(0)


CIHP_keys = [1, 5, 6, 7, 8, 9, 11, 12]
ATR_keys = [1, 4, 5, 6, 7, 8, 17]


def filter_parse(parse_array, mode='CIHP'):
    if mode == 'CIHP':
        filter_array = (parse_array == 5).astype(np.float32) + \
            (parse_array == 6).astype(np.float32) + \
            (parse_array == 7).astype(np.float32) + \
            (parse_array == 8).astype(np.float32) + \
            (parse_array == 9).astype(np.float32) + \
            (parse_array == 11).astype(np.float32) + \
            (parse_array == 1).astype(np.float32) + \
            (parse_array == 12).astype(np.float32)
    elif mode == 'ATR':
        filter_array = (parse_array == 4).astype(np.float32) + \
            (parse_array == 5).astype(np.float32) + \
            (parse_array == 6).astype(np.float32) + \
            (parse_array == 7).astype(np.float32) + \
            (parse_array == 1).astype(np.float32) + \
            (parse_array == 8).astype(np.float32) + \
            (parse_array == 17).astype(np.float32)
    return filter_array


def get_json_points(point_list, parse_array, mode):
    if mode == 'CIHP':
        mode_key = CIHP_keys
    elif mode == 'ATR':
        mode_key = ATR_keys
    res = {}
    coord_values = []
    index_keys = []
    for j in range(len(point_list)):
        coord_values.append(point_list[j][::-1])
        index_keys.append(parse_array[tuple(point_list[j][::-1])])

    for p in set(index_keys):
        if p in mode_key:
            res[str(p)] = []
    for keys, vals in zip(index_keys, coord_values):
        if keys in mode_key:
            res[str(keys)].append(vals.tolist())
    return res


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
        dino_cfg.merge_from_file("config/dift.py")
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

    def process_features(self, src_feat_path, src_mask_path, trg_feat_path, trg_parse_path, mode_parse):

        src_img = Image.open(src_feat_path)
        src_mask = np.load(src_mask_path)['arr_0']

        trg_img = Image.open(trg_feat_path)
        trg_parse = Image.open(trg_parse_path)

        W1, H1 = src_img.size
        W2, H2 = trg_img.size

        point_dict = {}

        for idx in range(len(src_mask)):
            src_img = np.array(
                src_img)*np.stack([src_mask[idx], src_mask[idx], src_mask[idx]], axis=-1)
            src_img = Image.fromarray(src_img)

            parse_array = np.array(trg_parse)

            filter_array = filter_parse(parse_array, mode=mode_parse)
            trg_img = Image.fromarray(np.uint8(
                np.stack([filter_array, filter_array, filter_array], axis=-1)*trg_img))

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

                src_tens = self.transforms(src_img).cuda()
                src_tens = src_tens.unsqueeze(0)

                trg_tens = self.transforms(trg_img).cuda()
                trg_tens = trg_tens.unsqueeze(0)

                src_featmaps = self.feature_extractor(
                    image=src_tens, prompt=prompt).float()
                trg_featmaps = self.feature_extractor(
                    image=trg_tens, prompt=prompt).float()

            src_size = (H1, W1)
            trg_size = (H2, W2)
            points = self.get_pair_points(
                src_mask[idx], src_featmaps, trg_featmaps, src_size, trg_size)
            point_dict[str(idx)] = get_json_points(
                points[0], parse_array, mode_parse)
        return point_dict

    def get_pair_points(self, src_mask, src_featmaps, trg_featmaps, src_size, trg_size):
        H1, W1 = src_size
        H2, W2 = trg_size
        h1, w1 = src_featmaps.shape[2:]
        h2, w2 = trg_featmaps.shape[2:]
        x, y = np.where(src_mask > 0)
        idx = np.random.choice(len(x), len(y)//(256))
        points = np.concatenate([[y[idx]], [x[idx]]], axis=0).transpose()
        src_kps = torch.tensor([points]).to(self.device).float()
        src_kps = scaling_coordinates(src_kps, (H1, W1), (h1, w1))
        nn_matches = nn_get_matches(
            src_featmaps, trg_featmaps, src_kps, l2_norm=self.enable_l2_norm)
        nn_matches = scaling_coordinates(nn_matches, (h2, w2), (H2, W2))

        points = torch.unique(nn_matches.int(), dim=1).cpu().numpy()
        return points
