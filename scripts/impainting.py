from __future__ import print_function
import torch
import numpy as np
from pathlib import Path
from .image import RGBDImage
from .data_process import ImageType, DataProcessForInference
from functools import partial

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import os
import sys
from torch.autograd import Variable
from .cspn.models import resnet50 as cspn_model
from .cspn.update_model import update_model

def cvt_device_tensor(image: np.ndarray, device: torch.device):
    return torch.Tensor(image).to(device)

class ImpaintingModule:
    def __init__(self, architecture, config_file_path, weight_file_path, image_width, image_height):
        '''
        cfg = get_default_config(architecture)
        cfg.merge_from_file(config_file_path)
        cfg.freeze()
        '''

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cspn_config = {"step": 24, "norm_type": "8sum"}
        self._model = self._setting_model(cspn_config, device, weight_file_path)
        #self._model = self._setting_model(cfg, device, weight_file_path)

        self._nd2t = partial(cvt_device_tensor, device = device)
        torch.set_deterministic(True)

        self._data_processor = DataProcessForInference(image_width = image_width, image_height = image_height)

    def set_depth_mean_and_std(self, depth_mean, depth_std, depth_norm=255):
        self._data_processor.set_depth_mean_and_std(depth_mean, depth_std, depth_norm)

    def _setting_model(self, cfg, device, weight_file_path):
        assert Path(weight_file_path).exists()
        #_model = MetaModel(cfg, device)
        _model = cspn_model(cfg)
        _model.to(device)
        if device.type == 'cuda':
            pretrained_state_dict = torch.load(weight_file_path)
        else:  #assume device.type == 'cpu'
            pretrained_state_dict = torch.load(weight_file_path, map_location=torch.device('cpu'))
        _model.eval()
        _model.load_state_dict(update_model(_model, pretrained_state_dict))

        #model_state_dict = _model.state_dict()
        #assert self._valification_weights(pretrained_state_dict, model_state_dict)
        #_model.load_state_dict(pretrained_state_dict['model'])
        return _model

    def _valification_weights(self, pretrained_state_dict, model_state_dict):
        wrong_key_count = 0
        wrong_shape_count = 0
        for k, v in pretrained_state_dict['model'].items():
            key_existance = "@" if k in model_state_dict else "X"
            print(f"key_existance model: {k} -> {key_existance}")
            if key_existance == "X":
                wrong_key_count += 1

            key_shape_check = "@" if model_state_dict[
                k].shape == pretrained_state_dict['model'][k].shape else "X"
            print(f"key shape check: {k} -> {key_shape_check}")
            if key_shape_check == "X":
                wrong_shape_count += 1

        print(wrong_key_count)
        print(wrong_shape_count)
        return (wrong_key_count == 0) and (wrong_shape_count == 0)

    def inference(self, rgbd_obj: RGBDImage):
        rgb_image = rgbd_obj.rgb
        depth_image = rgbd_obj.depth

        rgb_image_tfm = self._data_processor.transform_image(rgb_image, ImageType.RGB)
        depth_image_tfm = self._data_processor.transform_image(depth_image, ImageType.DEPTH)
        mask_image = np.zeros_like(depth_image_tfm)
        _, _, mask_idx_y, mask_idx_x = np.where(depth_image_tfm != 0)
        mask_image[:, :, mask_idx_y, mask_idx_x] = 1.0

        batch = {"color": self._nd2t(rgb_image_tfm), "raw_depth": self._nd2t(depth_image_tfm), "mask": self._nd2t(mask_image)}
        pred = self._model(batch)
        prediceted_depth = self._data_processor.rev_transform(pred)
        return prediceted_depth


    @property
    def model(self):
        return self._model

