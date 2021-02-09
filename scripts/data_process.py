import cv2
import numpy as np
from enum import Enum
import torch


class ImageType(Enum):
    RGB = 1
    DEPTH = 2

def add_dummy_dim(image):
    return image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))


def convert_img_dim(image):
    return image.reshape(image.shape[1], image.shape[2], image.shape[0])


class DataProcessForInference:
    def __init__(self, resize_x: int = 640, resize_y: int = 480, image_height: int = 1080, image_width: int = 1920, max_depth: int = 255):
        self._image_height = image_height
        self._image_width = image_width

        assert resize_x > resize_y
        self._resize_x = resize_x
        self._resize_y = resize_y

        self._resize_ratio_x = self._resize_x/max(self._image_height, self._image_width)
        self._resize_ratio_y = self._resize_y/min(self._image_height, self._image_width)
        self._enable_rotation = image_height > image_width

        self._pad_value = 0
        self._rgb_norm = np.array([255, 255, 255], dtype=np.float32)
        self._rgb_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self._rgb_std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        self._depth_mean = 128.0
        self._depth_std = 128.0
        self._max_depth = max_depth

    def set_depth_mean_and_std(self, depth_mean, depth_std, depth_norm):
        self.depth_norm = depth_norm
        self.depth_mean = depth_mean
        self.depth_std = depth_std

    def transform_image(self, image, image_type: ImageType):
        if image_type ==  image_type.RGB: image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self._enable_rotation: image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

        # resize
        image_resized = cv2.resize(image, None, fx=self._resize_ratio_x, fy=self._resize_ratio_y)

        # padding (to run inference by torch2trt)
        if image_type ==  image_type.RGB:
            rh, rw, _ = image_resized.shape
        else:
            rh, rw = image_resized.shape

        pad_w = (self._resize_x - rw) // 2
        pad_h = (self._resize_y - rh) // 2

        if image_type == ImageType.RGB:
            image_padded = cv2.copyMakeBorder(image_resized, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_CONSTANT, (self._pad_value, self._pad_value, self._pad_value))
        else:
            image_padded = cv2.copyMakeBorder(image_resized, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_CONSTANT, (self._pad_value))

        # normalize
        if image_type == ImageType.RGB:
            image_padded = image_padded.astype(np.float32)
            image_padded /= self._rgb_norm

            image_normalized = image_padded.copy()
            image_normalized -= self._rgb_mean
            image_normalized /= self._rgb_std
        else:
            image_padded = image_padded.astype(np.float32)
            zero_idx_y, zero_idx_x = np.where(image_padded == 0)

            image_normalized = image_padded.copy()
            image_normalized -= self._depth_mean
            image_normalized /= self._depth_std
            image_normalized[zero_idx_y, zero_idx_x] = 0
            image_normalized = image_normalized[:, :, np.newaxis] 

        image_preprocessed = add_dummy_dim(image_normalized).transpose((0,3,1,2)) # convert to BCHW
        return image_preprocessed

    def rev_transform(self, predicted_result: torch.Tensor):
        inv_resize_ratio_x = 1 / self._resize_ratio_x
        inv_resize_ratio_y = 1 / self._resize_ratio_y

        predicted_depth = predicted_result[0][0].cpu().detach().numpy()
        depth_resized = cv2.resize(predicted_depth, None, fx=inv_resize_ratio_x, fy=inv_resize_ratio_y, interpolation=cv2.INTER_LINEAR)

        # remove padding
        rh, rw = depth_resized.shape
        if self._enable_rotation:
            tl_y = (rh - self._image_width) // 2
            tl_x = (rw - self._image_height) // 2
            depth_cropped = depth_resized[tl_y : tl_y + self._image_width, tl_x : tl_x + self._image_height]
        else:
            tl_y = (rh - self._image_height) // 2
            tl_x = (rw - self._image_width) // 2
            depth_cropped = depth_resized[tl_y : tl_y + self._image_height, tl_x : tl_x + self._image_width]

        depth_inferred = depth_cropped * self._depth_std + self.depth_mean
        
        depth_inferred[np.where(depth_inferred < 0)] = 0
        depth_inferred[np.where(depth_inferred > self._max_depth)] = self._max_depth
        depth_inferred = depth_inferred.astype(np.int32)

        return depth_inferred