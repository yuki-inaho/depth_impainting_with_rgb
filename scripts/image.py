import cv2
import numpy as np
from pathlib import Path, PosixPath


def colorize_uint_image(img, max_var=255):
    img_colorized = np.zeros([img.shape[0], img.shape[1], 3]).astype(np.uint8)
    img_colorized[:, :, 1] = 255
    img_colorized[:, :, 2] = 255

    img_hue = img.copy().astype(np.float32)
    img_hue[np.where(img_hue > max_var)] = 0
    zero_idx = np.where((img_hue > max_var) | (img_hue == 0))
    img_hue *= 255.0 / max_var
    img_colorized[:, :, 0] = img_hue.astype(np.uint8)
    img_colorized = cv2.cvtColor(img_colorized, cv2.COLOR_HSV2RGB)
    img_colorized[zero_idx[0], zero_idx[1], :] = 0
    return img_colorized


class RGBDImage:
    def __init__(self, rgb_image_path:PosixPath, depth_image_path:PosixPath, max_depth=640):
        self._rgb_image_path = rgb_image_path
        self._depth_image_path = depth_image_path
        self._rgb_image = cv2.imread(str(rgb_image_path))
        self._image_width, self._image_height, _ = self._rgb_image.shape

        self._depth_image = cv2.imread(str(depth_image_path), cv2.IMREAD_ANYDEPTH)
        self._depth_image_colorized = colorize_uint_image(self._depth_image)

    @property
    def rgb(self):
        return self._rgb_image

    @property
    def depth(self):
        return self._depth_image

    @property
    def depth_colorized(self):
        return self._depth_image_colorized

    @property
    def rgb_name(self):
        suf = Path(self._rgb_image_path).suffix
        return Path(self._rgb_image_path).stem.replace(suf, ".jpg")

    @property
    def depth_name(self):
        return self.rgb_name.replace(".jpg", ".png")

    @property
    def depth_colorized_name(self):
        return self.rgb_name

    @property
    def image_size(self):
        return (self._image_width, self._image_height)

