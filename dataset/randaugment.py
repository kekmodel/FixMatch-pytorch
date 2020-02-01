# code in this file is adpated from
# https://github.com/ildoonet/pytorch-randaugment/blob/master/RandAugment/augmentations.py
# https://github.com/google-research/fixmatch/blob/master/third_party/auto_augment/augmentations.py
# https://github.com/google-research/fixmatch/blob/master/libml/ctaugment.py
import logging
import random

import numpy as np
import PIL
import PIL.ImageOps
import PIL.ImageEnhance
import PIL.ImageDraw
from PIL import Image

logger = logging.getLogger(__name__)

PARAMETER_MAX = 10


def AutoContrast(img, _):
    return PIL.ImageOps.autocontrast(img)


def Brightness(img, v):
    v = _float_parameter(v, 0.9) + 0.05
    assert 0.05 <= v <= 0.95
    return PIL.ImageEnhance.Brightness(img).enhance(v)


def Color(img, v):
    v = _float_parameter(v, 0.9) + 0.05
    assert 0.05 <= v <= 0.95
    return PIL.ImageEnhance.Color(img).enhance(v)


def Contrast(img, v):
    v = _float_parameter(v, 0.9) + 0.05
    assert 0.05 <= v <= 0.95
    return PIL.ImageEnhance.Contrast(img).enhance(v)


def Cutout(img, v):
    if v == 0:
        return img
    v = _int_parameter(v, min(img.size) * 0.5)
    return CutoutAbs(img, v)


def CutoutAbs(img, v):
    w, h = img.size
    x0 = np.random.uniform(0, w)
    y0 = np.random.uniform(0, h)
    x0 = int(max(0, x0 - v / 2.))
    y0 = int(max(0, y0 - v / 2.))
    x1 = min(w, x0 + v)
    y1 = min(h, y0 + v)
    xy = (x0, y0, x1, y1)
    # gray
    color = (127, 127, 127)
    img = img.copy()
    PIL.ImageDraw.Draw(img).rectangle(xy, color)
    return img


def Equalize(img, _):
    return PIL.ImageOps.equalize(img)


def Identity(img, v):
    return img


def Invert(img, _):
    return PIL.ImageOps.invert(img)


def Posterize(img, v):
    v = _int_parameter(v, 4) + 4
    assert 4 <= v <= 8
    return PIL.ImageOps.posterize(img, v)


def Rotate(img, v):
    v = _int_parameter(v, 30)
    if random.random() < 0.5:
        v = -v
    assert -30 <= v <= 30
    return img.rotate(v)


def Sharpness(img, v):
    v = _float_parameter(v, 0.9) + 0.05
    assert 0.05 <= v <= 0.95
    return PIL.ImageEnhance.Sharpness(img).enhance(v)


def ShearX(img, v):
    v = _float_parameter(v, 0.3)
    if random.random() < 0.5:
        v = -v
    assert -0.3 <= v <= 0.3
    return img.transform(img.size, PIL.Image.AFFINE, (1, v, 0, 0, 1, 0))


def ShearY(img, v):
    v = _float_parameter(v, 0.3)
    if random.random() < 0.5:
        v = -v
    assert -0.3 <= v <= 0.3
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, v, 1, 0))


def Solarize(img, v):
    v = _int_parameter(v, 256)
    assert 0 <= v <= 256
    return PIL.ImageOps.solarize(img, 256 - v)


def SolarizeAdd(img, v, threshold=128):
    v = _int_parameter(v, 110)
    if random.random() < 0.5:
        v = -v
    assert -110 <= v <= 110
    img_np = np.array(img).astype(np.int)
    img_np = img_np + v
    img_np = np.clip(img_np, 0, 255)
    img_np = img_np.astype(np.uint8)
    img = Image.fromarray(img_np)
    return PIL.ImageOps.solarize(img, threshold)


def TranslateX(img, v):
    v = _float_parameter(v, 0.3)
    if random.random() < 0.5:
        v = -v
    assert -0.3 <= v <= 0.3
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0))


def TranslateY(img, v):
    v = _float_parameter(v, 0.3)
    if random.random() < 0.5:
        v = -v
    assert -0.3 <= v <= 0.3
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v))


def _float_parameter(v, max_v):
    return float(v) * max_v / PARAMETER_MAX


def _int_parameter(v, max_v):
    return int(v * max_v / PARAMETER_MAX)


def augment_list():
    # FixMatch paper
    augs = [AutoContrast,
            Brightness,
            Color,
            Contrast,
            Equalize,
            Identity,
            Posterize,
            Rotate,
            Solarize,
            Sharpness,
            ShearX,
            ShearY,
            TranslateX,
            TranslateY]
    return augs


class RandAugCutout(object):
    def __init__(self, n, m):
        assert n >= 1
        assert 1 <= m <= 10
        self.n = n
        self.m = m
        self.augment_list = augment_list()

    def __call__(self, img):
        ops = random.choices(self.augment_list, k=self.n)
        for op in ops:
            val = np.random.randint(1, self.m)
            if random.random() < 0.5:
                img = op(img, val)
        img = CutoutAbs(img, 16)
        return img
