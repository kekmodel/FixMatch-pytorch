# code in this file is adpated from
# https://github.com/ildoonet/pytorch-randaugment/blob/master/RandAugment/augmentations.py
# https://github.com/google-research/fixmatch/blob/master/libml/ctaugment.py
import random

import numpy as np
import PIL
import PIL.ImageOps
import PIL.ImageEnhance
import PIL.ImageDraw
from PIL import Image


def AutoContrast(img, _):
    return PIL.ImageOps.autocontrast(img)


def Brightness(img, v):
    v = 0.9 * v + 0.05
    assert 0.05 <= v <= 0.95
    return PIL.ImageEnhance.Brightness(img).enhance(v)


def Color(img, v):
    v = 0.9 * v + 0.05
    assert 0.05 <= v <= 0.95
    return PIL.ImageEnhance.Color(img).enhance(v)


def Contrast(img, v):
    v = 0.9 * v + 0.05
    assert 0.05 <= v <= 0.95
    return PIL.ImageEnhance.Contrast(img).enhance(v)


def Cutout(img, v):
    assert 0 <= v <= 1
    if v == 0:
        return img
    v = 1 + int(v * min(img.size) * 0.499)
    return CutoutAbs(img, v)


def CutoutAbs(img, v):
    w, h = img.size
    x = np.random.uniform(0, w)
    y = np.random.uniform(0, h)
    x0 = max(0, x - v // 2)
    y0 = max(0, y - v // 2)
    x1 = min(w, x + v // 2)
    y1 = min(h, y + v // 2)
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
    v = 4 + int(v * 4.999)
    assert 4 <= v <= 8
    return PIL.ImageOps.posterize(img, v)


def Rotate(img, v):
    v = int(np.round((2. * v - 1.) * 30.))
    assert -30 <= v <= 30
    return img.rotate(v)


def Sharpness(img, v):
    v = 0.9 * v + 0.05
    assert 0.05 <= v <= 0.95
    return PIL.ImageEnhance.Sharpness(img).enhance(v)


def ShearX(img, v):
    v = (2. * v - 1.) * 0.3
    assert -0.3 <= v <= 0.3
    return img.transform(img.size, PIL.Image.AFFINE, (1, v, 0, 0, 1, 0))


def ShearY(img, v):
    v = (2. * v - 1.) * 0.3
    assert -0.3 <= v <= 0.3
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, v, 1, 0))


def Solarize(img, v):
    v = int(v * 255.999)
    assert 0 <= v <= 255
    return PIL.ImageOps.solarize(img, v)


def SolarizeAdd(img, v, threshold=128):
    v = int(2. * v - 1.) * 110
    assert -110 <= v <= 110
    img_np = np.array(img).astype(np.int)
    img_np = img_np + v
    img_np = np.clip(img_np, 0, 255)
    img_np = img_np.astype(np.uint8)
    img = Image.fromarray(img_np)
    return PIL.ImageOps.solarize(img, threshold)


def TranslateX(img, v):
    v = (2. * v - 1.) * 0.3
    assert -0.3 <= v <= 0.3
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0))


def TranslateY(img, v):
    v = (2. * v - 1.) * 0.3
    assert -0.3 <= v <= 0.3
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v))


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
        self.n = n
        self.m = m
        self.augment_list = augment_list()

    def __call__(self, img):
        ops = random.choices(self.augment_list, k=self.n)
        for op in ops:
            val = np.random.uniform(0, self.m) * 0.1
            img = op(img, val)
        img = Cutout(img, 1.0)
        return img
