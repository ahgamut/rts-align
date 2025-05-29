# -*- coding: utf-8 -*-
import os
import json
import numpy as np
from warnings import warn
from numpy.typing import ArrayLike
from typing import Optional
from skimage import io as skio
from skimage import util as skutil
from skimage import transform as sktrans
from skimage import measure as skmeas
from skimage import draw as skdraw

#
from rts_align.transforms import KabschEstimate


class AlignFunction:
    _extname_ = "<none>"

    def __init__(self, *args, **params):
        pass

    def _get_mapping(self, Q_img, K_img, corr, *args, **params):
        """
        receive Q_img, K_img, and correspondence info
        return something that can go into sktrans.warp
        """
        raise NotImplementedError("abstract base class")

    def __call__(self, Q_img, K_img, corr, *args, **params):
        return self._get_mapping(Q_img, K_img, corr, *args, **params)

    def align_K_to_Q(self, Q_img, K_img, corr, *args, **params):
        map_func = params.get("map_func", None)
        if map_func is None:
            map_func = self._get_mapping(Q_img, K_img, corr, *args, **params)
        return sktrans.warp(
            K_img,
            inverse_map=map_func,
            output_shape=Q_img.shape,
            mode="constant",
            cval=0,
        )


class KabschMapping(AlignFunction):
    _extname_ = "kabsch"

    def _get_mapping(self, Q_img, K_img, corr, *args, **params):
        Q_pts = corr["Q"]
        K_pts = corr["K"]

        # skimage.transform.warp is WHACK, it requires
        # the inverse transformation wrt align_K_to_Q
        map_est = KabschEstimate(dst=K_pts, src=Q_pts)

        # map_func might be nicer as a AffineTransform
        aff_mat = np.zeros((3, 3))
        aff_mat[:2, 2] = map_est.coefs[0, :]  # shift
        aff_mat[:2, 0] = map_est.coefs[1, :]  # x
        aff_mat[:2, 1] = map_est.coefs[2, :]  # y
        aff_mat[2, 2] = 1
        map_func = sktrans.AffineTransform(matrix=aff_mat)
        return map_func


class ImageComp:
    def __init__(self, img):
        self.img = np.array(img, dtype=np.float32)
        self.points = []


class ImageDesc:
    def __init__(self, raw_img, mask, markup):
        self.raw_img = raw_img
        self.markup = markup
        self.points = np.array(markup["points"], dtype=np.float32)
        self.bounds = np.array(markup.get("bounds", []), dtype=np.float32)[:, [1, 0]]
        self.mask = mask != 0

        if self.mask.min() == 0 and self.mask.max() == 0:
            if len(self.bounds) > 0:
                b2 = np.array(self.bounds, dtype=np.int32)
                self.mask = skdraw.polygon2mask(mask.shape, b2) != 0
            else:
                warn("mask is empty? assuming entire image is ok")
                self.mask.fill(1)
        self.img = self.raw_img * self.mask

    def rescale(self, f=1.0):
        self.raw_img = sktrans.rescale(self.raw_img, f, channel_axis=2, anti_aliasing=False)
        self.mask = sktrans.rescale(self.mask, f, channel_axis=2, anti_aliasing=False)
        self.bounds *= f
        self.points *= f
        self.img = self.raw_img * self.mask

    @classmethod
    def from_file(cls, img_file, mask_file, markup_file):
        raw_img = skio.imread(img_file, as_gray=False)
        mask = skio.imread(mask_file, as_gray=False) != 0
        markup = json.load(markup_file)
        return ImageDesc(raw_img, mask, markup)

    def _check_view(self):
        import matplotlib.pyplot as plt

        fig, axs = plt.subplots(nrows=1, ncols=4, figsize=(12, 9))
        plt.axis("off")
        axs[0].set_title("raw")
        axs[0].imshow(self.raw_img, cmap="Greys_r")
        axs[1].set_title("mask")
        axs[1].imshow(self.mask, cmap="Greys_r")
        msk = skmeas.grid_points_in_poly(self.img.shape, self.bounds[:, ::-1])
        axs[2].set_title("computed mask")
        axs[2].imshow(msk, cmap="Greys_r")
        axs[3].set_title("img")
        axs[3].imshow(self.img, cmap="Greys_r")
        plt.show()


def _get_mapped_points(corr):
    qpts = np.array(corr["Q"]["points"])
    qind = np.array(corr["Q"]["indices"], dtype=np.int32)
    kpts = np.array(corr["K"]["points"])
    kind = np.array(corr["K"]["indices"], dtype=np.int32)
    return {"Q": qpts[qind, :], "K": kpts[kind, :]}


class ImagePair:
    def __init__(self, sczip, rescale=1.0):
        self._qorig = ImageDesc.from_file(
            sczip.open(sczip.getinfo("Q_image.tiff"), "r"),
            sczip.open(sczip.getinfo("Q_mask.tiff"), "r"),
            sczip.open(sczip.getinfo("Q_markup.json"), "r"),
        )
        self._korig = ImageDesc.from_file(
            sczip.open(sczip.getinfo("K_image.tiff"), "r"),
            sczip.open(sczip.getinfo("K_mask.tiff"), "r"),
            sczip.open(sczip.getinfo("K_markup.json"), "r"),
        )

        if rescale != 1.0:
            self._qorig.rescale(rescale)
            self._korig.rescale(rescale)

    @property
    def Q_raw(self):
        return self._qorig.raw_img

    @property
    def K_raw(self):
        return self._korig.raw_img

    @property
    def Q_img(self):
        return self._qorig.img

    @property
    def K_img(self):
        return self._korig.img

    @property
    def Q_pts(self):
        return self._qorig.points

    @property
    def K_pts(self):
        return self._korig.points

    @property
    def Q_bounds(self):
        return self._qorig.bounds

    @property
    def K_bounds(self):
        return self._korig.bounds

    @property
    def Q_mask(self):
        return self._qorig.mask

    @property
    def K_mask(self):
        return self._korig.mask
