import sys
import os
import json
import argparse
import zipfile
import numpy as np
import skimage.io as skio

NAME = "sc-extract"
DESC = "extract image and JSON"

from rts_align import ImagePair


def runner(input_zip, prefix, rescale, output_dir=None):
    zfile = zipfile.ZipFile(input_zip, "r")
    ipair = ImagePair(zfile, rescale=rescale)
    bname = os.path.basename(input_zip)
    bbase = os.path.splitext(bname)[0]
    if output_dir is not None:
        fpath1 = os.path.join(output_dir, f"{prefix}-{bbase}-K.png")
        raw = np.uint8(255 * ipair.K_raw)
        skio.imsave(fpath1, raw)
        fpath2 = os.path.join(output_dir, f"{prefix}-{bbase}-K.json")
        with open(fpath2, "w") as fp2:
            res = dict(points=ipair.K_pts.tolist())
            json.dump(res, fp2)


def main():
    parser = argparse.ArgumentParser(NAME, description=DESC)
    parser.add_argument(
        "-i",
        "--input-zip",
        required=True,
        help="ZIP file containing shoecomp comparisons",
    )
    parser.add_argument(
        "-r",
        "--rescale",
        default=1.0,
        type=float,
        help="amount to rescale the images by",
    )
    parser.add_argument(
        "-o", "--output-dir", default="./", help="folder to save aligned images"
    )
    parser.add_argument(
        "-p", "--prefix", default="sample", help="prefix for the extracted images"
    )
    d = parser.parse_args()
    runner(
        input_zip=d.input_zip,
        rescale=d.rescale,
        prefix=d.prefix,
        output_dir=d.output_dir,
    )


if __name__ == "__main__":
    main()
