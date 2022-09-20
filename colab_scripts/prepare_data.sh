#!/bin/bash
pip install --upgrade --no-cache-dir gdown

cd /content/
echo "===> Creating ckpt/saved folder..."
mkdir SUIM-dev/ckpt/saved

echo "===> Downloading unet_rgb.hdf5 model"
id="1B8KdeaoS4aITkQ6wAU1ASfCZ9o2tdfCM"
gdown --id $id
filename="unet_rgb.hdf5"
src="/content/${filename}"
dst="/content/SUIM-dev/ckpt/saved"
mv $src ${dst}

echo "===> Downloading suimnet_vgg5.hdf5 model"
id="1vsjJt8AmRJssV1yS32_dI2A8AXNGXjiL"
gdown --id $id
filename="suimnet_vgg5.hdf5"
src="/content/${filename}"
dst="/content/SUIM-dev/ckpt/saved"
mv $src ${dst}

echo "===> Downloading suimnet_rsb5.hdf5 model"
id="1wZ9hzq2ctuWFykIy10eTWmSSWiP8zGxW"
gdown --id $id
filename="suimnet_rsb5.hdf5"
src="/content/${filename}"
dst="/content/SUIM-dev/ckpt/saved"
mv $src ${dst}