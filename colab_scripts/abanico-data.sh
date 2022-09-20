#!/bin/bash
pip install --upgrade --no-cache-dir gdown

cd /content/
echo "===> Creating folders..."
mkdir /content/DATA
mkdir /content/DATA/DataSet_ConchasAbanico

echo "===> Downloading ABANICO dataset"
id="1qJpTORrTuCETMWg3FBJeCaCrn8IVrBDh"

gdown --id $id
filename="train_val.rar"
src="/content/${filename}"
dst="/content/DATA/DataSet_ConchasAbanico"
mv $src ${dst}

echo "===> Unzipping ${filename}"
unrar x -Y "/content/DATA/DataSet_ConchasAbanico/${filename}" "/content/DATA/DataSet_ConchasAbanico"
rm "/content/DATA/DataSet_ConchasAbanico/${filename}"

#test images
id="1Hed8HHT2fp5AMZPagKexZamINm04ynis"

gdown --id $id
filename="images.zip"
src="/content/${filename}"
dst="/content/DATA/drive/MyDrive/DATA/DataSet_ConchasAbanico/data/test/"
mv $src ${dst}

echo "===> Unzipping ${filename}"
unzip -q "/content/DATA/drive/MyDrive/DATA/DataSet_ConchasAbanico/data/test/${filename}" -d dst
rm "/content/DATA/drive/MyDrive/DATA/DataSet_ConchasAbanico/data/test/${filename}"