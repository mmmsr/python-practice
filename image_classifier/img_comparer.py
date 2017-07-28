#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import cv2
import os
import shutil


# ここに指定したファイル名称の画像に似た画像を抽出します
STANDARD_IMG_FILE = 'some_image.jpg'

# ここで指定した名称のディレクトリの中に、比較したい画像ファイルを置いておきます
TARGET_IMG_DIR = 'TARGET_IMGS/'

# ここで指定した名称のディレクトリの中に、「類似している」とみなされた画像を出力します
RESULTS_DIR = 'RESULTS'

# ここで指定したサイズで画像を読み込みます。
IMG_SIZE = (200, 200)

# 算出される類似度がこれより小さい(類似している)場合、「似た画像」とみなされます
THRESHOLD = 150

MSG_POS = '類似度が基準値を超えています'
MSG_NEG = '類似度が基準値を超えていません'


def copy_file(file_name, destination):
    shutil.copy(file_name, destination)


def extract_images():
    standard_img = cv2.imread(STANDARD_IMG_FILE, cv2.IMREAD_GRAYSCALE)
    standard_img = cv2.resize(standard_img, IMG_SIZE)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    detector = cv2.AKAZE_create()
    (target_kp, target_des) = detector.detectAndCompute(standard_img, None)

    print('STANDARD_IMG_FILE: %s' % (STANDARD_IMG_FILE))

    files = os.listdir(TARGET_IMG_DIR)
    for file in files:
        if file == '.DS_Store' or file == STANDARD_IMG_FILE:
            continue

        comparing_img_path = TARGET_IMG_DIR + file
        try:
            comparing_img = cv2.imread(comparing_img_path, cv2.IMREAD_GRAYSCALE)
            comparing_img = cv2.resize(comparing_img, IMG_SIZE)
            (comparing_kp, comparing_des) = detector.detectAndCompute(comparing_img, None)
            matches = bf.match(target_des, comparing_des)
            dist = [m.distance for m in matches]
            ret = sum(dist) / len(dist)
        except cv2.error:
            ret = 100000

        if ret < THRESHOLD:
            print(file, ret, MSG_POS)
            copy_file(comparing_img_path,RESULTS_DIR)
        else:
            print(file, ret, MSG_NEG)


if __name__ == "__main__":
    extract_images()
