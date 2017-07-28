import cv2, matplotlib
import numpy as np
import matplotlib.pyplot as plt

# 読みたい画面のパス/ファイル名を指定します
im = cv2.imread('input.jpg')

# 2値化に使う閾値
THRESH_FOR_PORTAL_IMG = 250

# Step 1(a): ここでグレースケール画像を作る(ぼかしを併用しない場合)
imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

# Step 1(b): ここでグレースケール画像を作る(ガウスぼかしを併用する場合)
# imgray = cv2.cvtColor(cv2.GaussianBlur(im, (5, 5), 0), cv2.COLOR_BGR2GRAY)

# Step 2(a): グレースケール画像を2値化する(閾値処理の場合)
ret, thresh = cv2.threshold(imgray, THRESH_FOR_PORTAL_IMG, 255, cv2.THRESH_BINARY)

# Step 2(b): グレースケール画像を2値化する(キャニー法によるエッジ検出を行う場合)
# thresh = cv2.Canny(imgray, threshold1=50, threshold2=100)

# Step 3:  2値化画像をもとに輪郭検出する。contoursの中に輪郭データが入っている
image, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Step 4: 元画像のコピーを作成し、以後の作業はコピーに対して行うようにする
im_copied = np.copy(im)

# Step 5: 輪郭データのうち、小さすぎる領域は無視する
MIN_AREA = 100
large_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > MIN_AREA]

# 外接矩形を描画
for contour in large_contours:
    x, y, w, h = cv2.boundingRect(contour)
    cv2.rectangle(im_copied, (x, y), (x + w, y + h), (0, 255, 0), 3)

# Step 6(a):
# 元の画像と輪郭データをもとに、元画像の上に輪郭が描画された画像を作成する
# (全輪郭データを出力する場合)
# result_img = cv2.drawContours(im, contours, -1, (0, 255, 0), 3)
# result_img = cv2.drawContours(im_copied, large_contours, -1, (0, 255, 0), 3)

# Step 6(b):
# 元の画像と輪郭データをもとに、元画像の上に輪郭が描画された画像を作成する
# (4番目の輪郭データのみを出力する場合(輪郭番号ベタ指定))
# result_img = cv2.drawContours(im, contours, 3, (0,255,0), 3)

# Step 6(c):
# 元の画像と輪郭データをもとに、元画像の上に輪郭が描画された画像を作成する
# (4番目の輪郭データのみを出力する場合(輪郭をいったん別配列に切り出して指定))
# cnt = contours[4]
# result_img = cv2.drawContours(im, [cnt], 0, (0,255,0), 3)

# 結果の出力
cv2.imshow('result_img', im_copied)
cv2.imwrite('result.jpg', im_copied)
cv2.waitKey(0)
