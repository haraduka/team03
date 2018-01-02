#coding=utf-8

import cv2
import numpy as np
from PIL import Image

P_ID = 1

def resize_image(image, height, width):

    # 元々のサイズを取得
    org_height, org_width = image.shape[:2]

    # 大きい方のサイズに合わせて縮小
    if float(height)/org_height > float(width)/org_width:
        ratio = float(height)/org_height
    else:
        ratio = float(width)/org_width

    # リサイズ
    resized = cv2.resize(image,(int(org_height*ratio),int(org_width*ratio)))

    return resized  

def overlayOnPart(src_image, overlay_image, posX, posY):

    # オーバレイ画像のサイズを取得
    ol_height, ol_width = overlay_image.shape[:2]

    # OpenCVの画像データをPILに変換
    #　BGRAからRGBAへ変換
    src_image_RGBA = cv2.cvtColor(src_image, cv2.COLOR_BGR2RGB)
    overlay_image_RGBA = cv2.cvtColor(overlay_image, cv2.COLOR_BGRA2RGBA)

    #　PILに変換
    src_image_PIL=Image.fromarray(src_image_RGBA)
    overlay_image_PIL=Image.fromarray(overlay_image_RGBA)

    # 合成のため、RGBAモードに変更
    src_image_PIL = src_image_PIL.convert('RGBA')
    overlay_image_PIL = overlay_image_PIL.convert('RGBA')
    # 同じ大きさの透過キャンパスを用意 
    tmp = Image.new('RGBA', src_image_PIL.size, (255, 255,255, 0))
    # 用意したキャンパスに上書き
    tmp.paste(overlay_image_PIL, (posX, posY), overlay_image_PIL)
    # オリジナルとキャンパスを合成して保存
    result = Image.alpha_composite(src_image_PIL, tmp)

    return  cv2.cvtColor(np.asarray(result), cv2.COLOR_RGBA2BGRA)

def overlay_face():

    # 認識対象ファイルの読み込み
    image_path = "base.jpg"
    image = cv2.imread(image_path)

    # 上書きする画像の読み込み
    ol_imgae_path = "face_out.png"
    ol_image = cv2.imread(ol_imgae_path,cv2.IMREAD_UNCHANGED)   # アルファチャンネル(透過)も読みこむようにIMREAD_INCHANGEDを指定

    # グレースケールに変換
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 顔認識用特徴量のファイル指定
    cascade_path = "/usr/lib/anaconda/share/OpenCV/lbpcascades/lbpcascade_animeface.xml"
    #cascade_path = "/usr/lib/anaconda/share/OpenCV/haarcascades/haarcascade_frontalface_alt.xml"
    # カスケード分類器の特徴量を取得する
    cascade = cv2.CascadeClassifier(cascade_path)

    # 顔認識の実行
    facerecog = cascade.detectMultiScale(image_gray, scaleFactor=1.1, minNeighbors=1, minSize=(1, 1))


    if len(facerecog) > 0:


        # 認識結果を表示
        print ("認識結果")
        print ("(x,y)=(" + str(facerecog[P_ID][0]) + "," + str(facerecog[P_ID][1])+ ")" + \
        "  高さ："+str(facerecog[P_ID][2]) + \
        "  幅："+str(facerecog[P_ID][3]))

        # 認識範囲にあわせて画像をリサイズ
        resized_ol_image = resize_image(ol_image, facerecog[P_ID][2], facerecog[P_ID][3])

        # 上書きした画像の作成
        image = overlayOnPart(image, resized_ol_image, facerecog[P_ID][0], facerecog[P_ID][1])

    # 認識結果の出力
    cv2.imwrite("merge.png", image)


if __name__ == '__main__':
    overlay_face()
