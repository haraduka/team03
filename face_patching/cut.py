import cv2
P_ID = 1

def cut_face():

    # 認識対象ファイルの読み込み
    image_path = "base.png"
    image = cv2.imread(image_path)

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

        # 切り抜いた画像の作成
        start_x = int(facerecog[P_ID][0])
        start_y = int(facerecog[P_ID][1])
        dst = image[start_y:(start_y+facerecog[P_ID][2]), start_x:(start_x+facerecog[P_ID][3])]

    # 認識結果の出力
    cv2.imwrite("cut.png", dst)


if __name__ == '__main__':
    
    cut_face()
