import os
import cv2
import numpy as np

# ==========================================
# 問題一：house.tiff 格式轉換與檔案大小比較
# ==========================================
img_house = cv2.imread('house.tif', cv2.IMREAD_GRAYSCALE)

if img_house is not None:
    # 轉寫成 JPEG 與 PNG
    cv2.imwrite('house_output.jpg', img_house)
    cv2.imwrite('house_output.png', img_house)

    size_house_tiff = os.path.getsize('house.tif') / 1024
    size_house_jpg = os.path.getsize('house_output.jpg') / 1024
    size_house_png = os.path.getsize('house_output.png') / 1024

    print("=== 第一題結果 ===")
    print(f"house.tif  檔案大小: {size_house_tiff:.2f} KB")
    print(f"house.jpg  檔案大小: {size_house_jpg:.2f} KB")
    print(f"house.png  檔案大小: {size_house_png:.2f} KB\n")
else:
    print("找不到 house.tif，請檢查圖片是否存放在專案根目錄下！")


# ==========================================
# 問題二：bridge.tiff 轉二元與全彩影像
# ==========================================
img_bridge = cv2.imread('bridge.tif', cv2.IMREAD_GRAYSCALE)

if img_bridge is not None:
    # 1. 轉二元 (Binary) 影像
    _, img_bridge_binary = cv2.threshold(img_bridge, 128, 255, cv2.THRESH_BINARY)
    cv2.imwrite('bridge_binary.png', img_bridge_binary)

    # 2. 轉全彩 (True Color/RGB) 影像
    img_bridge_color = cv2.cvtColor(img_bridge, cv2.COLOR_GRAY2BGR)
    cv2.imwrite('bridge_color.png', img_bridge_color)

    size_bridge_tiff = os.path.getsize('bridge.tif') / 1024
    size_bridge_binary = os.path.getsize('bridge_binary.png') / 1024
    size_bridge_color = os.path.getsize('bridge_color.png') / 1024

    print("=== 第二題結果 ===")
    print(f"bridge.tif        檔案大小: {size_bridge_tiff:.2f} KB")
    print(f"bridge_binary.png 檔案大小: {size_bridge_binary:.2f} KB")
    print(f"bridge_color.png  檔案大小: {size_bridge_color:.2f} KB\n")
else:
    print("找不到 bridge.tif，請檢查圖片是否存放在專案根目錄下！")


# ==========================================
# 問題三：量化成 2 灰階、4 灰階與混色 (Dithering)
# ==========================================
if img_bridge is not None:
    # 1. 2 灰階量化 (0, 255)
    img_2gray = np.where(img_bridge >= 128, 255, 0).astype(np.uint8)
    cv2.imwrite('bridge_2gray.png', img_2gray)

    # 2. 4 灰階量化 (0, 85, 170, 255)
    img_4gray = (np.floor(img_bridge / 64) * 85).astype(np.uint8)
    cv2.imwrite('bridge_4gray.png', img_4gray)

    # 3. 2 灰階混色 (Floyd-Steinberg Dithering 演算法)
    def floyd_steinberg_dithering(img):
        h, w = img.shape
        img_float = img.astype(float)
        for y in range(h):
            for x in range(w):
                old_val = img_float[y, x]
                new_val = 255 if old_val >= 128 else 0
                img_float[y, x] = new_val
                err = old_val - new_val

                if x + 1 < w:
                    img_float[y, x + 1] += err * 7 / 16
                if y + 1 < h and x > 0:
                    img_float[y + 1, x - 1] += err * 3 / 16
                if y + 1 < h:
                    img_float[y + 1, x] += err * 5 / 16
                if y + 1 < h and x + 1 < w:
                    img_float[y + 1, x + 1] += err * 1 / 16
        return np.clip(img_float, 0, 255).astype(np.uint8)

    img_dither = floyd_steinberg_dithering(img_bridge)
    cv2.imwrite('bridge_dither.png', img_dither)

    print("=== 第三題圖片處理完成 ===")
    print("1. 已匯出 bridge_2gray.png (2 灰階黑白)")
    print("2. 已匯出 bridge_4gray.png (4 灰階)")
    print("3. 已匯出 bridge_dither.png (2 灰階混色)")