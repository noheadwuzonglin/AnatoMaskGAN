import os
import cv2

def resize_image(image_path, output_size=(224, 224)):
    # 读取图片
    img = cv2.imread(image_path)
    if img is None:
        print("加载图片失败:", image_path)
        return
    # 使用双线性插值进行图片缩放
    resized_img = cv2.resize(img, output_size, interpolation=cv2.INTER_LINEAR)
    # 保存图片（覆盖原文件）
    cv2.imwrite(image_path, resized_img)

def process_folder(root_folder):
    # 遍历根目录下所有子目录
    for subdir, dirs, files in os.walk(root_folder):
        # 获取当前文件夹的名称
        folder_name = os.path.basename(subdir)
        # 判断文件夹名称是否为 image 或 label
        if folder_name.lower() in ['image', 'mask']:
            for file in files:
                # 判断是否为常见图片格式
                if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
                    image_path = os.path.join(subdir, file)
                    print("正在处理:", image_path)
                    resize_image(image_path)

if __name__ == '__main__':
    # 请将此处路径修改为你需要处理的根文件夹路径
    root_folder = r"C:\Users\Ming\Desktop\SliceFS-net\datasets\testl2"
    process_folder(root_folder)
