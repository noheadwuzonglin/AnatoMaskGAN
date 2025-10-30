import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import colorsys
import random
from typing import Dict, Tuple, List


def grayscale_to_color_onehot(input_dir, output_dir):
    """
    将灰度掩码转换为彩色one-hot编码掩码（使用柔和协调的颜色，背景为暗绿色）
    :param input_dir: 输入文件夹路径（包含灰度掩码）
    :param output_dir: 输出文件夹路径（将保存彩色掩码）
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 获取所有图像文件
    img_files = [f for f in os.listdir(input_dir)
                 if f.endswith(('.png', '.jpg', '.jpeg', '.tif', '.bmp', '.tiff'))]

    if not img_files:
        print(f"在目录 {input_dir} 中未找到图像文件")
        return

    print(f"找到 {len(img_files)} 个图像文件，开始转换...")

    # 定义暗绿色背景
    BACKGROUND_COLOR = (51, 102, 51)  # 暗绿色 (RGB: 51, 102, 51)

    # 预定义一组柔和协调的颜色
    soft_colors = [
        # 柔和的基础色
        (230, 159, 0),  # 柔和橙色
        (86, 180, 233),  # 柔和蓝色
        (0, 158, 115),  # 柔和绿色
        (240, 228, 66),  # 柔和黄色
        (0, 114, 178),  # 柔和深蓝色
        (213, 94, 0),  # 柔和红褐色
        (204, 121, 167),  # 柔和紫色

        # 柔和的浅色变体
        (255, 204, 153),  # 浅橙色
        (153, 204, 255),  # 浅蓝色
        (153, 255, 204),  # 浅绿色
        (255, 255, 153),  # 浅黄色
        (153, 153, 255),  # 浅紫色
        (255, 153, 153),  # 浅红色
        (204, 255, 255),  # 浅青色

        # 柔和的中间色
        (192, 192, 192),  # 浅灰色
        (255, 192, 203),  # 浅粉色
        (221, 160, 221),  # 浅紫罗兰色
        (173, 216, 230),  # 浅天蓝色
        (245, 222, 179),  # 小麦色
        (144, 238, 144),  # 浅绿色
        (255, 182, 193),  # 浅粉色

        # 更多柔和的中间色
        (255, 160, 122),  # 浅珊瑚色
        (255, 218, 185),  # 桃色
        (175, 238, 238),  # 浅青色
        (255, 228, 196),  # 浅桃色
        (220, 220, 220),  # 浅灰色
        (250, 235, 215),  # 古董白色
        (255, 240, 245),  # 薰衣草淡紫色
        (240, 248, 255),  # 爱丽丝蓝色
    ]

    # 存储全局的类别-颜色映射
    global_class_color_map = {0: BACKGROUND_COLOR}  # 背景类别(0)映射到暗绿色

    for i, filename in enumerate(img_files):
        print(f"\n处理文件: {filename}")
        # 读取灰度图像
        img_path = os.path.join(input_dir, filename)
        grayscale = np.array(Image.open(img_path).convert('L'))

        # 获取图像中的唯一类别ID
        class_ids = np.unique(grayscale)
        num_classes = len(class_ids)

        print(f"  找到 {num_classes} 个类别: {class_ids}")

        # 创建彩色图像（RGB），初始化为背景色
        color_mask = np.full((*grayscale.shape, 3), BACKGROUND_COLOR, dtype=np.uint8)

        # 记录当前图像的类别-颜色映射
        class_color_map = {0: BACKGROUND_COLOR}  # 背景类别

        # 获取前景类别ID（排除背景0）
        foreground_ids = class_ids[class_ids > 0]

        # 为每个前景类别分配柔和协调的颜色
        for idx, class_id in enumerate(foreground_ids):
            # 方法1：使用预定义的柔和颜色
            color_idx = idx % len(soft_colors)  # 循环使用颜色列表
            r, g, b = soft_colors[color_idx]

            # 保存类别-颜色映射
            class_color_map[class_id] = (r, g, b)

            # 如果是新类别，添加到全局映射
            if class_id not in global_class_color_map:
                global_class_color_map[class_id] = (r, g, b)

            # 创建该类别的掩码
            class_mask = (grayscale == class_id)

            # 应用颜色
            color_mask[class_mask, 0] = r  # 红色通道
            color_mask[class_mask, 1] = g  # 绿色通道
            color_mask[class_mask, 2] = b  # 蓝色通道

        # 打印当前图像的类别-颜色映射
        print("  类别-颜色映射:")
        for class_id, color in sorted(class_color_map.items()):
            print(f"    类别 {class_id}: RGB{color}")

        # 保存为PNG图像
        output_path = os.path.join(output_dir, os.path.splitext(filename)[0] + '.png')
        Image.fromarray(color_mask).save(output_path)

        if (i + 1) % 10 == 0 or (i + 1) == len(img_files):
            print(f"已处理 {i + 1}/{len(img_files)} 文件")

    # 打印全局类别-颜色映射
    print("\n全局类别-颜色映射:")
    for class_id, color in sorted(global_class_color_map.items()):
        print(f"  类别 {class_id}: RGB{color}")

    print("\n转换完成！所有彩色掩码已保存为PNG格式（柔和色调，背景为暗绿色）")


if __name__ == "__main__":
    input_folder = r"C:\Users\Ming\Desktop\SliceFS-net\datasets\23"
    output_folder = r"C:\Users\Ming\Desktop\SliceFS-net\datasets\123"

    grayscale_to_color_onehot(input_folder, output_folder)
    print("转换完成！所有彩色掩码已保存为PNG格式（柔和色调，背景为暗绿色）")