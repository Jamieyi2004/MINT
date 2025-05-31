import os
import shutil

# 设置源目录
src_dir = "./datasets/mini-imagenet"

# 遍历源目录下的所有文件
for filename in os.listdir(src_dir):
    if filename.endswith(".jpg") and filename.startswith("n"):
        # 提取类别名（例如 n01532829）
        class_name = filename[:9]  # n + 8位数字
        # 提取样本编号（去掉类别名）
        sample_name = filename[9:]  # 后面部分如 00000005.jpg

        # 创建类别目录（如果不存在）
        class_dir = os.path.join(src_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)

        # 构建源文件路径和目标文件路径
        src_path = os.path.join(src_dir, filename)
        dst_path = os.path.join(class_dir, sample_name)

        # 移动并重命名文件
        shutil.move(src_path, dst_path)

print("mini-imagenet转换完成：图像已按类别归档并重命名。")
