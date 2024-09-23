from PIL import Image, ImageFile
import os
from tqdm import tqdm
import numpy as np

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

def split_tif_to_png(tif_path, output_dir, tile_size=(2494, 2064)):
    # 打开Tif文件
    tif_image = Image.open(tif_path)
    width, height = tif_image.size
    
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 遍历图像，按块大小裁剪图像
    for top in tqdm(range(0, height, tile_size[1]), desc="Rows"):
        for left in tqdm(range(0, width, tile_size[0]), desc="Columns", leave=False):
            box = (left, top, left + tile_size[0], top + tile_size[1])
            tile = tif_image.crop(box)

            # 转换为numpy数组
            tile_np = np.array(tile)

            # 检查是否含有空白像素
            if np.any(tile_np == 255):
                # 保存为png文件
                tile_path = os.path.join(output_dir, f"tile_{top}_{left}.png")
                tile.save(tile_path)

# 示例用法
tif_path = "data/0901a/0901aa_transparent_mosaic_group1.tif"
output_dir = "data/0901a/large_tiles"

split_tif_to_png(tif_path, output_dir)