# import packages
           
from text2seg.text2seg import Text2Seg
from matplotlib import pyplot as plt
import os
import numpy as np

label_color_map = {
    'barren': [255/255, 51/255, 51/255, 0.6],   # 红色
    'grassland': [51/255, 255/255, 51/255, 0.6],   # 绿色
    'car': [51/255, 51/255, 255/255, 0.6],   # 蓝色
    'obstacle': [255/255, 255/255, 51/255, 0.6],  # 黄色
    'buildings': [255/255, 51/255, 255/255, 0.6],  # 紫色
    'bulldozer': [51/255, 255/255, 255/255, 0.6],  # 青色
}

def add_legend(ax, label_color_map):
    # 添加图例
    for label, color in label_color_map.items():
        ax.plot([], [], color=np.array(color[:3]), label=label, linewidth=10)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), fontsize='small', frameon=False, ncol=len(label_color_map))

# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
# initialize Text2Seg object
test = Text2Seg()
# # generate masks using SAM + GroundingDINO
plt.rcParams['figure.dpi'] = 300
masks, annotated_frame_with_mask, annotated_frame, annotated_frame_with_mask_all = test.predict_dino("tile_83968_26624.png", ["barren", "grassland", "car", "obstacle", "buildings", "bulldozer"])
# plt.imshow(annotated_frame_with_mask)
# plt.axis('off')  # 去除坐标系
# add_legend(plt.gca(), label_color_map)  # 添加图例
# plt.savefig("output.png")

output_dir = "output"
# 检查目录是否存在，如果不存在则创建
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

plt.imshow(annotated_frame_with_mask_all)
plt.axis('off')  # 去除坐标系
# add_legend(plt.gca(), label_color_map)  # 添加图例
plt.savefig("output_all.png", bbox_inches='tight', pad_inches=0)