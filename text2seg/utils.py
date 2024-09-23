import torchvision.transforms as T
import numpy as np
from PIL import Image, ImageDraw, ImageFont

color_map = [
    [255/255, 51/255, 51/255, 0.6],   # 红色
    [51/255, 255/255, 51/255, 0.6],   # 绿色
    [51/255, 51/255, 255/255, 0.6],   # 蓝色
    [255/255, 255/255, 51/255, 0.6],  # 黄色
    [255/255, 51/255, 255/255, 0.6],  # 紫色
    [51/255, 255/255, 255/255, 0.6],  # 青色
    [128/255, 0/255, 128/255, 0.6],   # 深紫色
    [0/255, 128/255, 128/255, 0.6],   # 深青色
    [128/255, 128/255, 0/255, 0.6],   # 橄榄色
    [192/255, 192/255, 192/255, 0.6]  # 灰色
]


def transform_image(image):
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    image_transformed, _ = transform(image, None)
    return image_transformed


def draw_mask(mask, image):
    color_map = np.array([255/255, 51/255, 51/255, 0.6])
    h, w = mask.shape[-2:]
    # mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    mask = mask.reshape(h, w, 1) * color_map.reshape(1, 1, -1)  # (1024, 1024) -> (1024, 1024, 4)
    annotated_frame_pil = Image.fromarray(image).convert("RGBA")
    mask_image_pil = Image.fromarray((mask * 255).astype(np.uint8)).convert("RGBA")
    return np.array(Image.alpha_composite(annotated_frame_pil, mask_image_pil))

def draw_mask_multi(masks, image, label_list):

    label_color_map = {
        'barren': [255/255, 51/255, 51/255, 0.6],   # 红色
        'grassland': [51/255, 255/255, 51/255, 0.6],   # 绿色
        'car': [51/255, 51/255, 255/255, 0.6],   # 蓝色
        'obstacle': [255/255, 255/255, 51/255, 0.6],  # 黄色
        'buildings': [255/255, 51/255, 255/255, 0.6],  # 紫色
        'bulldozer': [51/255, 255/255, 255/255, 0.6],  # 青色
    }

    annotated_frame_pil = Image.fromarray(image).convert("RGBA")
    
    for i, mask in enumerate(masks):
        label = label_list[i]
        color = np.array(label_color_map[label])
        h, w = mask.shape[-2:]
        mask = ((mask.sum(dim=0)>0)*1).cpu().numpy()
        mask = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        mask_image_pil = Image.fromarray((mask * 255).astype(np.uint8)).convert("RGBA")
        annotated_frame_pil = Image.alpha_composite(annotated_frame_pil, mask_image_pil)
    
    # # 添加图例
    # draw = ImageDraw.Draw(annotated_frame_pil)
    # font = ImageFont.load_default()
    # legend_x, legend_y = 10, 10
    # for label, color in label_color_map.items():
    #     draw.rectangle([legend_x, legend_y, legend_x + 20, legend_y + 20], fill=tuple((np.array(color) * 255).astype(int)))
    #     draw.text((legend_x + 30, legend_y), label, fill=(255, 255, 255, 255), font=font)
    #     legend_y += 30
    
    return np.array(annotated_frame_pil)

def segment_image(image, segmentation_mask):
    image_array = image
    segmented_image_array = np.zeros_like(image_array)
    segmented_image_array[segmentation_mask] = image_array[segmentation_mask]
    segmented_image = Image.fromarray(segmented_image_array)
    black_image = Image.new("RGB", image.shape[:2], (0, 0, 0))
    transparency_mask = np.zeros_like(segmentation_mask, dtype=np.uint8)
    transparency_mask[segmentation_mask] = 255
    transparency_mask_image = Image.fromarray(transparency_mask, mode='L')
    black_image.paste(segmented_image, mask=transparency_mask_image)
    return black_image

def convert_box_xywh_to_xyxy(box):
    x1 = box[0]
    y1 = box[1]
    x2 = box[0] + box[2]
    y2 = box[1] + box[3]
    return [x1, y1, x2, y2]


