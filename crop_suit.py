from PIL import Image
from tqdm import tqdm
Image.MAX_IMAGE_PIXELS = None

def crop_transparent(image_path, output_path, tolerance=10):
    # Open the image
    img = Image.open(image_path).convert("RGBA")
    datas = img.getdata()

    # Find the bounding box of the non-transparent pixels
    def is_transparent(pixel):
        return pixel[3] <= tolerance

    non_transparent_pixels = [
        (x, y) for y in tqdm(range(img.height)) for x in range(img.width) if not is_transparent(datas[y * img.width + x])
    ]

    if not non_transparent_pixels:
        print("The image is fully transparent.")
        return

    min_x = min(x for x, y in non_transparent_pixels)
    max_x = max(x for x, y in non_transparent_pixels)
    min_y = min(y for x, y in non_transparent_pixels)
    max_y = max(y for x, y in non_transparent_pixels)

    # Crop the image
    cropped_img = img.crop((min_x, min_y, max_x + 1, max_y + 1))
    cropped_img.save(output_path)

# Example usage
crop_transparent("0901a/0901aa_transparent_mosaic_group1.tif", "cropped_image.tif")