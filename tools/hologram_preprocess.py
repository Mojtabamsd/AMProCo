import os
from PIL import Image
import numpy as np
import argparse


def rescale_image(image):
    image = np.array(image).astype(np.float32)
    min_val, max_val = image.min(), image.max()
    if max_val == min_val:
        scaled_image = np.full(image.shape, 255 if max_val > 0 else 0, dtype=np.uint8)
    else:
        scaled_image = 255 * (image - min_val) / (max_val - min_val)
    return Image.fromarray(scaled_image.astype(np.uint8))


def process_images(source_folder, dest_folder):
    for root, dirs, files in os.walk(source_folder):
        for file in files:
            if file.lower().endswith('.pgm'):
                source_file_path = os.path.join(root, file)
                with Image.open(source_file_path) as img:
                    scaled_img = rescale_image(img)
                    # scaled_img = img

                    relative_path = os.path.relpath(root, source_folder)
                    dest_dir = os.path.join(dest_folder, relative_path)
                    os.makedirs(dest_dir, exist_ok=True)

                    dest_file_path = os.path.join(dest_dir, f'{os.path.splitext(file)[0]}.bmp')
                    scaled_img.save(dest_file_path)
                    print(f"Processed and saved: {dest_file_path}")


if __name__ == '__main__':
    # Create the argument parser
    parser = argparse.ArgumentParser(description='Process PGM files, rescale them to [0,255] and save as BMP.')

    parser.add_argument('-i', '--input', required=True, help='Path to the source folder containing PGM files.')
    parser.add_argument('-o', '--output', required=True, help='Path to the destination folder for BMP files.')

    # Parse the arguments
    args = parser.parse_args()

    # # debug
    # args.input = r'D:\mojmas\files\data\result_sampling\lisstholo'
    # args.output = r'D:\mojmas\files\data\result_sampling\lisstholo_result'

    # Process the images using the provided arguments
    process_images(args.input, args.output)
