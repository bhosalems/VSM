import cv2
import numpy as np
import os
from collections import defaultdict

# Main directory containing inference folders
main_dir = "/home/csgrad/mbhosale/phd/ddpm_hallucination/halmin/DDPM-IP/log/Shapes_Var_Off_25K/ema_0.9999_030000"

# Tolerances for white pixel counts
tolerances = {
    "triangle": (63, 67),
    "square": (60, 68),
    "pentagon": (68, 72)
}

def count_white_pixels(region):
    return np.sum(region == 255)

def analyze_image(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None or img.shape != (64, 64):
        return None, "invalid image shape"

    _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    img = img[:63, :63]

    total_white_pixels = np.sum(img == 255)
    if total_white_pixels < 50:
        return None, "black image"

    regions = [img[:, 0:21], img[:, 21:42], img[:, 42:63]]
    detected_shapes = []
    region_shape_map = {0: "triangle", 1: "square", 2: "pentagon"}

    for i, region in enumerate(regions):
        shape = region_shape_map[i]
        white_pixels = count_white_pixels(region)
        if tolerances[shape][0] <= white_pixels <= tolerances[shape][1]:
            detected_shapes.append(shape)

    num_shapes = len(detected_shapes)
    if num_shapes == 0:
        return None, "black image"
    elif num_shapes > 3:
        return None, "more than 3 shapes"
        
    return detected_shapes, None

def process_directory(directory_path):
    shape_counts = defaultdict(int)
    image_shape_counts = defaultdict(int)
    invalid_images = []
    interesting_images = []
    image_files = [f for f in os.listdir(directory_path) if f.endswith(".png")]
    for image_name in image_files:
        img_path = os.path.join(directory_path, image_name)
        result, error = analyze_image(img_path)
        if error:
            if error == "more than 3 shapes":
                interesting_images.append((image_name, error))
            else:                
                invalid_images.append((image_name, error))
        else:
            for shape in result:
                shape_counts[shape] += 1
            image_shape_counts[len(result)] += 1
            
    return {
        "shape_counts": dict(shape_counts),
        "image_shape_counts": dict(image_shape_counts),
        "invalid_images": invalid_images,
        "total_images": len(image_files),
        "interesting_hallucinations": interesting_images
    }

# Main processing
all_results = {}
total_summary = {
    "shape_counts": defaultdict(int),
    "image_shape_counts": defaultdict(int),
    "invalid_images": [],
    "total_images": 0
}

# Find all inference directories
for entry in os.listdir(main_dir):
    entry_path = os.path.join(main_dir, entry)
    if os.path.isdir(entry_path) and "inference_" in entry:
        images_dir = os.path.join(entry_path, "images")
        if os.path.exists(images_dir):
            print(f"\nProcessing directory: {entry}")
            dir_results = process_directory(images_dir)
            all_results[entry] = dir_results
            
            # Update total summary
            for shape, count in dir_results["shape_counts"].items():
                total_summary["shape_counts"][shape] += count
            for num, count in dir_results["image_shape_counts"].items():
                total_summary["image_shape_counts"][num] += count
            total_summary["invalid_images"].extend(dir_results["invalid_images"])
            total_summary["total_images"] += dir_results["total_images"]
    import json
    json.dump(dir_results["interesting_hallucinations"], open(os.path.join(entry_path,"shapes_hal_interesting.json"), "w"), indent=4)
    json.dump(dir_results, open(os.path.join(entry_path,"shapes_hal.json"), "w"), indent=4)

# Print individual directory reports
for dir_name, results in all_results.items():
    print(f"\nDirectory: {dir_name}")
    print(f"Total images: {results['total_images']}")
    print(f"Valid shapes: {results['shape_counts']}")
    print(f"Images with shapes: {dict(results['image_shape_counts'])}")
    print(f"Invalid images: {len(results['invalid_images'])}")

# Print final summary
print("\n=== TOTAL SUMMARY ===")
print(f"Total directories processed: {len(all_results)}")
print(f"Total images analyzed: {total_summary['total_images']}")
print(f"Total shape counts:")
for shape, count in total_summary["shape_counts"].items():
    print(f" - {shape}: {count}")
print("\nImages by shape count:")
for num, count in sorted(total_summary["image_shape_counts"].items()):
    print(f" - {num} shapes: {count}")
print(f"\nTotal invalid images: {len(total_summary['invalid_images'])}")

print("hallucination rate: ", (len(total_summary['invalid_images']) / total_summary['total_images'])*100, "%")



