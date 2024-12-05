import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

# Path to the SAM model checkpoint
sam_checkpoint = "sam_vit_b_01ec64.pth"
model_type = "vit_b"

# Device configuration
device = "cuda"  # """"cuda" if torch.cuda.is_available() else "cpu"

# Initialize the SAM model
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

# Configure the automatic mask generator
mask_generator = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side=8,
    pred_iou_thresh=0.86,
    stability_score_thresh=0.92,
    crop_n_layers=1,
    crop_n_points_downscale_factor=2,
    min_mask_region_area=100,
)

# Load the input image
image_path = r"C:\Users\diego\Downloads\descarga (6).png"  # Replace with your image file path
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for SAM
image = cv2.pyrDown(image)

# Generate masks
masks = mask_generator.generate(image)


# Function to display annotations with different colors
def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:, :, 3] = 0  # Alpha channel initialized to 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])  # Random RGB with alpha 0.35
        img[m] = color_mask
    ax.imshow(img)


# Plot the results
plt.figure(figsize=(10, 10))
plt.imshow(image)
show_anns(masks)
plt.axis('off')
plt.show()
