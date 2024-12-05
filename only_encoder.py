from segment_anything import sam_model_registry, SamPredictor
import cv2
import torch

sam_checkpoint = "sam_vit_b_01ec64.pth"
model_type = "vit_b"

device = "cuda"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

predictor = SamPredictor(sam)

image_path = r"C:\Users\diego\Downloads\descarga (6).png"  # Replace with your image file path
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for SAM
image = cv2.pyrDown(image)

input_image = predictor.transform.apply_image(image)
input_image_torch = torch.as_tensor(input_image, device="cuda")
input_image_torch = input_image_torch.permute(2, 0, 1).contiguous()[None, :, :, :]

input_image = predictor.model.preprocess(input_image_torch)
features = predictor.model.image_encoder(input_image)

