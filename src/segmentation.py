import io
import requests
from PIL import Image
import torch
import numpy
import pdb 
import numpy as np 

from transformers import DetrFeatureExtractor, DetrForSegmentation
from transformers.models.detr.feature_extraction_detr import rgb_to_id



def segment_image(image_path, is_path=True, is_url=False):
    """
    Takes in an image as path or numpy array and returns segmentation ID map [H, W]. 

    Careful because if image is smaller than 1000 pixels will resize 
    """
    # Open image 
    if is_url:
        url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        image = Image.open(requests.get(url, stream=True).raw) # PIL image 
    
    if is_path:
        image = Image.open(image_path).convert('RGB') # PIL image 
    
    else:
        # Convert the NumPy array to a PIL image
        image = Image.fromarray(np.uint8(image_path))

    feature_extractor = DetrFeatureExtractor.from_pretrained("facebook/detr-resnet-50-panoptic")
    model = DetrForSegmentation.from_pretrained("facebook/detr-resnet-50-panoptic")

    # prepare image for the model
    print('image: ', image)
    inputs = feature_extractor(images=image, return_tensors="pt")

    # forward pass
    outputs = model(**inputs)
    
    # use the `post_process_panoptic` method of `DetrFeatureExtractor` to convert to COCO format
    processed_sizes = torch.as_tensor(inputs["pixel_values"].shape[-2:]).unsqueeze(0)
    result = feature_extractor.post_process_panoptic(outputs, processed_sizes)[0]

    # the segmentation is stored in a special-format png
    panoptic_seg = Image.open(io.BytesIO(result["png_string"]))
    panoptic_seg = numpy.array(panoptic_seg, dtype=numpy.uint8) # [H, W, C]
    
    # retrieve the ids corresponding to each mask
    panoptic_seg_id = rgb_to_id(panoptic_seg) # [H, W]

    return panoptic_seg_id
