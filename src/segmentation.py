import io
import requests
from PIL import Image
import torch
import numpy
import pdb 
import numpy as np 
from utils import utils 

from transformers import DetrFeatureExtractor, DetrForSegmentation
from transformers.models.detr.feature_extraction_detr import rgb_to_id

def map_array(A, mapping_dict):
    B = np.array([mapping_dict[int(a)] for a in np.nditer(A)])
    return B.reshape(A.shape)

def get_binary_masks(panoptic_seg_id):
    # Get the unique ids in the panoptic_seg_id
    unique_ids = np.unique(panoptic_seg_id)
    
    # Initialize a list to store the binary masks
    binary_masks = []
    
    # Loop over the unique ids
    for id in unique_ids:
        # Create a binary mask for each id
        binary_mask = (panoptic_seg_id == id).astype(np.uint8)
        binary_masks.append(binary_mask)
        
    return binary_masks

def segment_image(image_path, labels_list, src_img, is_path=True, is_url=False):
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

    # List with dictionaries with info of the resulting segmentation maps 
    segment_info = result['segments_info']

    # Convert segment_id to category_id
    unique_seg_ids = np.unique(panoptic_seg_id).tolist()

    category_ids = {}
    for i in range(len(unique_seg_ids)):
        # Handle edge case problem when segment info is an empty list 
        if not segment_info:  
            area = panoptic_seg_id.shape[0] * panoptic_seg_id.shape[1]
            segment_info = [{'id': 0, 'isthing': False, 'category_id': 184, 'area': area}] # placeholder

        assert len(segment_info) == len(unique_seg_ids), "error segmenting this mask"
        category_id = segment_info[i]['category_id']
        category_ids[i] = category_id
    
    #pdb.set_trace()
    panoptic_seg_cat_ids = map_array(panoptic_seg_id, category_ids) # [H, W]

    # Resize 
    segm_ids = utils.resize_segmentation_mask(panoptic_seg_id, src_img.shape[0], src_img.shape[1])
    segm_cat_ids = utils.resize_segmentation_mask(panoptic_seg_cat_ids, src_img.shape[0], src_img.shape[1])

    # Build a dictionary to map id to name
    id_to_name = {d["id"]: d["name"] for d in labels_list}
    
    # Use this dictionary to map ids to names
    panoptic_seg_labels = np.array([id_to_name.get(int(id), "background") for id in np.nditer(segm_cat_ids)]).reshape(segm_ids.shape) # [H, W]
    
    # Get binary mask 
    binary_masks = get_binary_masks(segm_cat_ids) # list of length = num_labels with each element [H, W]

    return segm_ids, segm_cat_ids, panoptic_seg_labels, binary_masks
