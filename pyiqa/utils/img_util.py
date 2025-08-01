import cv2
import math
import numpy as np
import torch
from PIL import Image, ImageOps
import io
import warnings
from functools import lru_cache
import torchvision.transforms.functional as TF

# Cache frequently used transforms for performance
@lru_cache(maxsize=16)
def _get_cached_transform(size=None, interpolation=Image.BILINEAR):
    """Get cached torchvision transform for common operations."""
    transforms = []
    if size is not None:
        transforms.append(TF.resize)
    return transforms

def imread2tensor(img_path, rgb=True, out_type=np.float32, backend='auto'):
    """
    Optimized image reading and conversion to tensor.
    
    Args:
        img_path: Path to image or image array
        rgb: Whether to return RGB (True) or BGR (False)
        out_type: Output data type
        backend: Backend to use ('auto', 'cv2', 'pil')
    """
    if isinstance(img_path, (str, io.IOBase)):
        # Choose optimal backend
        if backend == 'auto':
            # Use PIL for JPEG/PNG, CV2 for others
            if isinstance(img_path, str) and img_path.lower().endswith(('.jpg', '.jpeg', '.png')):
                backend = 'pil'
            else:
                backend = 'cv2'
        
        if backend == 'pil':
            # PIL is often faster for JPEG/PNG
            try:
                img = Image.open(img_path)
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                img = np.array(img, dtype=out_type)
                if not rgb:  # Convert to BGR if needed
                    img = img[:, :, ::-1]
            except Exception:
                # Fallback to CV2
                backend = 'cv2'
        
        if backend == 'cv2':
            # OpenCV reading
            img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
            if img is None:
                raise IOError(f'Cannot read image: {img_path}')
            
            if img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            elif img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
            elif not rgb:
                pass  # Keep BGR
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            img = img.astype(out_type)
    else:
        # Already an array
        img = img_path.astype(out_type)
    
    # Normalize to [0, 1] if needed
    if img.max() > 1.1:  # Assume 8-bit image
        img = img / 255.0
    
    # Convert to tensor efficiently
    if len(img.shape) == 2:
        img = img[:, :, None]
    
    # HWC to CHW conversion - use contiguous for better performance
    img_tensor = torch.from_numpy(img.transpose(2, 0, 1)).contiguous()
    
    return img_tensor.float()

def img2tensor(imgs, bgr2rgb=True, float32=True):
    """
    Optimized numpy array to tensor conversion.
    
    Args:
        imgs: Input images (numpy array or list of arrays)
        bgr2rgb: Whether to convert BGR to RGB
        float32: Whether to convert to float32
    """
    def _img2tensor_single(img):
        if img.shape[2] == 3 and bgr2rgb:
            # More efficient BGR to RGB conversion
            img = img[:, :, ::-1]  # Using slicing instead of cv2.cvtColor
        
        # Use efficient transpose and contiguous
        img_tensor = torch.from_numpy(img.transpose(2, 0, 1)).contiguous()
        
        if float32:
            img_tensor = img_tensor.float()
        return img_tensor

    if isinstance(imgs, list):
        return [_img2tensor_single(img) for img in imgs]
    else:
        return _img2tensor_single(imgs)

def tensor2img(tensor, rgb2bgr=True, out_type=np.uint8, min_max=(0, 1)):
    """
    Optimized tensor to image conversion.
    
    Args:
        tensor: Input tensor
        rgb2bgr: Whether to convert RGB to BGR
        out_type: Output data type
        min_max: Value range for normalization
    """
    if not (torch.is_tensor(tensor) or (isinstance(tensor, np.ndarray) and len(tensor.shape) == 4)):
        raise TypeError(f'Input type is not supported: {type(tensor)}')

    if torch.is_tensor(tensor):
        # Move to CPU if needed
        if tensor.device != torch.device('cpu'):
            tensor = tensor.detach().cpu()
        tensor = tensor.numpy()

    # Handle different tensor shapes efficiently
    if len(tensor.shape) == 4:
        # Batch of images - process efficiently
        result = []
        for i in range(tensor.shape[0]):
            img = tensor[i]
            img = _process_single_tensor(img, rgb2bgr, out_type, min_max)
            result.append(img)
        return result
    elif len(tensor.shape) == 3:
        return _process_single_tensor(tensor, rgb2bgr, out_type, min_max)
    else:
        raise ValueError(f'Unsupported tensor shape: {tensor.shape}')

def _process_single_tensor(tensor, rgb2bgr, out_type, min_max):
    """Process a single tensor image efficiently."""
    # Normalize to [0, 1]
    tensor = (tensor - min_max[0]) / (min_max[1] - min_max[0])
    tensor = tensor.clip(0, 1)
    
    # CHW to HWC
    if tensor.shape[0] in [1, 3]:
        tensor = tensor.transpose(1, 2, 0)
    
    # Convert data type efficiently
    if out_type == np.uint8:
        # Optimized uint8 conversion
        tensor = (tensor * 255.0).round().astype(np.uint8)
    else:
        tensor = tensor.astype(out_type)
    
    # Handle color space conversion
    if tensor.shape[2] == 3 and rgb2bgr:
        tensor = tensor[:, :, ::-1]  # Efficient BGR conversion
    
    return tensor
