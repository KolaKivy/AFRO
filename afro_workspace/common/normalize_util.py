import numpy as np
from typing import Dict, List

def get_range_normalizer(array: np.ndarray):
    """
    Given an array, returns a function that normalizes a value to [-1, 1]
    and a function that denormalizes a value from [-1, 1] to the original range.
    """
    max_v = array.max(axis=0)
    min_v = array.min(axis=0)
    
    def normalize(value):
        # (value - min) / (max - min) -> [0, 1]
        # 2 * ... - 1 -> [-1, 1]
        return 2 * (value - min_v) / (max_v - min_v + 1e-8) - 1

    def denormalize(value):
        # (value + 1) / 2 -> [0, 1]
        # ... * (max - min) + min -> [min, max]
        return (value + 1) / 2 * (max_v - min_v + 1e-8) + min_v
    
    return normalize, denormalize

def get_image_range_normalizer():
    """
    Returns a function that normalizes an image from [0, 255] to [-1, 1].
    """
    def normalize(image):
        return image.astype(np.float32) / 127.5 - 1.0
    
    def denormalize(image):
        return ((image + 1.0) * 127.5).astype(np.uint8)
        
    return normalize, denormalize

def get_identity_normalizer():
    """
    Returns a function that does nothing.
    """
    def normalize(value):
        return value
    
    def denormalize(value):
        return value
    
    return normalize, denormalize

def get_string_normalizer(array: np.ndarray):
    """
    Returns a function that maps a string to an integer and back.
    """
    vocab = np.unique(array)
    key_to_idx = {key: i for i, key in enumerate(vocab)}
    
    def normalize(key):
        return key_to_idx.get(key, -1) # Return -1 for out-of-vocabulary keys
    
    def denormalize(idx):
        if idx < 0 or idx >= len(vocab):
            return None # Or some default string
        return vocab[idx]
        
    return normalize, denormalize

class Normalizer:
    def __init__(self, normalizer_funcs):
        self.normalizer_funcs = normalizer_funcs

    def __call__(self, data):
        return self.normalize(data)

    def normalize(self, data):
        normalized_data = dict()
        for key, funcs in self.normalizer_funcs.items():
            normalize_func, _ = funcs
            normalized_data[key] = normalize_func(data[key])
        return normalized_data

    def denormalize(self, data):
        denormalized_data = dict()
        for key, funcs in self.normalizer_funcs.items():
            _, denormalize_func = funcs
            denormalized_data[key] = denormalize_func(data[key])
        return denormalized_data

class SingleFieldNormalizer:
    def __init__(self, normalize_func, denormalize_func):
        self.normalize_func = normalize_func
        self.denormalize_func = denormalize_func

    def __call__(self, data):
        return self.normalize(data)

    def normalize(self, data):
        return self.normalize_func(data)

    def denormalize(self, data):
        return self.denormalize_func(data)
