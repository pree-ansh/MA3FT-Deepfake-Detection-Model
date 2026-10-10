from skimage.feature import graycomatrix, graycoprops
import numpy as np
from scipy.stats import skew, kurtosis
from skimage.measure import shannon_entropy

def extract_texture_features(image,prefix = ''):
    """
    Extract texture features using the Grey-Level Co-occurrence Matrix (GLCM).
    
    Parameters:
    image (numpy.ndarray): The grayscale input image.
    
    Returns:
    dict: A dictionary of texture features (contrast, dissimilarity, homogeneity, energy).
    """
    if image.dtype == 'float32' or image.dtype == 'float64':
        # Scale float image to 0-255 and convert to uint8
        image = (image * 255).astype('uint8')
    elif image.dtype != 'uint8':
        # Directly convert other types to uint8 (you may lose precision if it's a larger type)
        image = image.astype('uint8')

    # Compute GLCM (Grey-Level Co-occurrence Matrix)
    glcm = graycomatrix(image, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    
    # Extract texture properties from GLCM
    texture_features = {
        prefix + 'contrast': graycoprops(glcm, 'contrast')[0, 0],
        prefix + 'dissimilarity': graycoprops(glcm, 'dissimilarity')[0, 0],
        prefix + 'homogeneity': graycoprops(glcm, 'homogeneity')[0, 0],
        prefix + 'energy': graycoprops(glcm, 'energy')[0, 0]
    }
    
    return texture_features


def extract_statistical_features(image,prefix = ''):
    """
    Extract relevant statistical features from an image (NumPy array).
    
    Parameters:
    image (numpy.ndarray): Input image (grayscale or color).
    
    Returns:
    dict: A dictionary containing statistical features of the image.
    """
    # If the image is color, convert it to grayscale for statistical analysis
    if len(image.shape) == 3:
        image = np.mean(image, axis=2)  # Convert to grayscale by averaging channels
    
    # Flatten the image array to a 1D array for easy statistics computation
    pixels = image.flatten()
    
    # Compute basic statistics
    mean_val = np.mean(pixels)
    std_dev = np.std(pixels)
    variance = np.var(pixels)
    min_val = np.min(pixels)
    max_val = np.max(pixels)
    skewness = skew(pixels)
    kurt = kurtosis(pixels)
    
    # Compute energy (sum of squared pixel intensities)
    energy = np.sum(np.square(pixels))
    
    # Compute entropy (using skimage exposure module)
    entropy = shannon_entropy(image)
    
    # Create a dictionary to hold all the features
    features = {
        prefix + 'mean': mean_val,
        prefix + 'std_dev': std_dev,
        prefix + 'variance': variance,
        prefix + 'min_val': min_val,
        prefix + 'max_val': max_val,
        prefix + 'skewness': skewness,
        prefix + 'kurtosis': kurt,
        prefix + 'energy': energy,
        prefix + 'entropy': entropy
    }
    
    return features