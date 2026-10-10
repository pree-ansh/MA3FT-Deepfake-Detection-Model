import cv2
import numpy as np
from matplotlib import pyplot as plt
from ImageFeatures import extract_texture_features, extract_statistical_features
from VideoProcessing import get_frame_from_video



def calculate_noise_std(image, neighborhood_size=3):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Convert to floating-point type
    gray = gray.astype(np.float32)

    # Calculate the standard deviation of pixel values in small neighborhoods
    noise_std = cv2.boxFilter(gray, -1, (neighborhood_size, neighborhood_size))
    noise_std = cv2.sqrt(cv2.absdiff(cv2.pow(gray, 2), noise_std))

    return noise_std

def detect_noise_patterns(image_path1, image_path2, neighborhood_size=3, display=True):
    # Read the images

    if isinstance(image_path1,str):
        image1 = cv2.imread(image_path1)
        image2 = cv2.imread(image_path2)
    else:
        image1 = image_path1
        image2 = image_path2
    
    # Resize images to a common size
    common_size = (min(image1.shape[1], image2.shape[1]), min(image1.shape[0], image2.shape[0]))
    image1_resized = cv2.resize(image1, common_size, interpolation=cv2.INTER_AREA)
    image2_resized = cv2.resize(image2, common_size, interpolation=cv2.INTER_AREA)

    # Calculate noise standard deviations for each image
    noise_std1 = calculate_noise_std(image1_resized, neighborhood_size)
    noise_std2 = calculate_noise_std(image2_resized, neighborhood_size)

    noise_patterns = cv2.cvtColor(np.abs(noise_std1 - noise_std2).astype(np.uint8), cv2.COLOR_GRAY2RGB)

    # Display the images and the differences in noise patterns
    if display:
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 3, 1)
        plt.imshow(cv2.cvtColor(image1_resized, cv2.COLOR_BGR2RGB))
        plt.title('Image 1')
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.imshow(cv2.cvtColor(image2_resized, cv2.COLOR_BGR2RGB))
        plt.title('Image 2')
        plt.axis('off')

        
        plt.subplot(1, 3, 3)
        plt.imshow(noise_patterns)
        plt.title('Noise Patterns Artifacts')
        plt.axis('off')

        plt.show()
    
    return noise_patterns



if __name__ == '__main__':
    original_image_path = 'orig_cropped_face.png'
    compressed_image_path = 'fake_cropped_face.png'

    # Visualize differences in noise patterns
    detect_noise_patterns(original_image_path, compressed_image_path)

    dataset_path = 'E:/Projects/datasets/dfdc_train_part_0/'
    OriginalVideo = "wfzjxzhdkj.mp4"
    FakeVideo = "slhinthqcq.mp4"

    original_image = get_frame_from_video(dataset_path + OriginalVideo)

    fake_image = get_frame_from_video(dataset_path + FakeVideo)

    noise_patterns = detect_noise_patterns(original_image_path, compressed_image_path, display=False)

    if noise_patterns.ndim == 3:
        noise_patterns = noise_patterns[:,:,1]

    texture_Features =  extract_texture_features(noise_patterns, prefix='noise_patterns_')

    statistical_features = extract_statistical_features(noise_patterns, prefix='noise_patterns_')
    
    pass



