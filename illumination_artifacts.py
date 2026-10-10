import cv2
import numpy as np
from matplotlib import pyplot as plt
from ImageFeatures import extract_texture_features, extract_statistical_features
from VideoProcessing import get_frame_from_video

def calculate_mean_brightness(image):
    # Convert the image to grayscale
    if isinstance(image,str):
        image = cv2.imread(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Calculate the mean brightness
    mean_brightness = np.mean(gray)

    return mean_brightness

def detect_illumination_artifacts(image_path1, image_path2, display=True):
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

    # Calculate mean brightness for each image
    mean_brightness1 = calculate_mean_brightness(image1_resized)
    mean_brightness2 = calculate_mean_brightness(image2_resized)

    # Display the images and mean brightness values
    if display:
        plt.figure(figsize=(10, 5))

        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(image1_resized, cv2.COLOR_BGR2RGB))
        plt.title(f'Mean Brightness: {mean_brightness1:.2f}')
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(cv2.cvtColor(image2_resized, cv2.COLOR_BGR2RGB))
        plt.title(f'Mean Brightness: {mean_brightness2:.2f}')
        plt.axis('off')

        plt.show(block=True)

    # Return the absolute Artifact in mean brightness
    return np.abs(mean_brightness1 - mean_brightness2)

def create_illumination_heatmap(image1, image2):

    if isinstance(image1,str):
        image1 = cv2.imread(image1)
        image2 = cv2.imread(image2)
    else:
        image1 = image1
        image2 = image2

    # Calculate mean brightness for each image
    mean_brightness1 = calculate_mean_brightness(image1)
    mean_brightness2 = calculate_mean_brightness(image2)

    # Compute the absolute Artifact in mean brightness
    illumination_Artifact =  np.abs(image2 - mean_brightness2)

    # Create a heatmap based on the illumination Artifact
    heatmap = np.zeros_like(image1)
    heatmap = illumination_Artifact  # Red channel for visualization

    return heatmap

def visualize_illumination_Artifacts(image_path1, image_path2):
    # Read the images
    image1 = cv2.imread(image_path1)
    image2 = cv2.imread(image_path2)

    # Resize images to a common size
    common_size = (min(image1.shape[1], image2.shape[1]), min(image1.shape[0], image2.shape[0]))
    image1_resized = cv2.resize(image1, common_size, interpolation=cv2.INTER_AREA)
    image2_resized = cv2.resize(image2, common_size, interpolation=cv2.INTER_AREA)

    # Create the illumination heatmap
    heatmap = create_illumination_heatmap(image1_resized, image2_resized)

    # Display the images and the illumination heatmap
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(image1_resized, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(cv2.cvtColor(image2_resized, cv2.COLOR_BGR2RGB))
    plt.title('DeepFake Image')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(heatmap)
    plt.title('Illumination Artifact Heatmap')
    plt.axis('off')

    plt.show(block=True)


if __name__ == '__main__':
    
    image_path1 = 'fake_cropped_face.png'
    image_path2 = 'orig_cropped_face.png'

    # Detect illumination Artifacts and optionally display the result
    mean_brightness = detect_illumination_artifacts(image_path1, image_path2, display=True)

    # Visualize illumination Artifacts
    visualize_illumination_Artifacts(image_path1, image_path2)

    dataset_path = 'E:/Projects/datasets/dfdc_train_part_0/'
    OriginalVideo = "wfzjxzhdkj.mp4"
    FakeVideo = "slhinthqcq.mp4"

    # Visualize block-wise compression artifacts
    original_image = get_frame_from_video(dataset_path + OriginalVideo)

    fake_image = get_frame_from_video(dataset_path + FakeVideo)

    # Detect illumination Artifacts and optionally display the result
    illumination_artifacts = detect_illumination_artifacts(original_image, fake_image, display=True)
    
    heatmap = create_illumination_heatmap(original_image, fake_image)

    if heatmap.ndim == 3:
        heatmap = heatmap[:,:,1]   
    

    texture_Features =  extract_texture_features(heatmap, prefix='illumination_')

    statistical_features = extract_statistical_features(heatmap, prefix='illumination_')
    
    pass

