import cv2
import numpy as np
from matplotlib import pyplot as plt
from VideoProcessing import get_frame_from_video
from ImageFeatures import extract_texture_features, extract_statistical_features

def detect_wrinkles_and_fine_lines(image_path1, image_path2, display=True):

    if isinstance(image_path1,str):
        # Read the images
        image1 = cv2.imread(image_path1, cv2.IMREAD_GRAYSCALE)
        image2 = cv2.imread(image_path2, cv2.IMREAD_GRAYSCALE)
    else:
        image1 = image_path1
        image2 = image_path2

    # Resize images to a common size
    common_size = (min(image1.shape[1], image2.shape[1]), min(image1.shape[0], image2.shape[0]))
    image1_resized = cv2.resize(image1, common_size, interpolation=cv2.INTER_AREA)
    image2_resized = cv2.resize(image2, common_size, interpolation=cv2.INTER_AREA)

    # Apply Canny edge detection
    edges1 = cv2.Canny(image1_resized, 10, 50)
    edges2 = cv2.Canny(image2_resized, 10, 50)

    # Display the images and the Artifacts in wrinkles and fine lines
    if display:
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 3, 1)
        plt.imshow(image1_resized, cmap='gray')
        plt.title('Original Image')
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.imshow(image2_resized, cmap='gray')
        plt.title('DeepFake Image')
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.imshow(edges1 - edges2, cmap='gray')
        plt.title('Wrinkles and Fine Lines Artifacts')
        plt.axis('off')

        plt.show(block=True)
    
    return edges1 - edges2



if __name__ == '__main__':
    image_path1 = 'fake_cropped_face.png'
    image_path2 = 'orig_cropped_face.png'

    # Visualize Artifacts in wrinkles and fine lines
    wrinkels = detect_wrinkles_and_fine_lines(image_path1, image_path2)


    dataset_path = 'E:/Projects/datasets/dfdc_train_part_0/'
    OriginalVideo = "wfzjxzhdkj.mp4"
    FakeVideo = "slhinthqcq.mp4"

    original_image = get_frame_from_video(dataset_path + OriginalVideo)
    fake_image = get_frame_from_video(dataset_path + FakeVideo)    

    wrinkels = detect_wrinkles_and_fine_lines(original_image, fake_image,display=False)

    texture_Features =  extract_texture_features(wrinkels, prefix='facial_artifact_')

    statistical_features = extract_statistical_features(wrinkels, prefix='facial_artifact_')
    
    pass

