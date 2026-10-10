import cv2
import numpy as np
from matplotlib import pyplot as plt
from ImageFeatures import extract_texture_features, extract_statistical_features
from VideoProcessing import get_frame_from_video
 
def calculate_blockwise_differences(image1, image2, block_size):
    # Convert the images to grayscale
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # Calculate the dimensions of the images
    h, w = gray1.shape

    # Calculate the number of blocks in each dimension
    num_blocks_h = h // block_size[0]
    num_blocks_w = w // block_size[1]

    # Initialize an array to store block-wise differences
    blockwise_diff = np.zeros((num_blocks_h, num_blocks_w))

    # Loop over each block and calculate the absolute difference
    for i in range(num_blocks_h):
        for j in range(num_blocks_w):
            block1 = gray1[i * block_size[0]:(i + 1) * block_size[0], j * block_size[1]:(j + 1) * block_size[1]]
            block2 = gray2[i * block_size[0]:(i + 1) * block_size[0], j * block_size[1]:(j + 1) * block_size[1]]
            
            blockwise_diff[i, j] = np.abs(np.mean(block1) - np.mean(block2))

    return blockwise_diff

def detect_compression_artifacts_blockwise(original_image_path, compressed_image_path, block_size=(8, 8), display=True):
    # Read the images

    if isinstance(original_image_path,str):
        original_image = cv2.imread(original_image_path)
        compressed_image = cv2.imread(compressed_image_path)
    else:
        original_image = original_image_path
        compressed_image = compressed_image_path

    # Resize images to a common size
    common_size = (min(original_image.shape[1], compressed_image.shape[1]),
                   min(original_image.shape[0], compressed_image.shape[0]))
    original_resized = cv2.resize(original_image, common_size, interpolation=cv2.INTER_AREA)
    compressed_resized = cv2.resize(compressed_image, common_size, interpolation=cv2.INTER_AREA)

    # Calculate block-wise differences
    blockwise_diff = calculate_blockwise_differences(original_resized, compressed_resized, block_size)

    # Display the images and the block-wise differences
    if display:
        plt.figure(figsize=(14, 5))

        plt.subplot(1, 3, 1)
        plt.imshow(cv2.cvtColor(original_resized, cv2.COLOR_BGR2RGB))
        plt.title('Original Image')
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.imshow(cv2.cvtColor(compressed_resized, cv2.COLOR_BGR2RGB))
        plt.title('Compressed Image')
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.imshow(blockwise_diff, cmap='viridis', vmin=0, vmax=np.max(blockwise_diff))
        plt.title('Block-wise Differences')
        plt.colorbar()
        plt.axis('off')

        plt.show(block=True)

    return blockwise_diff




if __name__ == '__main__':
    original_image_path = 'orig_cropped_face.png'
    compressed_image_path = 'fake_cropped_face.png'
    dataset_path = 'D:/datasets/dfdc_train_part_0/'
    OriginalVideo = "aayrffkzxn.mp4"
    FakeVideo = "abhggqdift.mp4"

    # Specify the block size for SSIM calculation
    block_size = (4, 4)

    # Visualize block-wise compression artifacts
    #blockwise_diff = detect_compression_artifacts_blockwise(original_image_path, compressed_image_path, block_size=block_size)

    original_image = get_frame_from_video(dataset_path + OriginalVideo)

    fake_image = get_frame_from_video(dataset_path + FakeVideo)
    Plot = True
    blockwise_diff = detect_compression_artifacts_blockwise(original_image, fake_image, block_size=block_size,display=Plot)

    texture_Features =  extract_texture_features(blockwise_diff, prefix='compression_')

    statistical_features = extract_statistical_features(blockwise_diff, prefix='compression_')
    
    pass

