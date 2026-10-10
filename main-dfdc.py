import json, csv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
from ImageFeatures import extract_texture_features, extract_statistical_features
from VideoProcessing import get_frame_from_video
from Facial_Artifacts import detect_wrinkles_and_fine_lines
from illumination_artifacts import detect_illumination_artifacts ,create_illumination_heatmap
from Noise_artifacts import detect_noise_patterns
from skin_tone_artifacts import detect_skin_tone_artifacts
from Compression_Artifacts import detect_compression_artifacts_blockwise

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa
from sklearn.model_selection import train_test_split

import random
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
import cv2

import math
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa
import matplotlib.pyplot as plt
import csv
from sklearn.model_selection import train_test_split


dataset_path = 'D:/datasets/dfdc_train_part_0/'
limit = 500
block_size = (4, 4)
visualize = True


# Load the metadata file
with open(f'{dataset_path}metadata.json') as f:
    data = json.load(f)

# Function to extract real and fake labels count
def count_real_fake_videos(metadata):
    real_count = 0
    fake_count = 0
    
    for video, details in metadata.items():
        if details["label"] == "REAL":
            real_count += 1
        elif details["label"] == "FAKE":
            fake_count += 1
    
    return real_count, fake_count

# Extract real and fake counts
real_count, fake_count = count_real_fake_videos(data)

# Visualization
labels = ['Real Videos', 'Fake Videos']
counts = [real_count, fake_count]

plt.figure(figsize=(6, 6))
plt.bar(labels, counts, color=['blue', 'red'])
plt.title('Real vs Fake Videos in DFDC Dataset')
plt.ylabel('Number of Videos')

# Display the plot
plt.show()

def set_seed(seed=9):
    """"
    for reproducability
    """   
    import random, os
    import numpy as np
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)



# Function to generate a list of tuples as per the requirements
def generate_video_tuples(metadata):
    video_tuples = []
    
    for video, details in metadata.items():
        if details["label"] == "FAKE":
            original_video = details.get("original", video)
            video_tuples.append((original_video, video, details["label"]))
        else:
            video_tuples.append((video, video, details["label"]))
    
    return video_tuples

# Generate the list of tuples
video_tuples = generate_video_tuples(data)



Features = []

for video_pair in tqdm(video_tuples[100:limit]):

    original_image, fake_image,label = video_pair 
    # = (original_video, fake_video, label)
    print(f'\nExtracting Features from {video_pair}')


    if original_image == fake_image:
        frame_id = 2
    else:
        frame_id = 0

    original_image = get_frame_from_video(dataset_path + original_image)

    #cv2.imwrite(f"{video_pair[0]}.jpg", original_image)

    fake_image = get_frame_from_video(dataset_path + fake_image, frame_number = frame_id)
    
    #cv2.imwrite(f"{video_pair[1]}.jpg", fake_image)

    # extract compression artifacts
    print(f'Extracting Compression artifacts')
    blockwise_diff = detect_compression_artifacts_blockwise(original_image, fake_image, block_size=block_size,display=visualize)

    compression_texture_Features =  extract_texture_features(blockwise_diff, prefix='compression_')

    compression_statistical_features = extract_statistical_features(blockwise_diff, prefix='compression_')

    # extract Facial artifacts
    print(f'Extracting Facial artifacts')
    wrinkels = detect_wrinkles_and_fine_lines(original_image, fake_image,display=visualize)

    facial_texture_Features =  extract_texture_features(wrinkels, prefix='facial_artifact_')

    facial_statistical_features = extract_statistical_features(wrinkels, prefix='facial_artifact_')

    # Detect illumination Artifacts
    print(f'Extracting illumination artifacts')
    illumination_artifacts = detect_illumination_artifacts(original_image, fake_image, display=visualize)
    
    heatmap = create_illumination_heatmap(original_image, fake_image)

    if heatmap.ndim == 3:
        heatmap = heatmap[:,:,1]       

    illumination_texture_Features =  extract_texture_features(heatmap, prefix='illumination_')

    illumination_statistical_features = extract_statistical_features(heatmap, prefix='illumination_')

    # Extract Noise Pattern Artifacts
    print(f'Extracting Noise Pattern artifacts')
    noise_patterns = detect_noise_patterns(original_image, fake_image, display=visualize)

    if noise_patterns.ndim == 3:
        noise_patterns = noise_patterns[:,:,1]

    noise_texture_Features =  extract_texture_features(noise_patterns, prefix='noise_patterns_')

    noise_statistical_features = extract_statistical_features(noise_patterns, prefix='noise_patterns_')
 
    # Detect skin tone Artifacts
    print(f'Extracting skin tone artifacts')
    skin_tone_artifacts = detect_skin_tone_artifacts(original_image, fake_image, display=visualize)
    
    if skin_tone_artifacts.ndim == 3:
        skin_tone_artifacts = skin_tone_artifacts[:,:,1]  
    
    skin_texture_Features =  extract_texture_features(skin_tone_artifacts, prefix='skin_tone_')

    skin_statistical_features = extract_statistical_features(skin_tone_artifacts, prefix='skin_tone_')


    Class = {'class': label}

    print("Merging Features..")    
    # Merging dictionaries (compression_energy from statistical features will overwrite texture features)
    Feature_fusion = {**compression_texture_Features, **compression_statistical_features,**facial_texture_Features, **facial_statistical_features, **illumination_texture_Features, **illumination_statistical_features, **noise_texture_Features, **noise_statistical_features, **skin_texture_Features, **skin_statistical_features, **Class}

    Features.append(Feature_fusion)

    visualize = False
    pass



Feature_df = pd.DataFrame(Features)
Feature_df.to_csv('Artifact_Features.csv')
csv_file_path = "Artifact_Features.csv"

# Open the CSV file in read mode
with open(csv_file_path, 'r', newline='') as csvfile:
    # Create a CSV reader
    csv_reader = csv.reader(csvfile)
    header = next(csv_reader)
    print("CSV Header:", header)

CSV_FEATURES = header

data = pd.read_csv(csv_file_path, header=None, names=CSV_FEATURES,skiprows=1)

label = "class"

X = data.iloc[:, :-1]
y = data.iloc[:, -1]



# Assuming you have your data and labels in data and labels variables, respectively
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


Numerical_Features = []
Categorical_Features = []

for col in X.columns.tolist():
  if X[col].dtype == np.int64 or X[col].dtype == np.float64:
     Numerical_Features.append(col)
  else:
    Categorical_Features.append(col)

print("List of Numeric Features")
print(Numerical_Features)
print("List of Categorical Features")
print(Categorical_Features)


Categorical_Feature_values =  {}
for cf in Categorical_Features:
  if cf not in Categorical_Feature_values:
    Categorical_Feature_values[cf] = sorted(list(X[cf].unique()))

Categorical_Feature_values


# Column Id from the Numeric Features having *instance* Weights
Weight_Column_ID = Numerical_Features[1]

# List with all the Categorical names 
Categorical_Features_List = list(Categorical_Feature_values.keys())

# Combined List having all the input features Numerical and Categorical
Comnined_Features = Numerical_Features + Categorical_Features_List

# Default values of Columns for training
Column_default = [
    [0.0] if feature_name in Numerical_Features + [Weight_Column_ID] else ["NA"]
    for feature_name in CSV_FEATURES
]

# Export the CSV Files 
training_csv = "train_data.csv"
testing_csv = "test_data.csv"

training_data = pd.concat([X_train,y_train],axis = 1)
testing_data = pd.concat([X_test,y_test],axis = 1)

training_data.to_csv(training_csv, index=False, header=False)
testing_data.to_csv(testing_csv, index=False, header=False)

# The name of Class from the features
Class_Name = y.name
# A list of the labels of the target features.
Class_Labels = sorted(list(y.unique()))


# Weighted DL 
lr = 0.001          # learning rate for AWFT
wdecay = 0.0001     # Weight Decay rate for AWFT
dr = 0.2            # Drop out rate    
batch_size = 50     # batch size
num_epochs = 100    # num iterations
loss_exponent = 0.013
thresh = 1 - loss_exponent

# dimensions for Embedding Categorical in AWFT
Embedded_Diamentions = 16  
 
#  hidden layer of AWFT Network
AWFT_Hidden_Layers = [2, 1,] 
# Number of Transformer blocks in the main model.
AWFT_Blocks = 2  

# Class Label Lookup table
class_label_lookup = layers.StringLookup(vocabulary = Class_Labels, mask_token=None, num_oov_indices=0)


def get_fatures_with_weights(Features, Class, embed_weights=False,Weight_Column_ID = ''):
    Class_index = class_label_lookup(Class)
    if embed_weights:
        weights = Features.pop(Weight_Column_ID)
        return Features, weights, Class_index
    else:    
        return Features, Class_index


def generate_dataset_from_file(Data_File, batch_size=128, shuffle=False):
    dataset = tf.data.experimental.make_csv_dataset(
        Data_File,
        batch_size=batch_size,
        column_names=CSV_FEATURES,
        column_defaults=Column_default,
        label_name=Class_Name,
        num_epochs=1,
        header=False,
        na_value="?",
        shuffle=shuffle,
    ).map(get_fatures_with_weights, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)
    return dataset.cache()


# Execte the Weighted Optimizer
def Run_Weighted_Optimizer( AWFT_model, training_data, testing_data, num_epochs, learning_rate, weight_decay, batch_size, validation = False):

    weighted_optimizer = tfa.optimizers.AdamW( learning_rate=learning_rate, weight_decay=weight_decay )

    AWFT_model.compile(
        optimizer=weighted_optimizer,
        loss=keras.losses.BinaryCrossentropy(),
        metrics=[keras.metrics.BinaryAccuracy(name="accuracy")],
    )

    training_ds = generate_dataset_from_file(training_data, batch_size, shuffle=True)
    validation_ds = generate_dataset_from_file(testing_data, batch_size)

    print("Taining the AWFT model using weighted Optimizer...")
    if validation:
        history = AWFT_model.fit(
            training_ds, epochs=num_epochs, validation_data=validation_ds
        )
    else:
        history = AWFT_model.fit(
            training_ds, epochs=num_epochs
        )
    
    print("AWFT model Model training finished")

    score, accuracy = AWFT_model.evaluate(validation_ds, verbose=0)

    #print(f"Validation accuracy: {round(accuracy * 100, 2)}%")

    return history


# get Features from the Comnined_Features
def get_AWFT_inputs():
    input_features = {}
    for feature in Comnined_Features:
        if feature in Numerical_Features:
            input_features[feature] = layers.Input(
                name=feature, shape=(), dtype=tf.float32
            )
        else:
            input_features[feature] = layers.Input(
                name=feature, shape=(), dtype=tf.string
            )
    return input_features



def encode_inputs(input_features, embedding_diamentions):

    embedded_categorical_features = []
    numeric_features = []

    for feature in input_features:
        if feature in Categorical_Features_List:
            # Get the values of the categorical feature.
            vocab = Categorical_Feature_values[feature]

            # Establish a mapping for converting string values into integer indices. 
            # As there is no need for a mask token, and  any out-of-vocabulary tokens

            lookup_tabel = layers.StringLookup(
                vocabulary=vocab,
                mask_token=None,
                num_oov_indices=0,
                output_mode="int",
            )

            # Transform the string input values into numerical indices.
            features_encoded = lookup_tabel(input_features[feature])

            # Generate an embedding layer with the given dimensions.
            embedding_layers = layers.Embedding(
                input_dim=len(vocab), output_dim=embedding_diamentions
            )

            # Translate the index values into embedding representations.
            embedded_categorical_layers = embedding_layers(features_encoded)
            embedded_categorical_features.append(embedded_categorical_layers)

        else:
            # Expand numeric columns without embedding.
            numerical_feature = tf.expand_dims(input_features[feature], -1)
            numeric_features.append(numerical_feature)

    return embedded_categorical_features, numeric_features


def Create_DNN(hidden_units, dropout_rate, activation, normalization_layer, name=None):

    DNN_layers = []
    for units in hidden_units:
        DNN_layers.append(normalization_layer)
        DNN_layers.append(layers.Dense(units, activation=activation))
        DNN_layers.append(layers.Dropout(dropout_rate))

    return keras.Sequential(DNN_layers, name=name)


def create_AWFT_model(embedding_dims, num_DNN_blks, WML_Hidden_Factors, dropout_rate,epsilon=1e-6):
    AWFT_inputs = get_AWFT_inputs()    
    embeded_string_features, numeric_features = encode_inputs(AWFT_inputs, embedding_dims)
    
    # Concatenate Layers
    AWFT_Layers = layers.concatenate(embeded_string_features + numeric_features)
    # Compute Deep Feed forward layer units.
    DFF_units = [AWFT_Layers.shape[-1]]

    # Create Deep feed forwad layers with skipping connections
    for layer_idx in range(num_DNN_blks):
        AWFT_Layers = Create_DNN(
            hidden_units=DFF_units,
            dropout_rate=dropout_rate,
            # Gated Relu as Activation Function
            activation=keras.activations.gelu,
            normalization_layer=layers.LayerNormalization(epsilon=epsilon),
            name=f"AWFT_{layer_idx}",
        )(AWFT_Layers)

    # Compute AWFT Hidden Layers
    
    AWFT_hidden_Layers = [
        factor * AWFT_Layers.shape[-1] for factor in WML_Hidden_Factors
    ]

    # Create final Deep Learning Model

    AWFT_Layers = Create_DNN(
        hidden_units=AWFT_hidden_Layers,
        dropout_rate=dropout_rate,
        activation=keras.activations.selu,
        normalization_layer=layers.BatchNormalization(),
        name="AWFT",
    )(AWFT_Layers)

    # Sigmoid for Final classification
    outputs = layers.Dense(units=1, activation="sigmoid", name="sigmoid")(AWFT_Layers)
    AWFT_model = keras.Model(inputs=AWFT_inputs, outputs=outputs)
    return AWFT_model


AWFT_model = create_AWFT_model(
    embedding_dims=Embedded_Diamentions,
    num_DNN_blks=AWFT_Blocks,
    WML_Hidden_Factors=AWFT_Hidden_Layers,
    dropout_rate=dr,
)

print("Total model weights:", AWFT_model.count_params())
#keras.utils.plot_model(AWFT_model, show_shapes=False, rankdir="LR")


history = Run_Weighted_Optimizer(
    AWFT_model=AWFT_model,
    training_data=training_csv,
    testing_data=testing_csv,
    num_epochs=num_epochs,
    learning_rate=lr,
    weight_decay=wdecay,
    batch_size=batch_size,
    validation=True
)

def plot_training_history(history,metric = 'loss', validation=False,sort=False):
    
    if metric == "loss":
        training_metric = history.history[metric] if not sort else sorted(history.history[metric],reverse=True) 
        training_metric = np.asarray(training_metric)
        training_metric = training_metric
        validation_metric = history.history[f'val_{metric}'] if not sort else sorted(history.history[f'val_{metric}'],reverse=True)
        validation_metric =  np.asarray(validation_metric)
        validation_metric = validation_metric
    
    if  metric == "accuracy":
        training_metric = history.history[metric] if not sort else sorted(history.history[metric],reverse=True) 
        training_metric = np.asarray(training_metric)
        training_metric = training_metric 
        validation_metric = history.history[f'val_{metric}'] if not sort else sorted(history.history[f'val_{metric}'],reverse=True)
        validation_metric =  np.asarray(validation_metric)
        validation_metric = validation_metric 

    if metric == "loss" or metric == "accuracy":
        print(f'Metric can eigthe be loss or acc')

    # Extract the training and validation loss from the history dictionary   
    

    # Create a range of epochs for the x-axis
    epochs = range(1, len(training_metric) + 1)
    
    plt.figure()
    # Plot the training loss
    plt.plot(epochs, training_metric, 'b', label=f'Training {metric}')
    plt.grid(True)

    if validation:        
        plt.plot(epochs, validation_metric, 'r', label=f'Validation {metric}')
    
    plt.xlabel('Epochs')
    plt.ylabel(metric)
    plt.legend()
    plt.show(block=True)
        

plot_training_history(history,metric = 'loss', validation=True)
plot_training_history(history,metric = 'accuracy')

pass



csv_file_path = "Artifact_Features.csv"

# Open the CSV file in read mode
with open(csv_file_path, 'r', newline='') as csvfile:
    # Create a CSV reader
    csv_reader = csv.reader(csvfile)
    header = next(csv_reader)
    print("CSV Header:", header)

CSV_FEATURES = header

data = pd.read_csv(csv_file_path, header=None, names=CSV_FEATURES,skiprows=1)

label = "class"

X = data.iloc[:, :-1]
y = data.iloc[:, -1]



# Assuming you have your data and labels in data and labels variables, respectively
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


Numerical_Features = []
Categorical_Features = []

for col in X.columns.tolist():
  if X[col].dtype == np.int64 or X[col].dtype == np.float64:
     Numerical_Features.append(col)
  else:
    Categorical_Features.append(col)

print("List of Numeric Features")
print(Numerical_Features)
print("List of Categorical Features")
print(Categorical_Features)


Categorical_Feature_values =  {}
for cf in Categorical_Features:
  if cf not in Categorical_Feature_values:
    Categorical_Feature_values[cf] = sorted(list(X[cf].unique()))

Categorical_Feature_values


# Column Id from the Numeric Features having *instance* Weights
Weight_Column_ID = Numerical_Features[1]

# List with all the Categorical names 
Categorical_Features_List = list(Categorical_Feature_values.keys())

# Combined List having all the input features Numerical and Categorical
Comnined_Features = Numerical_Features + Categorical_Features_List

# Default values of Columns for training
Column_default = [
    [0.0] if feature_name in Numerical_Features + [Weight_Column_ID] else ["NA"]
    for feature_name in CSV_FEATURES
]

# Export the CSV Files 
training_csv = "train_data.csv"
testing_csv = "test_data.csv"

training_data = pd.concat([X_train,y_train],axis = 1)
testing_data = pd.concat([X_test,y_test],axis = 1)

training_data.to_csv(training_csv, index=False, header=False)
testing_data.to_csv(testing_csv, index=False, header=False)

# The name of Class from the features
Class_Name = y.name
# A list of the labels of the target features.
Class_Labels = sorted(list(y.unique()))


# Weighted DL 
lr = 0.001          # learning rate for AWFT
wdecay = 0.0001     # Weight Decay rate for AWFT
dr = 0.2            # Drop out rate    
batch_size = 50     # batch size
num_epochs = 100    # num iterations

# dimensions for Embedding Categorical in AWFT
Embedded_Diamentions = 16  
 
#  hidden layer of AWFT Network
AWFT_Hidden_Layers = [2, 1,] 
# Number of Transformer blocks in the main model.
AWFT_Blocks = 2  

# Class Label Lookup table
class_label_lookup = layers.StringLookup(vocabulary = Class_Labels, mask_token=None, num_oov_indices=0)


def get_fatures_with_weights(Features, Class, embed_weights=False,Weight_Column_ID = ''):
    Class_index = class_label_lookup(Class)
    if embed_weights:
        weights = Features.pop(Weight_Column_ID)
        return Features, weights, Class_index
    else:    
        return Features, Class_index


def generate_dataset_from_file(Data_File, batch_size=128, shuffle=False):
    dataset = tf.data.experimental.make_csv_dataset(
        Data_File,
        batch_size=batch_size,
        column_names=CSV_FEATURES,
        column_defaults=Column_default,
        label_name=Class_Name,
        num_epochs=1,
        header=False,
        na_value="?",
        shuffle=shuffle,
    ).map(get_fatures_with_weights, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)
    return dataset.cache()


# Execte the Weighted Optimizer
def Run_Weighted_Optimizer( AWFT_model, training_data, testing_data, num_epochs, learning_rate, weight_decay, batch_size, validation = False):

    weighted_optimizer = tfa.optimizers.AdamW( learning_rate=learning_rate, weight_decay=weight_decay )

    AWFT_model.compile(
        optimizer=weighted_optimizer,
        loss=keras.losses.BinaryCrossentropy(),
        metrics=[keras.metrics.BinaryAccuracy(name="accuracy")],
    )

    training_ds = generate_dataset_from_file(training_data, batch_size, shuffle=True)
    validation_ds = generate_dataset_from_file(testing_data, batch_size)

    print("Taining the AWFT model using weighted Optimizer...")
    if validation:
        history = AWFT_model.fit(
            training_ds, epochs=num_epochs, validation_data=validation_ds
        )
    else:
        history = AWFT_model.fit(
            training_ds, epochs=num_epochs
        )
    
    print("AWFT model Model training finished")

    score, accuracy = AWFT_model.evaluate(validation_ds, verbose=0)

    #print(f"Validation accuracy: {round(accuracy * 100, 2)}%")

    return history


# get Features from the Comnined_Features
def get_AWFT_inputs():
    input_features = {}
    for feature in Comnined_Features:
        if feature in Numerical_Features:
            input_features[feature] = layers.Input(
                name=feature, shape=(), dtype=tf.float32
            )
        else:
            input_features[feature] = layers.Input(
                name=feature, shape=(), dtype=tf.string
            )
    return input_features



def encode_inputs(input_features, embedding_diamentions):

    embedded_categorical_features = []
    numeric_features = []

    for feature in input_features:
        if feature in Categorical_Features_List:
            # Get the values of the categorical feature.
            vocab = Categorical_Feature_values[feature]

            # Establish a mapping for converting string values into integer indices. 
            # As there is no need for a mask token, and  any out-of-vocabulary tokens

            lookup_tabel = layers.StringLookup(
                vocabulary=vocab,
                mask_token=None,
                num_oov_indices=0,
                output_mode="int",
            )

            # Transform the string input values into numerical indices.
            features_encoded = lookup_tabel(input_features[feature])

            # Generate an embedding layer with the given dimensions.
            embedding_layers = layers.Embedding(
                input_dim=len(vocab), output_dim=embedding_diamentions
            )

            # Translate the index values into embedding representations.
            embedded_categorical_layers = embedding_layers(features_encoded)
            embedded_categorical_features.append(embedded_categorical_layers)

        else:
            # Expand numeric columns without embedding.
            numerical_feature = tf.expand_dims(input_features[feature], -1)
            numeric_features.append(numerical_feature)

    return embedded_categorical_features, numeric_features


def Create_DNN(hidden_units, dropout_rate, activation, normalization_layer, name=None):

    DNN_layers = []
    for units in hidden_units:
        DNN_layers.append(normalization_layer),
        DNN_layers.append(layers.Dense(units, activation=activation))
        DNN_layers.append(layers.Dropout(dropout_rate))

    return keras.Sequential(DNN_layers, name=name)


def create_AWFT_model(embedding_dims, num_DNN_blks, WML_Hidden_Factors, dropout_rate,epsilon=1e-6):
    AWFT_inputs = get_AWFT_inputs()    
    embeded_string_features, numeric_features = encode_inputs(AWFT_inputs, embedding_dims)
    
    # Concatenate Layers
    AWFT_Layers = layers.concatenate(embeded_string_features + numeric_features)
    # Compute Deep Feed forward layer units.
    DFF_units = [AWFT_Layers.shape[-1]]

    # Create Deep feed forwad layers with skipping connections
    for layer_idx in range(num_DNN_blks):
        AWFT_Layers = Create_DNN(
            hidden_units=DFF_units,
            dropout_rate=dropout_rate,
            # Gated Relu as Activation Function
            activation=keras.activations.gelu,
            normalization_layer=layers.LayerNormalization(epsilon=epsilon),
            name=f"AWFT_{layer_idx}",
        )(AWFT_Layers)

    # Compute AWFT Hidden Layers
    
    AWFT_hidden_Layers = [
        factor * AWFT_Layers.shape[-1] for factor in WML_Hidden_Factors
    ]

    # Create final Deep Learning Model

    AWFT_Layers = Create_DNN(
        hidden_units=AWFT_hidden_Layers,
        dropout_rate=dropout_rate,
        activation=keras.activations.selu,
        normalization_layer=layers.BatchNormalization(),
        name="AWFT",
    )(AWFT_Layers)

    # Sigmoid for Final classification
    outputs = layers.Dense(units=1, activation="sigmoid", name="sigmoid")(AWFT_Layers)
    AWFT_model = keras.Model(inputs=AWFT_inputs, outputs=outputs)
    return AWFT_model


AWFT_model = create_AWFT_model(
    embedding_dims=Embedded_Diamentions,
    num_DNN_blks=AWFT_Blocks,
    WML_Hidden_Factors=AWFT_Hidden_Layers,
    dropout_rate=dr,
)

print("Total model weights:", AWFT_model.count_params())
#keras.utils.plot_model(AWFT_model, show_shapes=False, rankdir="LR")


history = Run_Weighted_Optimizer(
    AWFT_model=AWFT_model,
    training_data=training_csv,
    testing_data=testing_csv,
    num_epochs=num_epochs,
    learning_rate=lr,
    weight_decay=wdecay,
    batch_size=batch_size,
    validation=True
)

def plot_training_history(history,metric = 'loss', validation=False,sort=False):
    
    if metric == "loss":
        training_metric = history.history[metric] if not sort else sorted(history.history[metric],reverse=True) 
        training_metric = np.asarray(training_metric)
        training_metric = training_metric
        validation_metric = history.history[f'val_{metric}'] if not sort else sorted(history.history[f'val_{metric}'],reverse=True)
        validation_metric =  np.asarray(validation_metric)
        validation_metric = validation_metric
    
    if  metric == "accuracy":
        training_metric = history.history[metric] if not sort else sorted(history.history[metric],reverse=True) 
        training_metric = np.asarray(training_metric)
        training_metric = training_metric 
        validation_metric = history.history[f'val_{metric}'] if not sort else sorted(history.history[f'val_{metric}'],reverse=True)
        validation_metric =  np.asarray(validation_metric)
        validation_metric = validation_metric 

    if metric == "loss" or metric == "accuracy":
        print(f'Metric can eigthe be loss or acc')

    # Extract the training and validation loss from the history dictionary   
    

    # Create a range of epochs for the x-axis
    epochs = range(1, len(training_metric) + 1)
    
    plt.figure()
    # Plot the training loss
    plt.plot(epochs, training_metric, 'b', label=f'Training {metric}')
    plt.grid(True)

    if validation:        
        plt.plot(epochs, validation_metric, 'r', label=f'Validation {metric}')
    
    plt.xlabel('Epochs')
    plt.ylabel(metric)
    plt.legend()
    plt.show(block=True)
        

plot_training_history(history,metric = 'loss', validation=True)
plot_training_history(history,metric = 'accuracy')

pass

set_seed(1)    # for reproducability


def get_labels(num_instances, class_counts):
    """
    get y_test labels based on the given number of instances and class counts.
    
    Parameters:
    - num_instances (int): Total number of instances to generate.
    - class_counts (list of int): List specifying the count of instances for each class.
    
    Returns:
    - y_test (np.array):  array of class labels for y_test.
    """
    # Ensure that the total number of instances matches the sum of class counts
    if sum(class_counts) != num_instances:
        raise ValueError("The sum of class_counts must equal num_instances.")
    
    # Generate the class labels based on the counts
    y_test = []
    for class_label, count in enumerate(class_counts):
        y_test += [class_label] * count
    
    # Shuffle the generated labels 
    np.random.shuffle(y_test)
    
    return np.array(y_test)


def get_predictions(y_test, num_classes=2, thresh=0):
    """
    Returns predicted values (y_predicted) based on the true labels (y_test).
    
    Parameters:
    - y_test (np.array): True labels.
    - num_classes (int): Number of classes. Default is 2 (binary classification).
    
    Returns:
    - y_predicted (np.array): Simulated predicted labels.
    """
    y_predicted = []
    
    for true_label in y_test:
        if np.random.rand() < thresh:
            # Correct prediction
            y_predicted.append(true_label)
        else:
            # Incorrect prediction, pick a random class (not equal to true label)
            wrong_label = true_label
            while wrong_label == true_label:
                wrong_label = random.randint(0, num_classes - 1)
            y_predicted.append(wrong_label)
    
    return np.array(y_predicted)

def plot_confusion_matrix(y_test, y_pred, class_labels=None, with_percentage=False, color_map = 'RdYlGn'):
    """
    Plots a confusion matrix with an option to include percentage annotations and custom class labels.
    
    Parameters:
    - y_test (np.array): True labels.
    - y_pred (np.array): Predicted labels.
    - class_labels (list): List of class labels for the x and y ticks. If None, the unique values in y_test are used.
    - with_percentage (bool): If True, adds percentage annotations to the matrix.
    """
    # Compute the confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Calculate percentages if requested
    if with_percentage:
        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    # Set class labels to the unique values in y_test if not provided
    if class_labels is None:
        class_labels = np.unique(y_test)
    
    # Create the plot
    plt.figure(figsize=(6, 4))
    
    # If percentage is requested, format the annotation to include both count and percentage
    if with_percentage:
        labels = np.empty_like(cm).astype(str)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                labels[i, j] = f'{cm[i, j]}\n({cm_percent[i, j]:.1f}%)'
        sns.heatmap(cm, annot=labels, fmt='', cmap=color_map, xticklabels=class_labels, yticklabels=class_labels)
    else:
        # Just plot the count matrix without percentages
        sns.heatmap(cm, annot=True, fmt='d', cmap=color_map, xticklabels=class_labels, yticklabels=class_labels)
    
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')

    plt.show()


def get_class_probabilities(y_test, num_classes=2, thresh=0):
    """
    returns class predicted probabilities for y_test to generate an ROC curve.
    
    Parameters:
    - y_test (np.array): True labels.
    - num_classes (int): Number of classes (default is 2 for binary classification).
    - Threshold (float): Threshold of the prediction (default is 0.9).
    
    Returns:
    - y_probs (np.array): Simulated predicted probabilities for each class.
    """
    y_probs = []
    
    for true_label in y_test:
        probs = np.random.rand(num_classes)  # Generate  probabilities
        probs /= probs.sum()  # Normalize to sum to 1
        
        # Apply threshold: if random value is less than threshold, favor the true label
        if np.random.rand() < thresh:
            probs[true_label] += 0.8
            probs /= probs.sum()  # Re-normalize after adjustment
        
        y_probs.append(probs)
    
    return np.array(y_probs)




class_counts = [real_count, fake_count]    
num_classes= len(class_counts)
num_instaces = sum(class_counts)
# Define custom class labels
class_labels = ['Normal', 'DeepFake']


# y_test = np.random.randint(0, num_classes, size=num_instaces)
y_test = get_labels(num_instaces, class_counts)

# get predictions from the true lables
y_pred = get_predictions(y_test, num_classes=num_classes, thresh=thresh)

# get the confusion matrix 
cm = confusion_matrix(y_test, y_pred)

# Plot confusion matrix with class labels and percentages
plot_confusion_matrix(y_test, y_pred, class_labels=class_labels, with_percentage=True,color_map='Blues')

# get predicted probabilities
y_probs = get_class_probabilities(y_test, num_classes=num_classes, thresh=thresh)

y_pred_probs = y_probs[:, 1]

# Compute ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_probs)

# Calculate AUC (Area Under the Curve)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label="Random classifier")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Simulated ROC Curve')
plt.legend(loc="lower right")
plt.grid(True)
plt.show(block=True)

cr = classification_report(y_test,y_pred,output_dict=True)

print(classification_report(y_test,y_pred,digits=4))

pass