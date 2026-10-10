import math
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa
import matplotlib.pyplot as plt
import csv
from sklearn.model_selection import train_test_split

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


from itertools import product

# Learning Rate (α): from 0.025 to 0.0002, focused around 1e-3
learning_rates = [2.5e-2, 1.25e-2, 5e-3, 4e-3, 3e-3, 2e-3, 1e-3, 5e-4, 3e-4, 2e-4]

# Weight Decay (λ): from 0 (no decay) to 0.003, concentrated in 1e-4 to 1e-3
weight_decays = [0, 1e-6, 1e-5, 3e-5, 1e-4, 3e-4, 5e-4, 7.5e-4, 1e-3, 3e-3]

# Dropout Rate (d): from 0 to 0.45 in 0.05 increments
dropout_rates = [i / 100 for i in range(0, 50, 5) ]  # [0.0, 0.05, 0.10, ..., 0.45]

# Batch Size (b): from 16 to 80
batch_sizes = [16, 24, 32, 40, 48, 50, 56, 64, 72, 80]

# Number of Epochs (E): 50 to 150, 
epoch_range = list(range(50, 151, 10))  # [50, 60, ..., 150]


# Keep track of the best model and accuracy
best_accuracy = 0.0
best_params = {}

# Loop over all combinations
for lr, wd, dr, bs in product(learning_rates, weight_decays, dropout_rates, batch_sizes):
    print(f"Running with lr={lr}, wdecay={wd}, dr={dr}, batch_size={bs}")

    # Create a new model for each run
    model = create_AWFT_model(
        embedding_dims=Embedded_Diamentions,
        num_DNN_blks=AWFT_Blocks,
        WML_Hidden_Factors=AWFT_Hidden_Layers,
        dropout_rate=dr,
    )

    # Train the model
    history = Run_Weighted_Optimizer(
        AWFT_model=model,
        training_data=training_csv,
        testing_data=testing_csv,
        num_epochs=num_epochs,
        learning_rate=lr,
        weight_decay=wd,
        batch_size=bs,
        validation=True
    )

    # Get final validation accuracy
    _, val_accuracy = model.evaluate(generate_dataset_from_file(testing_csv, bs), verbose=0)

    # Track the best
    if val_accuracy > best_accuracy:
        best_accuracy = val_accuracy
        best_params = {'lr': lr, 'wdecay': wd, 'dr': dr, 'batch_size': bs}

print("\nBest Hyperparameters:")
print(best_params)
print(f"Best Validation Accuracy: {best_accuracy:.4f}")
