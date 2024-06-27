import os
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetV2B2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split
# create tensorboard callback (import from helper function)
from helper_functions import create_tensorboard_callback

def load_and_preprocess_data(csv_path, image_folder, img_size, batch_size):
    # Read CSV
    try:
        df_merged = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return None, None, None

    if df_merged.empty or 'filename' not in df_merged.columns:
        print("CSV file is empty or does not contain the 'filename' column")
        return None, None, None

    # Shuffle the DataFrame and limit it to the first 2000 rows
    df_merged = df_merged.sample(frac=1, random_state=42).head(2000)
    df_merged['filename'] = df_merged['filename'].apply(lambda x: os.path.join(image_folder, x + ".png"))

    # Split dataset
    train_df, test_df = train_test_split(df_merged, test_size=0.2, random_state=42)
    test_df, val_df = train_test_split(test_df, test_size=0.5, random_state=42)

    # Data generators
    train_datagen = ImageDataGenerator(
        rescale=1.0/255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    validation_datagen = ImageDataGenerator(rescale=1.0/255)

    try:
        train_generator = train_datagen.flow_from_dataframe(
            dataframe=train_df,
            x_col='filename',
            y_col=train_df.columns[1:],
            target_size=(img_size, img_size),
            batch_size=batch_size,
            class_mode='raw'
        )
        validation_generator = validation_datagen.flow_from_dataframe(
            dataframe=val_df,
            x_col='filename',
            y_col=val_df.columns[1:],
            target_size=(img_size, img_size),
            batch_size=batch_size,
            class_mode='raw'
        )
        test_generator = validation_datagen.flow_from_dataframe(
            dataframe=test_df,
            x_col='filename',
            y_col=test_df.columns[1:],
            target_size=(img_size, img_size),
            batch_size=batch_size,
            class_mode='raw',
            shuffle=False
        )
    except Exception as e:
        print(f"Error creating data generators: {e}")
        return None, None, None

    return train_generator, validation_generator, test_generator

def create_model(img_size, num_classes, base_weights_path):
    base_model = EfficientNetV2B2(weights=base_weights_path, include_top=False, input_tensor=Input(shape=(img_size, img_size, 3)))
    base_model.trainable = False

    inputs = tf.keras.layers.Input(shape=(img_size, img_size, 3), name='Input_layer')
    x = base_model(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='sigmoid')(x)

    model = tf.keras.Model(inputs, outputs)
    return model

def train_top_layers(model, train_generator, validation_generator, num_epochs, initial_learning_rate):
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps=100000,
        decay_rate=1.0,
        staircase=True)

    model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
                  optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
                  metrics=[tf.keras.metrics.BinaryAccuracy()])

    checkpoint_path = 'runs/best_top_layers.ckpt'
    model_checkpoint = ModelCheckpoint(checkpoint_path, save_best_only=True, save_weights_only=True, verbose=0)
    checkpoint_last = ModelCheckpoint('runs/last_model_top_layers.ckpt', monitor='val_loss', save_best_only=False, save_freq='epoch')
    early_stopping = EarlyStopping(monitor='val_loss', mode='min', patience=20)
    tensorboard_callback = create_tensorboard_callback(dir_name='MA', experiment_name='ori_multilabel_laptop')

    history = model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=num_epochs,
        callbacks=[tensorboard_callback, model_checkpoint, checkpoint_last, early_stopping],
        verbose=1
    )

    return history

def train_all_layers(model, train_generator, validation_generator, num_epochs, initial_learning_rate):
    model.layers[1].trainable = True
    for layer in model.layers[1].layers[:-60]:
        layer.trainable = False

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps=100000,
        decay_rate=1.0,
        staircase=True)

    model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
                  optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
                  metrics=[tf.keras.metrics.BinaryAccuracy()])

    checkpoint_path = 'runs/best_all_layers.ckpt'
    model_checkpoint = ModelCheckpoint(checkpoint_path, save_best_only=True, save_weights_only=True, verbose=0)
    checkpoint_last = ModelCheckpoint('runs/last_model_all_layers.ckpt', monitor='val_loss', save_best_only=False, save_freq='epoch')
    early_stopping = EarlyStopping(monitor='val_loss', mode='min', patience=20)
    tensorboard_callback = create_tensorboard_callback(dir_name='MA', experiment_name='ori_finetuned_laptop')

    history = model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=num_epochs,
        callbacks=[tensorboard_callback, model_checkpoint, checkpoint_last, early_stopping],
        verbose=1
    )

    return history

def train_efficientnet_multilabel(csv_path, image_folder, img_size=640, batch_size=10, num_epochs=40):
    tf.keras.backend.clear_session()
    
    train_generator, validation_generator, test_generator = load_and_preprocess_data(csv_path, image_folder, img_size, batch_size)
    
    if train_generator is None or validation_generator is None or test_generator is None:
        return

    num_classes = 24
    base_weights_path = r"D:/Skripsweet/BISMILLAH_KELAR/efficientnetv2-b2_notop.h5"
    model = create_model(img_size, num_classes, base_weights_path)

    history_top = train_top_layers(model, train_generator, validation_generator, num_epochs // 2, initial_learning_rate=2e-4)
    model.evaluate(test_generator)
    model.save("early_model_ori.h5")

    history_all = train_all_layers(model, train_generator, validation_generator, num_epochs // 2, initial_learning_rate=2e-4)
    model.evaluate(test_generator)
    model.save('final_model_ori.h5')

    return history_top, history_all

if __name__ == "__main__":
    csv_path = 'D:/Skripsweet/BISMILLAH_KELAR/dataset/image_datasets.csv'  # Update with the actual path
    image_folder = 'D:/Skripsweet/BISMILLAH_KELAR/dataset/images'

    history_top, history_all = train_efficientnet_multilabel(csv_path, image_folder)