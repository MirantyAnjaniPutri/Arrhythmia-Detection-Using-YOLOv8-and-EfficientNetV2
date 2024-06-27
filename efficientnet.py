import os
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetV2S
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split

def train_efficientnet_multilabel(csv_path, image_folder, img_size=512, batch_size=4, num_epochs=10):
    # Enable mixed precision
    tf.keras.mixed_precision.set_global_policy('mixed_float16')

    # Clear previous sessions
    tf.keras.backend.clear_session()

    # Read CSV
    try:
        df_merged = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return

    # Check if DataFrame is empty or incorrect
    if df_merged.empty or 'filename' not in df_merged.columns:
        print("CSV file is empty or does not contain the 'filename' column")
        return

    # Shuffle the DataFrame and limit it to the first 2000 rows
    df_merged = df_merged.sample(frac=1, random_state=42).head(2000)
    num_classes = len(df_merged.columns) - 1

    # Ensure image paths are correct
    df_merged['filename'] = df_merged['filename'].apply(lambda x: os.path.join(image_folder, x + ".png"))
    print(df_merged.head())

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
        print("Train dataset generated.")
    except Exception as e:
        print(f"Error creating train generator: {e}")
        return

    try:
        validation_generator = validation_datagen.flow_from_dataframe(
            dataframe=val_df,
            x_col='filename',
            y_col=val_df.columns[1:],
            target_size=(img_size, img_size),
            batch_size=batch_size,
            class_mode='raw'
        )
        print("Validation dataset generated.")
    except Exception as e:
        print(f"Error creating validation generator: {e}")
        return

    try:
        test_generator = validation_datagen.flow_from_dataframe(
            dataframe=test_df,
            x_col='filename',
            y_col=test_df.columns[1:],
            target_size=(img_size, img_size),
            batch_size=batch_size,
            class_mode='raw',
            shuffle=False
        )
        print("Test dataset generated.")
    except Exception as e:
        print(f"Error creating test generator: {e}")
        return

    # Model definition
    base_model = EfficientNetV2S(include_top=False, input_tensor=Input(shape=(img_size, img_size, 3)))

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(num_classes, activation='sigmoid')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    # Train only the top layers first
    for layer in base_model.layers:
        layer.trainable = False

    # Compile model
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
    print("Model compiled.")

    # Callbacks
    checkpoint_best = ModelCheckpoint('best_model.keras', monitor='val_loss', save_best_only=True, mode='min')
    checkpoint_last = ModelCheckpoint('last_model.keras', monitor='val_loss', save_best_only=False, save_freq='epoch')
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # Train top layers
    print("Starting training top layers.")
    try:
        history_top = model.fit(
            train_generator,
            steps_per_epoch=len(train_generator),
            validation_data=validation_generator,
            validation_steps=len(validation_generator),
            epochs=num_epochs // 2,
            callbacks=[checkpoint_best, checkpoint_last, early_stopping],
            verbose=1
        )
    except Exception as e:
        print(f"Error during training top layers: {e}")
        return

    # Unfreeze all layers
    for layer in base_model.layers:
        layer.trainable = True

    # Recompile model with a lower learning rate
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), loss='binary_crossentropy', metrics=['accuracy'])
    print("Model recompiled.")

    # Train all layers
    print("Starting training all layers.")
    try:
        history_all = model.fit(
            train_generator,
            steps_per_epoch=len(train_generator),
            validation_data=validation_generator,
            validation_steps=len(validation_generator),
            epochs=num_epochs // 2,
            callbacks=[checkpoint_best, checkpoint_last, early_stopping],
            verbose=1
        )
    except Exception as e:
        print(f"Error during training all layers: {e}")
        return

    # Evaluate model
    try:
        test_loss, test_accuracy = model.evaluate(test_generator, verbose=1)
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_accuracy:.4f}")
    except Exception as e:
        print(f"Error during evaluation: {e}")

    # Save final model
    model.save('final_model.h5')
    return history_all

if __name__ == "__main__":
    csv_path = 'D:/Skripsweet/BISMILLAH_KELAR/dataset/image_datasets.csv'  # Update with the actual path
    image_folder = 'D:/Skripsweet/BISMILLAH_KELAR/dataset/images'

    history = train_efficientnet_multilabel(csv_path, image_folder)
