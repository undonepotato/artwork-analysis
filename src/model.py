import numpy as np
import pandas as pd
import PIL
import matplotlib.pyplot as plt
import tensorflow as tf
import keras_tuner as kt
import tensorboard
import logging

resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu="aa-ml-tpu")
tf.config.experimental_connect_to_cluster(resolver)
tf.tpu.experimental.initialize_tpu_system(resolver)
print("All TPU devices: ", tf.config.list_logical_devices("TPU"))
strategy = tf.distribute.experimental.TPUStrategy(resolver)

# Set up logging

logging.basicConfig(level=logging.INFO)

logging.info("\nImports finished")
logging.info("Using NumPy version %s", np.__version__)
logging.info("Using TensorFlow version %s", tf.__version__)
logging.info("Using Pandas version %s", pd.__version__)
logging.info("Using PIL version %s", PIL.__version__)

EXPECTED_SIZE = (300, 300)  # pixels

# Load data

data_df = pd.read_parquet(
    "/home/icyseas_outlook_com/files/maindata/aadata/wikiart/data"
)
logging.info("Loaded initial data from Parquet")

data_df["image"] = [list(d.values())[0] for d in data_df["image"]]

# Set a test image for later display and confirmation
# test_image = data_df["image"][9]
# logging.warning("Test image saved")

# Decode JPEGs

image_dataset = tf.data.Dataset.from_tensor_slices(data_df["image"]).map(
    lambda x: tf.image.decode_jpeg(x, channels=3),
    num_parallel_calls=tf.data.AUTOTUNE,
)
label_dataset = tf.data.Dataset.from_tensor_slices(
    data_df.iloc[:, 1:].to_numpy()
)
logging.info("Decoded JPEGs")

dataset = tf.data.Dataset.zip((image_dataset, label_dataset))
dataset = dataset.shuffle(dataset.cardinality())

logging.info("Created and shuffled dataset")


# Split into training, validation, and testing
train = dataset.take(int(0.8 * len(dataset)))
val = dataset.skip(int(0.8 * len(dataset))).take(int(0.1 * len(dataset)))
test = dataset.skip(int(0.9 * len(dataset)))


logging.info("Split into training, validation, and testing")


# Augment images
@tf.function
def augment_image(image, seed):
    image = tf.image.random_brightness(image, max_delta=0.5, seed=seed)
    image = tf.image.random_contrast(image, lower=0.1, upper=0.9, seed=seed)
    image = tf.image.random_saturation(image, lower=0.1, upper=0.9, seed=seed)
    image = tf.image.random_hue(image, max_delta=0.1, seed=seed)

    return image


# Preprocess images
@tf.function
def preprocess_image(image):
    image = tf.keras.layers.Resizing(image, EXPECTED_SIZE)
    # rescaling is included as part of the EfficientNetV2 model, but
    # do it here anyway and just take include_preprocessing to off
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.keras.layers.Rescaling(1.0 / 127.5, offset=-1)(image)
    # [-1, 1] range
    return image


# Preprocess only images for all of train, val, test first
# Then augment only images in train

rng = tf.random.Generator.from_seed(1989)  # (tv)

# Display the test image
# test_image = preprocess_image(tf.image.decode_jpeg(test_image))
# ti = tf.image.convert_image_dtype(test_image, tf.uint8) # Needs [0, 1] range
# ti = tf.image.encode_jpeg(ti)
# ti = np.asarray(ti).tolist()
# PIL.Image.open(BytesIO(ti)).save("test.jpg")

# This also makes the other labels be in a list as
# model.fit expects for a tf.data.Dataset

train = train.map(
    lambda image, *labels: (
        preprocess_image(image),
        rng.make_seeds(2)[0],
        [*labels],
    ),
    num_parallel_calls=tf.data.AUTOTUNE,
).cache()

train = (
    train.map(
        lambda image, *labels: (
            augment_image(image, rng.make_seeds(2)[0]),
            [*labels],
        )
    )
    .batch(128, drop_remainder=True, num_parallel_calls=tf.data.AUTOTUNE)
    .prefetch(tf.data.AUTOTUNE)
)

# Only preprocess images for val and test

val = val.map(
    lambda image, *labels: (
        preprocess_image(image),
        [*labels],
    ),
    num_parallel_calls=tf.data.AUTOTUNE,
).prefetch(tf.data.AUTOTUNE)

test = test.map(
    lambda image, *labels: (
        preprocess_image(image),
        *labels,
    ),
    num_parallel_calls=tf.data.AUTOTUNE,
).prefetch(tf.data.AUTOTUNE)

logging.info(
    "Preprocessing complete, train dataset cardinality: %s", train.cardinality()
)
logging.info(
    "Preprocessing complete, val dataset cardinality: %s", val.cardinality()
)
logging.info(
    "Preprocessing complete, test dataset cardinality: %s", test.cardinality()
)

logging.info(
    "Data cleaning, processing, and augmentation complete; moving to training"
)

efficientnet_v2 = (
    tf.keras.applications.EfficientNetV2B0(  # Change to other sizes as needed
        include_preprocessing=False,
        weights="imagenet",
        input_shape=(300, 300, 3),
        include_top=False,
    )
)

logging.info("Loaded EfficientNetV2B0")

efficientnet_v2.trainable = False
print(efficientnet_v2.summary())

# Global average pooling and getting the
# features in a MLP-ready state

global_average_pooling = tf.keras.layers.GlobalAveragePooling2D(
    data_format="channels_last", keepdims=False
)


def create_single_classification_model() -> tf.keras.Model:
    inputs = tf.keras.layers.Input(shape=(300, 300, 3))
    shared_task = efficientnet_v2(inputs)
    shared_task = global_average_pooling(shared_task)

    artist_task = tf.keras.layers.Dense(129, activation="softmax")(shared_task)
    genre_task = tf.keras.layers.Dense(11, activation="softmax")(shared_task)
    style_task = tf.keras.layers.Dense(27, activation="softmax")(shared_task)
    model = tf.keras.Model(
        inputs=inputs, outputs=[artist_task, genre_task, style_task]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.01
        ),  # TODO Change learning rate
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy", "loss"],
        steps_per_execution="auto",  # TODO: CHANGE LATER TO A TUNED VALUE
    )

    return model


def create_three_classification_model(hp) -> tf.keras.Model:
    inputs = tf.keras.layers.Input(shape=(300, 300, 3))
    shared_task = efficientnet_v2(inputs)
    shared_task = global_average_pooling(shared_task)

    # Artist
    artist_task = tf.keras.layers.Dense(
        hp.Int("artist_units_0", min_value=64, max_value=512, step=64),
        activation="relu",
    )(shared_task)
    artist_task = tf.keras.layers.Dense(
        hp.Int("artist_units_1", min_value=64, max_value=512, step=64),
        activation="relu",
    )(artist_task)
    artist_task = tf.keras.layers.Dense(129, activation="softmax")(artist_task)

    # Genre

    genre_task = tf.keras.layers.Dense(
        hp.Int("genre_units_0", min_value=64, max_value=512, step=64),
        activation="relu",
    )(shared_task)
    genre_task = tf.keras.layers.Dense(
        hp.Int("genre_units_1", min_value=64, max_value=512, step=64),
        activation="relu",
    )(genre_task)
    genre_task = tf.keras.layers.Dense(129, activation="softmax")(genre_task)

    # Style

    style_task = tf.keras.layers.Dense(
        hp.Int("style_units_0", min_value=64, max_value=512, step=64),
        activation="relu",
    )(shared_task)
    style_task = tf.keras.layers.Dense(
        hp.Int("style_units_1", min_value=64, max_value=512, step=64),
        activation="relu",
    )(style_task)
    style_task = tf.keras.layers.Dense(129, activation="softmax")(style_task)

    model = tf.keras.Model(
        inputs=inputs, outputs=[artist_task, genre_task, style_task]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.01
        ),  # TODO Change learning rate
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy", "loss"],
        steps_per_execution="auto",  # TODO: CHANGE LATER TO A TUNED VALUE
    )

    return model


def create_five_classification_model(hp) -> tf.keras.Model:
    inputs = tf.keras.layers.Input(shape=(300, 300, 3))
    shared_task = efficientnet_v2(inputs)
    shared_task = global_average_pooling(shared_task)

    # Artist
    artist_task = tf.keras.layers.Dense(
        hp.Int("artist_units_0", min_value=64, max_value=512, step=64),
        activation="relu",
    )(shared_task)
    artist_task = tf.keras.layers.Dense(
        hp.Int("artist_units_1", min_value=64, max_value=512, step=64),
        activation="relu",
    )(artist_task)
    artist_task = tf.keras.layers.Dense(
        hp.Int("artist_units_2", min_value=64, max_value=512, step=64),
        activation="relu",
    )(artist_task)
    artist_task = tf.keras.layers.Dense(
        hp.Int("artist_units_3", min_value=64, max_value=512, step=64),
        activation="relu",
    )(artist_task)
    artist_task = tf.keras.layers.Dense(129, activation="softmax")(artist_task)

    # Genre

    genre_task = tf.keras.layers.Dense(
        hp.Int("genre_units_0", min_value=64, max_value=512, step=64),
        activation="relu",
    )(shared_task)
    genre_task = tf.keras.layers.Dense(
        hp.Int("genre_units_1", min_value=64, max_value=512, step=64),
        activation="relu",
    )(genre_task)
    genre_task = tf.keras.layers.Dense(
        hp.Int("genre_units_2", min_value=64, max_value=512, step=64),
        activation="relu",
    )(genre_task)
    genre_task = tf.keras.layers.Dense(
        hp.Int("genre_units_3", min_value=64, max_value=512, step=64),
        activation="relu",
    )(genre_task)
    genre_task = tf.keras.layers.Dense(129, activation="softmax")(genre_task)

    # Style

    style_task = tf.keras.layers.Dense(
        hp.Int("style_units_0", min_value=64, max_value=512, step=64),
        activation="relu",
    )(shared_task)
    style_task = tf.keras.layers.Dense(
        hp.Int("style_units_1", min_value=64, max_value=512, step=64),
        activation="relu",
    )(style_task)
    style_task = tf.keras.layers.Dense(
        hp.Int("style_units_2", min_value=64, max_value=512, step=64),
        activation="relu",
    )(style_task)
    style_task = tf.keras.layers.Dense(
        hp.Int("style_units_3", min_value=64, max_value=512, step=64),
        activation="relu",
    )(style_task)
    style_task = tf.keras.layers.Dense(129, activation="softmax")(style_task)

    model = tf.keras.Model(
        inputs=inputs, outputs=[artist_task, genre_task, style_task]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.01
        ),  # TODO Change learning rate
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy", "loss"],
        steps_per_execution="auto",  # TODO: CHANGE LATER TO A TUNED VALUE
    )

    return model


# Create model

with strategy.scope():
    tuner = kt.Hyperband(
        create_three_classification_model,
        objective="val_accuracy",
        max_epochs=20,
        factor=3,
        directory="tuner",
        project_name="aa-ml",
    )
    # model = create_model(multi_layered_classifications=False) # Compiled
    # Note: This will use sparse categorical cross-entropy loss for each of the three tasks.
    # They'll be equally balanced in the loss_weights since they weren't provided.

logging.info("Created model, starting tuning")

tuner.search(train, epochs=30, validation_data=val)
best_hps = tuner.get_best_hyperparameters(num_trials=1)

# Train model
model.fit(
    train,
    epochs=30,  # With an early stopping callback, probably won't reach this
    callbacks=[
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            min_delta=0.001,
            patience=7,
            restore_best_weights=True,
        )
    ],  # TODO: Update later; make sure to add early stopping and Tensorboard at least
    validation_data=val,
)
# TODO: Maybe add timestamps?
# TODO: You literally have one singular day
# TODO: Also make sure to tune batch sizes!
