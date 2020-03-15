from tensorflow import keras
# from keras import layers
from preprocess import get_sample
import tensorflow as tf

ds_sefaria = tf.data.Dataset.from_generator(get_sample, args=[], output_types=(tf.int8, tf.int8), output_shapes = ( [29, 1024], [4] ) )
#TODO: shapes and arguments should be parametric

inputs = keras.Input(shape=(29, 1024), name='characters')
x = keras.layers.Flatten()(inputs) 
x = keras.layers.Dense(64, activation='relu', name='dense_1')(x)
x = keras.layers.Dense(64, activation='relu', name='dense_2')(x)
outputs = keras.layers.Dense(4, name='predictions')(x)

model = keras.Model(inputs=inputs, outputs=outputs)

DATASET_SIZE = 8377
train_size = int(0.7 * DATASET_SIZE)
val_size = int(0.15 * DATASET_SIZE)
test_size = int(0.15 * DATASET_SIZE)


model.compile(optimizer=keras.optimizers.RMSprop(),  # Optimizer
              # Loss function to minimize
              loss='categorical_crossentropy',
              # List of metrics to monitor
              metrics=['accuracy'])

model.summary()


full_dataset = ds_sefaria.shuffle(buffer_size=DATASET_SIZE)
train_dataset = full_dataset.take(train_size)
test_dataset = full_dataset.skip(train_size)
val_dataset = test_dataset.skip(val_size)
test_dataset = test_dataset.take(test_size)

train_dataset = train_dataset.batch(64)
test_dataset = test_dataset.batch(64)
val_dataset = val_dataset.batch(64)

model.fit(train_dataset, epochs=3)
print('\n# Evaluate')
result = model.evaluate(test_dataset)
dict(zip(model.metrics_names, result))