import tensorflow as tf
from densenet_tf import DenseNet121
import tensorflow_model_optimization as tfmot

model = DenseNet121(
    include_top=True, weights=None, input_tensor=None,
    input_shape=(256, 256, 3), pooling=None, classes=3, dropout_rate=0.1,
)

# Convert the model
converter = tf.lite.TFLiteConverter.from_keras_model(model) # path to the SavedModel directory
converter.experimental_new_converter = True
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS,
]
tflite_model = converter.convert()

# Save the model.
with open('model.tflite', 'wb') as f:
  f.write(tflite_model)
