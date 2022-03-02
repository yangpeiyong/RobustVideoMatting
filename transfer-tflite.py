# import tensorflow as tf

# model = tf.keras.models.load_model('rvm_mobilenetv3_tf')
# model = tf.function(model)

# converter = tf.lite.TFLiteConverter.from_keras_model(model)
# tflite_model = converter.convert()

# # Save the model.
# with open('rvm_mobilenetv3.tflite', 'wb') as f:
#   f.write(tflite_model)


import tensorflow as tf

# Convert the model
converter = tf.lite.TFLiteConverter.from_saved_model("rvm_mobilenetv3_tf") # path to the SavedModel directory
tflite_model = converter.convert()

# Save the model.
with open('model.tflite', 'wb') as f:
  f.write(tflite_model)