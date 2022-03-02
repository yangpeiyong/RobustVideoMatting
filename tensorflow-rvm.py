import tensorflow as tf

model = tf.keras.models.load_model('rvm_mobilenetv3_tf')
model = tf.function(model)
