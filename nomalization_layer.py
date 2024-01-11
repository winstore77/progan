# pixel-wise feature vector normalization layer
class PixelNormalization(Layer):
    # initialize the layer
    def __init__(self, **kwargs):
        super(PixelNormalization, self).__init__(**kwargs)
 
    # perform the operation
    def call(self, inputs):
        # computing pixel values
        values = inputs**2.0
        mean_values = backend.mean(values, axis=-1, keepdims=True)
        mean_values += 1.0e-8
        l2 = backend.sqrt(mean_values)
        normalized = inputs / l2
        return normalized
 
    # define the output shape of the layer
    def compute_output_shape(self, input_shape):
        return input_shape
