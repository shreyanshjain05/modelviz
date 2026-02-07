"""
TensorFlow/Keras Demo

Demonstrates modelviz visualization with Keras Sequential models.
"""

from modelviz import visualize


def create_keras_mlp():
    """
    Create a simple Keras MLP.
    
    Uses lazy import to avoid requiring TensorFlow at module level.
    """
    import tensorflow as tf
    
    return tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(10,)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation='softmax'),
    ])


def create_keras_cnn():
    """
    Create a simple Keras CNN for image classification.
    """
    import tensorflow as tf
    
    return tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax'),
    ])


def main():
    """Run the demo."""
    print("Creating Keras MLP model...")
    
    try:
        model = create_keras_mlp()
    except ImportError:
        print("TensorFlow is not installed. Please install with:")
        print("  pip install tensorflow")
        return
    
    print("\nModel summary:")
    model.summary()
    
    print("\nGenerating visualization...")
    
    # Visualize (no input_shape needed for Keras - it's built-in)
    diagram = visualize(model, title="Keras MLP")
    
    # Save to file
    diagram.render("keras_mlp_visualization", format="svg", cleanup=True)
    print("Saved visualization to: keras_mlp_visualization.svg")
    
    # Also visualize CNN
    print("\nCreating Keras CNN model...")
    model_cnn = create_keras_cnn()
    
    print("\nModel summary:")
    model_cnn.summary()
    
    diagram_cnn = visualize(model_cnn, title="Keras CNN")
    diagram_cnn.render("keras_cnn_visualization", format="svg", cleanup=True)
    print("Saved visualization to: keras_cnn_visualization.svg")


if __name__ == "__main__":
    main()
