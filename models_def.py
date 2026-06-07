"""Custom model wrappers used by train.py and loaded by app.py via joblib."""
import numpy as np
from sklearn.preprocessing import StandardScaler


class PersistenceModel:
    """Baseline: future AQI equals current AQI (column `aqi`)."""

    def __init__(self, aqi_col_idx: int):
        self.aqi_col_idx = aqi_col_idx

    def fit(self, X, y):
        return self

    def predict(self, X):
        return X[:, self.aqi_col_idx].astype(np.float64)


def build_tensorflow_model(input_dim: int):
    import tensorflow as tf
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


class TensorFlowAQIModel:
    """Sklearn-like wrapper for Keras regression models."""

    def __init__(self):
        self.scaler = StandardScaler()
        self.model = None
        self.input_dim = None

    def fit(self, X, y, epochs=40, batch_size=64, validation_split=0.1):
        import tensorflow as tf
        self.input_dim = X.shape[1]
        Xs = self.scaler.fit_transform(X)
        self.model = build_tensorflow_model(self.input_dim)
        self.model.fit(
            Xs, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=0,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss", patience=5, restore_best_weights=True,
                ),
            ],
        )
        return self

    def predict(self, X):
        Xs = self.scaler.transform(X)
        return self.model.predict(Xs, verbose=0).ravel()
