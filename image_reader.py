import os
import numpy as np
from pathlib import Path
from typing import Optional, List
import warnings

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_XLA_FLAGS", "--tf_xla_enable_xla_devices=false")
os.environ.setdefault("KERAS_BACKEND", "tensorflow")


class SudokuImageReader:
    MODEL_VERSION = 2
    
    def __init__(self):
        self.model = None
        self._cv2 = None
        self._tf = None
        self._check_dependencies()
    
    def _check_dependencies(self) -> None:
        try:
            import cv2
            self._cv2 = cv2
        except ImportError:
            self._cv2 = None
        
        try:
            import tensorflow as tf
            try:
                tf.config.set_visible_devices([], "GPU")
            except Exception:
                pass
            try:
                tf.config.optimizer.set_jit(False)
            except Exception:
                pass
            self._tf = tf
        except ImportError:
            try:
                import keras
                self._tf = keras
            except ImportError:
                self._tf = None
    
    def _ensure_dependencies(self) -> None:
        missing = []
        if self._cv2 is None:
            missing.append("opencv-python")
        if self._tf is None:
            missing.append("tensorflow (or keras)")
        
        if missing:
            raise ImportError(
                f"Missing required packages: {', '.join(missing)}\n"
                f"Install them with: pip install opencv-python tensorflow"
            )
    
    def _load_or_create_model(self):
        if self.model is not None:
            return
        
        self._ensure_dependencies()
        
        model_path = Path(__file__).parent / "digit_model.h5"
        version_path = Path(__file__).parent / "digit_model.version"

        try:
            if model_path.exists() and version_path.exists() and version_path.read_text().strip() == str(self.MODEL_VERSION):
                if hasattr(self._tf, 'keras'):
                    self.model = self._tf.keras.models.load_model(str(model_path))
                else:
                    self.model = self._tf.models.load_model(str(model_path))
                return
        except Exception:
            pass
        
        print("Training digit recognition model on MNIST dataset...")
        self._train_mnist_model()
    
    def _train_mnist_model(self):
        self._ensure_dependencies()
        
        if hasattr(self._tf, 'keras'):
            keras = self._tf.keras
        else:
            keras = self._tf
        
        if hasattr(self._tf, 'config'):
            try:
                self._tf.config.set_visible_devices([], "GPU")
            except Exception:
                pass
            try:
                self._tf.config.optimizer.set_jit(False)
            except Exception:
                pass

        keras.backend.clear_session()
        keras.utils.set_random_seed(42)
        
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
        
        x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
        x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0

        data_augmentation = keras.Sequential([
            keras.layers.RandomRotation(0.08, fill_mode='constant'),
            keras.layers.RandomTranslation(0.08, 0.08, fill_mode='constant'),
            keras.layers.RandomZoom(0.05, fill_mode='constant'),
        ], name='data_augmentation')
        
        model = keras.Sequential([
            keras.layers.Input(shape=(28, 28, 1)),
            data_augmentation,
            keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.Flatten(),
            keras.layers.Dropout(0.35),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(10, activation='softmax')
        ])
        
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3),
                     loss='sparse_categorical_crossentropy',
                     metrics=['accuracy'])
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(x_train, y_train, epochs=8, batch_size=128,
                      validation_split=0.1, verbose=0)
        
        test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
        print(f"MNIST test accuracy: {test_acc:.3f}")
        self.model = model
        
        try:
            model_path = Path(__file__).parent / "digit_model.h5"
            version_path = Path(__file__).parent / "digit_model.version"
            model.save(str(model_path))
            version_path.write_text(str(self.MODEL_VERSION))
            print(f"Model saved to {model_path}")
        except Exception as e:
            print(f"Could not save model: {e}")
    
    def preprocess_image(self, image_path: str) -> np.ndarray:
        self._ensure_dependencies()
        cv2 = self._cv2
        
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        return thresh
    
    def find_sudoku_grid(self, img: np.ndarray) -> Optional[np.ndarray]:
        self._ensure_dependencies()
        cv2 = self._cv2
        
        contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        largest_contour = max(contours, key=cv2.contourArea)
        
        epsilon = 0.02 * cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, epsilon, True)
        
        if len(approx) != 4:
            x, y, w, h = cv2.boundingRect(largest_contour)
            approx = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=np.float32)
        else:
            approx = approx.reshape(4, 2).astype(np.float32)
        
        approx = self._order_points(approx)
        
        size = 450
        dst = np.array([[0, 0], [size, 0], [size, size], [0, size]], dtype=np.float32)
        
        matrix = cv2.getPerspectiveTransform(approx, dst)
        warped = cv2.warpPerspective(img, matrix, (size, size))
        
        return warped
    
    def _order_points(self, pts: np.ndarray) -> np.ndarray:
        rect = np.zeros((4, 2), dtype=np.float32)
        
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        
        return rect
    
    def extract_cells(self, grid_img: np.ndarray) -> List[np.ndarray]:
        cells = []
        cell_size = grid_img.shape[0] // 9
        
        for i in range(9):
            for j in range(9):
                margin = cell_size // 10
                y1 = i * cell_size + margin
                y2 = (i + 1) * cell_size - margin
                x1 = j * cell_size + margin
                x2 = (j + 1) * cell_size - margin
                
                cell = grid_img[y1:y2, x1:x2]
                cells.append(cell)
        
        return cells
    
    def recognize_digit(self, cell_img: np.ndarray) -> int:
        self._ensure_dependencies()
        cv2 = self._cv2
        
        if np.sum(cell_img) < cell_img.size * 255 * 0.03:
            return 0
        
        gray = cell_img
        if len(gray.shape) == 3:
            gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)

        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        thresh = cv2.bitwise_not(thresh)

        coords = cv2.findNonZero(thresh)
        if coords is not None:
            x, y, w, h = cv2.boundingRect(coords)
            digit_only = thresh[y:y + h, x:x + w]
            if digit_only.size > 0:
                size = max(w, h)
                canvas = np.zeros((size, size), dtype=np.uint8)
                pad_x = (size - w) // 2
                pad_y = (size - h) // 2
                canvas[pad_y:pad_y + h, pad_x:pad_x + w] = digit_only
                resized = cv2.resize(canvas, (20, 20))
            else:
                resized = cv2.resize(thresh, (20, 20))
        else:
            resized = cv2.resize(thresh, (20, 20))

        padded = np.pad(resized, ((4, 4), (4, 4)), mode='constant', constant_values=0)
        normalized = padded.astype('float32') / 255.0
        input_img = normalized.reshape(1, 28, 28, 1)
        
        self._load_or_create_model()
        predictions = self.model.predict(input_img, verbose=0)
        digit = np.argmax(predictions)
        confidence = np.max(predictions)
        
        if confidence < 0.55:
            return 0
        
        return int(digit)
    
    def read_sudoku_from_image(self, image_path: str) -> np.ndarray:
        print(f"Processing image: {image_path}")
        
        preprocessed = self.preprocess_image(image_path)
        
        grid = self.find_sudoku_grid(preprocessed)
        if grid is None:
            raise ValueError("Could not find sudoku grid in the image")
        
        cells = self.extract_cells(grid)
        
        board = np.zeros((9, 9), dtype=int)
        print("Recognizing digits...")
        
        for i, cell in enumerate(cells):
            row = i // 9
            col = i % 9
            digit = self.recognize_digit(cell)
            board[row, col] = digit
        
        return board


def read_sudoku_from_image(image_path: str) -> np.ndarray:
    reader = SudokuImageReader()
    return reader.read_sudoku_from_image(image_path)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        try:
            board = read_sudoku_from_image(image_path)
            print("\nExtracted sudoku board:")
            print(board)
        except Exception as e:
            print(f"Error: {e}")
    else:
        print("Usage: python3 image_reader.py <image_path>")
        print("\nThis module requires:")
        print("  - opencv-python: pip install opencv-python")
        print("  - tensorflow: pip install tensorflow")
