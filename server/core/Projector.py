import os
import pickle
import logging
import numpy as np

try:
    # pyrefly: ignore [missing-import]
    import umap
except ImportError:
    umap = None

try:
    from sklearn.decomposition import PCA
except ImportError:
    PCA = None

logger = logging.getLogger(__name__)

class MapProjector:
    """
    Dimensionality reduction manager to project 128D VAE beatmap embeddings
    into 2D (coord_x, coord_y) coordinates for Fog of War mapping.
    Uses UMAP (with PCA fallback for small sample sizes or missing dependencies).
    """

    def __init__(self, model_path: str = None):
        if model_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            model_path = os.path.join(base_dir, "models", "umap_projector.pkl")
        self.model_path = model_path
        self.reducer = None
        self._load_model()

    def _load_model(self):
        if os.path.exists(self.model_path):
            try:
                with open(self.model_path, 'rb') as f:
                    self.reducer = pickle.load(f)
                logger.info(f"Loaded existing 2D map projector model from {self.model_path}")
            except Exception as e:
                logger.warning(f"Failed to load projector model from {self.model_path}: {e}")
                self.reducer = None

    def save_model(self):
        if self.reducer is not None:
            try:
                os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
                with open(self.model_path, 'wb') as f:
                    pickle.dump(self.reducer, f)
                logger.info(f"Saved 2D map projector model to {self.model_path}")
            except Exception as e:
                logger.error(f"Failed to save projector model to {self.model_path}: {e}")

    def fit(self, embeddings: np.ndarray):
        """
        Fit UMAP (or PCA fallback) on an array of shape (N, D).
        """
        if embeddings is None or len(embeddings) == 0:
            logger.warning("No embeddings provided to fit MapProjector.")
            return

        embeddings = np.asarray(embeddings, dtype=np.float32)
        n_samples = len(embeddings)

        # Use UMAP if umap-learn is installed and we have at least 15 samples
        use_umap = False
        if n_samples >= 15 and umap is not None:
            try:
                n_neighbors = min(15, n_samples - 1)
                self.reducer = umap.UMAP(
                    n_components=2,
                    n_neighbors=n_neighbors,
                    min_dist=0.1,
                    metric='cosine',
                    random_state=42
                )
                self.reducer.fit(embeddings)
                use_umap = True
                logger.info(f"Fitted UMAP reducer on {n_samples} map embeddings.")
            except Exception as e:
                logger.warning(f"UMAP fit failed or unavailable: {e}. Falling back to PCA.")

        if not use_umap and PCA is not None:
            n_components = min(2, n_samples)
            self.reducer = PCA(n_components=n_components, random_state=42)
            self.reducer.fit(embeddings)
            logger.info(f"Fitted PCA fallback reducer on {n_samples} map embeddings.")

        self.save_model()

    def transform(self, embedding: np.ndarray) -> tuple[float, float]:
        """
        Transform a single 1D embedding vector (shape (128,)) or 2D array into (coord_x, coord_y).
        """
        if embedding is None:
            return 0.0, 0.0

        arr = np.asarray(embedding, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)

        if self.reducer is None:
            # Fallback if no fitted model exists yet: use first two dimensions of VAE embedding
            if arr.shape[1] >= 2:
                return round(float(arr[0, 0]), 4), round(float(arr[0, 1]), 4)
            return 0.0, 0.0

        try:
            coords = self.reducer.transform(arr)
            cx = float(coords[0, 0])
            cy = float(coords[0, 1]) if coords.shape[1] > 1 else 0.0
            return round(cx, 4), round(cy, 4)
        except Exception as e:
            logger.error(f"Error projecting embedding: {e}")
            if arr.shape[1] >= 2:
                return round(float(arr[0, 0]), 4), round(float(arr[0, 1]), 4)
            return 0.0, 0.0

    def transform_batch(self, embeddings: np.ndarray) -> list[tuple[float, float]]:
        """
        Transform multiple 1D or 2D embeddings into a list of (coord_x, coord_y) tuples.
        """
        if embeddings is None or len(embeddings) == 0:
            return []

        arr = np.asarray(embeddings, dtype=np.float32)
        if self.reducer is None:
            self.fit(arr)

        if self.reducer is None:
            return [(round(float(row[0]), 4), round(float(row[1]), 4)) if len(row) >= 2 else (0.0, 0.0) for row in arr]

        try:
            coords = self.reducer.transform(arr)
            result = []
            for row in coords:
                cx = float(row[0])
                cy = float(row[1]) if len(row) > 1 else 0.0
                result.append((round(cx, 4), round(cy, 4)))
            return result
        except Exception as e:
            logger.error(f"Error projecting embedding batch: {e}")
            return [(round(float(row[0]), 4), round(float(row[1]), 4)) if len(row) >= 2 else (0.0, 0.0) for row in arr]
