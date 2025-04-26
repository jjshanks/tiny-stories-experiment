import hashlib
import json
import pickle
from pathlib import Path
from typing import Any, Callable, Dict, Tuple, TypeVar

# Define a generic type for our cached data
T = TypeVar('T')

class CacheManager:
    """Manages disk caching for intermediate results."""

    def __init__(self, cache_dir: Path):
        """
        Initializes the CacheManager.

        Args:
            cache_dir: The directory to store cache files.
        """
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        print(f"CacheManager initialized with directory: {self.cache_dir}")

    def get_cache_key(self, params: dict) -> str:
        """
        Generate a unique cache key based on input parameters.

        Args:
            params: Dictionary of parameters to hash.

        Returns:
            String hash representing the unique parameter combination.
        """
        param_str = json.dumps(params, sort_keys=True)
        return hashlib.md5(param_str.encode()).hexdigest()

    def cache_exists(self, cache_key: str, step_name: str) -> bool:
        """
        Check if a cache file exists for the given key and step.

        Args:
            cache_key: The hash key for the parameter set.
            step_name: The processing step (e.g., 'dataset_subsets').

        Returns:
            Boolean indicating if cache exists.
        """
        cache_file = self.cache_dir / f"{step_name}_{cache_key}.pkl"
        return cache_file.exists()

    def save_to_cache(self, data: Any, cache_key: str, step_name: str):
        """
        Save data to cache file for future reuse.

        Args:
            data: The data to cache.
            cache_key: The hash key for the parameter set.
            step_name: The processing step being cached.
        """
        cache_file = self.cache_dir / f"{step_name}_{cache_key}.pkl"
        with open(cache_file, "wb") as f:
            pickle.dump(data, f)
        print(f"Cached {step_name} to {cache_file}")

    def load_from_cache(self, cache_key: str, step_name: str) -> Any:
        """
        Load previously processed data from cache.

        Args:
            cache_key: The hash key for the parameter set.
            step_name: The processing step to load.

        Returns:
            The cached data.
        """
        cache_file = self.cache_dir / f"{step_name}_{cache_key}.pkl"
        with open(cache_file, "rb") as f:
            data = pickle.load(f)
        print(f"Loaded {step_name} from cache {cache_file}")
        return data
        
    def get_or_compute(self, params: dict, step_name: str, compute_fn: Callable[[], T], 
                      no_cache: bool = False) -> Tuple[T, str]:
        """
        Get data from cache if it exists, or compute and cache it if not.
        
        Args:
            params: Parameters to generate the cache key.
            step_name: Name of the processing step.
            compute_fn: Function that computes the data if not in cache.
            no_cache: If True, always compute and don't use/update cache.
            
        Returns:
            A tuple of (computed_data, cache_key)
        """
        cache_key = self.get_cache_key(params)
        
        if not no_cache and self.cache_exists(cache_key, step_name):
            return self.load_from_cache(cache_key, step_name), cache_key
            
        result = compute_fn()
        
        if not no_cache:
            self.save_to_cache(result, cache_key, step_name)
            
        return result, cache_key
