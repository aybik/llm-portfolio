import os
from typing import Dict, Any, Optional


class Config:
    """Configuration management."""

    def __init__(self, config_dict: Optional[Dict] = None):
        self._config = config_dict or {}

    def get(self, key: str, default: Any = None) -> Any:
        """Get config value."""
        return self._config.get(key, default)

    def set(self, key: str, value: Any):
        """Set config value."""
        self._config[key] = value

    @classmethod
    def from_env(cls) -> "Config":
        """Create config from environment variables."""
        config = {
            "openai_api_key": os.getenv("OPENAI_API_KEY"),
            "model": os.getenv("MODEL", "gpt-4"),
            "temperature": float(os.getenv("TEMPERATURE", "0.0")),
            "max_tokens": int(os.getenv("MAX_TOKENS", "2000")),
        }
        return cls(config)


if __name__ == "__main__":
    # Test configuration
    config = Config.from_env()
    print(f"Model: {config.get('model')}")
    print(f"Temperature: {config.get('temperature')}")
    print(f"Max tokens: {config.get('max_tokens')}")
