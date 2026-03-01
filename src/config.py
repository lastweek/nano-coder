"""Centralized configuration for Nano-Coder."""

from typing import Optional
from pathlib import Path
import yaml
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMConfig(BaseSettings):
    """LLM provider configuration."""

    model_config = SettingsConfigDict(
        env_prefix="LLM_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )

    provider: str = Field(default="openai", description="LLM provider: openai, azure, ollama, custom")
    base_url: Optional[str] = Field(default=None, description="Custom API base URL")
    model: Optional[str] = Field(default=None, description="Model name")
    api_key: Optional[str] = Field(default=None, description="API key (auto-detected from provider)")

    @field_validator("provider")
    @classmethod
    def validate_provider(cls, v: str) -> str:
        """Normalize provider name to lowercase."""
        return v.lower()


class LoggingConfig(BaseSettings):
    """Logging configuration."""

    model_config = SettingsConfigDict(
        env_prefix="",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )

    enabled: bool = Field(default=True, alias="ENABLE_LOGGING")
    async_mode: bool = Field(default=False, alias="ASYNC_LOGGING")
    log_dir: str = Field(default="logs", alias="LOG_DIR")
    buffer_size: int = Field(default=10, ge=1, le=100)


class AgentConfig(BaseSettings):
    """Agent behavior configuration."""

    model_config = SettingsConfigDict(
        env_prefix="AGENT_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )

    max_iterations: int = Field(default=10, ge=1, le=100)


class UIConfig(BaseSettings):
    """UI/Display configuration."""

    model_config = SettingsConfigDict(
        env_prefix="",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )

    enable_streaming: bool = Field(default=True)
    loading_indicator_interval: float = Field(default=0.8, gt=0)


class Config:
    """Global configuration container."""

    _instance: Optional['Config'] = None
    _config_path: Path = Path("config.yaml")

    def __init__(self, config_dict: Optional[dict] = None):
        """Initialize all config sections.

        Priority (highest to lowest):
        1. Environment variables
        2. config.yaml values
        3. Default values
        """
        config_dict = config_dict or {}

        # Initialize each config section
        # Create instances first (loads env vars via pydantic-settings)
        # Then apply yaml values only for fields that don't have values set
        self.llm = self._create_config(LLMConfig, config_dict.get("llm", {}))
        self.logging = self._create_config(LoggingConfig, config_dict.get("logging", {}))
        self.agent = self._create_config(AgentConfig, config_dict.get("agent", {}))
        self.ui = self._create_config(UIConfig, config_dict.get("ui", {}))

    @staticmethod
    def _create_config(config_class, yaml_values: dict):
        """Create a config section with proper priority: env vars > yaml > defaults.

        Args:
            config_class: Pydantic Settings class to instantiate
            yaml_values: Dict of values from config.yaml

        Returns:
            Config instance with env vars taking precedence over yaml values
        """
        # Import os to check if env var is set
        import os

        # Check each field to see if an env var is set
        # Only apply yaml value if no env var is set
        filtered_values = {}
        for key, value in yaml_values.items():
            # Get the env var name for this field
            # For LLMConfig, env prefix is "LLM_"
            # For LoggingConfig, no prefix, but uses aliases like "ENABLE_LOGGING"
            # For AgentConfig, env prefix is "AGENT_"
            # For UIConfig, no prefix

            env_var_name = None
            if hasattr(config_class, 'model_config'):
                # Check for env_prefix in settings config
                env_prefix = config_class.model_config.get('env_prefix', '')
                # Build env var name: PREFIX + KEY (uppercased)
                env_var_name = (env_prefix + key).upper()

                # For fields with aliases (like ENABLE_LOGGING), check those too
                if key in config_class.model_fields:
                    field = config_class.model_fields[key]
                    if hasattr(field, 'alias'):
                        env_var_name = field.alias

            # Check if env var is set
            if env_var_name and env_var_name in os.environ:
                # Env var is set, skip yaml value (env var takes precedence)
                continue
            elif not env_var_name and key.upper() in os.environ:
                # Try uppercase key without prefix
                continue
            else:
                # No env var set, use yaml value
                filtered_values[key] = value

        # Create instance with filtered yaml values
        # Pydantic-settings will still apply env vars for any fields not in filtered_values
        return config_class(**filtered_values)

    @classmethod
    def load(cls, config_path: Optional[str] = None) -> 'Config':
        """Load or create singleton config instance.

        Priority:
        1. config.yaml (if exists and not in test mode) → provides base values
        2. Environment variables → override yaml and defaults
        3. Class defaults → fallback
        """
        if cls._instance is not None:
            return cls._instance

        import os

        # Load from config.yaml if it exists and not in test mode
        config_dict = {}
        if config_path:
            cls._config_path = Path(config_path)

        # Skip loading config.yaml in test mode (when NANO_CODER_TEST is set)
        is_test_mode = os.environ.get("NANO_CODER_TEST", "").lower() == "true"

        if not is_test_mode and cls._config_path.exists():
            try:
                with open(cls._config_path) as f:
                    config_dict = yaml.safe_load(f) or {}
                print(f"Loaded configuration from {cls._config_path}")
            except Exception as e:
                print(f"Warning: Failed to load config from {cls._config_path}: {e}")
                print("Using environment variables and defaults.")
        else:
            if is_test_mode:
                print("Test mode: Skipping config.yaml, using environment variables and defaults.")
            else:
                print(f"No config.yaml found at {cls._config_path}")
                print("Using environment variables and defaults.")

        # Create config instance (env vars will be applied by pydantic-settings)
        cls._instance = cls(config_dict)

        return cls._instance

    @classmethod
    def reload(cls) -> 'Config':
        """Force reload the configuration.

        This resets the singleton and creates a new instance with current
        environment variables. Useful for testing.

        Returns:
            New Config instance
        """
        cls._instance = None
        new_instance = cls.load()
        # Update global config variable
        global config
        config = new_instance
        return new_instance


# Global config instance
config = Config.load()
