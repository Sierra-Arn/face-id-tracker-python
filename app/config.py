# app/config.py
from functools import lru_cache
from pydantic import Field, field_validator, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Tuple

class Settings(BaseSettings):
    """Application settings and configuration."""

    # Model settings
    FACE_RECOGNITION_MODEL: str = Field(default="hog", description="Face recognition model (hog or cnn)")
    FACE_DISTANCE_THRESHOLD: float = Field(default=0.6, ge=0.0, le=1.0, description="Face recognition distance threshold")

    # Display settings
    CAMERA_INDEX: int = Field(default=0, description="Camera device index")
    CAMERA_WIDTH: int = Field(default=1280, gt=0, description="Camera capture width")
    CAMERA_HEIGHT: int = Field(default=720, gt=0, description="Camera capture height")
    RESIZE_FACTOR: int = Field(default=2, gt=0, description="Resize factor for processing")
    
    FONT: str = Field(default="FONT_HERSHEY_COMPLEX", description="Font for display text")
    COLOR_KNOWN: str = Field(default="0,255,0", description="RGB color for known faces (BGR format)")
    COLOR_UNKNOWN: str = Field(default="0,0,255", description="RGB color for unknown faces (BGR format)")
    COLOR_INFO: str = Field(default="255,0,0", description="RGB color for information text (BGR format)")
    
    # Path settings
    PHOTOS_DIR: str = Field(default="photos", description="Directory for storing face photos")

    # Localization
    UNKNOWN_PERSON_LABEL: str = Field(default="Unknown", description="Label for unknown persons")
    TIMEZONE_OFFSET: int = Field(default=0, description="Timezone offset in hours")
    TIMEZONE_LABEL: str = Field(default="UTC", description="Timezone label")

    @property
    def color_known(self) -> Tuple[int, int, int]:
        """Returns color for known faces in BGR format"""
        return self._parse_color(self.COLOR_KNOWN)

    @property
    def color_unknown(self) -> Tuple[int, int, int]:
        """Returns color for unknown faces in BGR format"""
        return self._parse_color(self.COLOR_UNKNOWN)

    @property
    def color_info(self) -> Tuple[int, int, int]:
        """Returns color for informational text in BGR format"""
        return self._parse_color(self.COLOR_INFO)

    @field_validator('COLOR_KNOWN', 'COLOR_UNKNOWN', 'COLOR_INFO', mode='before')
    @classmethod
    def validate_color_format(cls, value: str) -> str:
        """
        Validates color string format before setting value
        Checks format: 3 integers (0-255) separated by commas
        """
        try:
            parts = list(map(int, value.split(',')))
            if len(parts) != 3 or any(not (0 <= x <= 255) for x in parts):
                raise ValueError()
        except ValueError:
            raise ValidationError(
                f"Invalid color format: {value}. Expected 3 integers (0-255) separated by commas"
            )
        return value

    @staticmethod
    def _parse_color(color_str: str) -> Tuple[int, int, int]:
        """
        Internal method to convert color string to tuple of integers
        Should only be called after successful validation
        """
        return tuple(map(int, color_str.split(',')))  # type: ignore

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore"
    )

@lru_cache
def get_settings() -> Settings:
    """Returns cached application settings instance"""
    settings = Settings()
    return settings