"""Configuration management for the DroidBot-GPT framework."""

import os
from typing import Optional

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings

# Load environment variables
load_dotenv()


class Config(BaseSettings):
    """Configuration class for DroidBot-GPT framework."""
    
    # OpenAI Configuration
    openai_api_key: str = Field(default="", description="OpenAI API key (optional for basic functionality)")
    openai_model: str = Field(default="gpt-4")
    openai_max_tokens: int = Field(default=2000)
    openai_temperature: float = Field(default=0.7)

    # Phi Ground Configuration
    use_phi_ground: bool = Field(default=True, description="Enable Phi Ground for action generation")
    phi_ground_model: str = Field(default="microsoft/Phi-3-vision-128k-instruct", description="Phi Ground model to use")
    phi_ground_temperature: float = Field(default=0.7, description="Temperature for Phi Ground generation")
    phi_ground_max_tokens: int = Field(default=256, description="Max tokens for Phi Ground generation")
    phi_ground_confidence_threshold: float = Field(default=0.5, description="Minimum confidence for Phi Ground actions")

    # Prompt parameters
    prompt_max_ui_elements: int = Field(default=20)
    prompt_max_history: int = Field(default=10)
    
    # Android Configuration
    android_device_id: str = Field(default="emulator-5554")
    android_platform_version: int = Field(default=30)
    android_app_package: str = Field(default="com.android.settings")
    android_activity: str = Field(default=".Settings")
    
    # Computer Vision Configuration
    cv_confidence_threshold: float = Field(default=0.8)
    cv_template_matching_threshold: float = Field(default=0.9)
    cv_template_max_scales: int = Field(default=5)  # number of scales above and below 1.0
    cv_template_scale_step: float = Field(default=0.9)  # Factor to multiply per step (<1)
    cv_template_threads: int = Field(default=4)  # Thread pool size for template matching
    save_vision_debug: bool = Field(default=True)
    vision_debug_dir: str = Field(default="vision_debug")  # legacy debug dir
    ocr_images_dir: str = Field(default="ocr_images")  # New directory for OCR overlays
    cv_screenshot_quality: int = Field(default=90)
    cv_max_retries: int = Field(default=3)
    use_fast_screencap: bool = Field(default=True)  # Use adb exec-out screencap for fast capture
    use_ocr: bool = Field(default=False)  # OCR off by default with UI-Venus

    # OCR Engine
    # Path to the Tesseract OCR binary (leave None to use system PAT    tesseract_cmd: Optional[str] = Field(default=None, description="Path to Tesseract executable")

    # Framework Configuration
    log_level: str = Field(default="INFO")
    debug_mode: bool = Field(default=False)
    save_screenshots: bool = Field(default=True)
    screenshot_dir: str = Field(default="screenshots")
    max_wait_time: int = Field(default=10)
    retry_attempts: int = Field(default=3)
    
    # Resource Management
    max_memory_usage: str = Field(default="512MB")
    cpu_throttle_threshold: int = Field(default=80)
    batch_size: int = Field(default=5)
    
    # API Configuration
    api_host: str = Field(default="localhost")
    api_port: int = Field(default=8000)
    api_workers: int = Field(default=4)
    cors_origins: list[str] = Field(default=["http://localhost:3000"])
    
    # Version Control
    enable_version_control: bool = Field(default=True)
    version_backup_count: int = Field(default=5)
    auto_rollback: bool = Field(default=True)
    
    # Venus Configuration
    use_ui_venus: bool = Field(default=True, description="Enable UI-Venus for action generation")
    ui_venus_model: str = Field(default="inclusionAI/UI-Venus-Ground-7B", description="UI-Venus model to use")
    ui_venus_confidence_threshold: float = Field(default=0.5)
    
    class Config:
        """Pydantic configuration for environment loading."""

        env_file = ".env"
        case_sensitive = False
        extra = "ignore"  # Ignore unexpected env vars rather than raising errors
    
    def validate_config(self) -> bool:
        """Validate configuration values."""
        if self.cv_confidence_threshold < 0 or self.cv_confidence_threshold > 1:
            raise ValueError("CV confidence threshold must be between 0 and 1")
        
        if self.max_wait_time <= 0:
            raise ValueError("Max wait time must be positive")
        
        return True
    
    def validate_basic_config(self) -> bool:
        """Validate basic configuration for testing without OpenAI API key."""
        if self.cv_confidence_threshold < 0 or self.cv_confidence_threshold > 1:
            raise ValueError("CV confidence threshold must be between 0 and 1")
        
        if self.max_wait_time <= 0:
            raise ValueError("Max wait time must be positive")
        
        return True
    
    def get_screenshot_path(self) -> str:
        """Get the full path to screenshot directory."""
        return os.path.join(os.getcwd(), self.screenshot_dir)


# Global configuration instance
try:
    config = Config()
except Exception as e:
    print(f"Warning: Could not load configuration: {e}")
    # Create a minimal config for basic functionality
    config = Config(openai_api_key="")