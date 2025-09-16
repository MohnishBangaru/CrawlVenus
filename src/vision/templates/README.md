# Template Images Directory

This directory is used for storing template images for template matching functionality in the vision module.

## Purpose
Template matching allows the system to detect specific UI elements by comparing them against pre-defined template images. This is useful for:
- Detecting buttons, icons, or other UI elements
- Recognizing specific patterns in screenshots
- Automating interactions with known UI components

## Usage
Place PNG, JPG, or JPEG template images in this directory. The system will automatically load them for template matching.

## File Format
- Supported formats: PNG, JPG, JPEG
- Images should be in grayscale or color (will be converted to grayscale automatically)
- Use descriptive filenames as they become the template names

## Example
```
templates/
├── button_play.png
├── icon_settings.jpg
└── logo_app.png
```

## Note
If this directory is empty, template matching will be skipped without errors.
