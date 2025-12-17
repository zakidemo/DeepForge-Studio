# Changelog
All notable changes to this project will be documented in this file.

## [1.0.3] 
### Added
- Dataset folder-structure guidance in exported deep learning scripts/notebooks.
- Friendlier ML export behavior (explicit X/y guard message to prevent silent failures).

### Changed
- Exported configuration JSON is now sanitized for consistency:
  - Deep learning modes (`pretrained`, `scratch`, `custom`) enforce `mlConfig: null`.
  - ML mode (`modelMode: ml`) enforces `customLayers: []` and `customLayerConfigs: []`.
- ML exports use scikit-learn pipelines where appropriate (e.g., `StandardScaler + KNN` when scaling is enabled).
- ML exports no longer include TensorFlow seeding/imports.

### Fixed
- ML export formatting issues that could lead to indentation/import-related runtime errors.
- Reduced configuration “cross-contamination” when switching between ML and DL modes before exporting.

## [1.0.2] 
### Added
- GitHub Actions workflows for CI and GitHub Pages deployment (`.github/workflows`).
- Expanded VGG16 (from scratch) export option for educational transparency.

### Fixed
- Export naming and mode-selection UX improvements (pretrained vs scratch selection via modal).

## [1.0.0] 
### Added
- Visual architecture selection (prebuilt) and Custom Builder (layer-by-layer).
- Modal selection for models supporting both From Scratch and Pretrained.
- Export: Python training script (`.py`) and Google Colab notebook (`.ipynb`).
- Export/Import experiment configuration (`.json`).
- Optional AI Optimizer integration (Gemini API key stored locally).

### Fixed
- Export mode consistency (ML vs DL vs pretrained).
- Code preview reliability and safer notifications.
