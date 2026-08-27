# Changelog
All notable changes to this project will be documented in this file.

## [2.5.0]
### Added
- **Exported image scripts now verify the class count against the dataset.**
  The tool cannot see your data, so a head built for N classes against a
  directory containing M produced an opaque shape error inside the loss function
  minutes into training. The script now stops immediately with the counts, the
  class names, and what to change.

### Fixed
- "Freeze all except last layer" froze every layer with weights. Most backbones
  end in a pooling layer that has no parameters, so taking the last layer
  literally made the option identical to freezing the whole backbone. It now
  leaves the last layer that actually has weights trainable, and reports which
  one, giving four distinct behaviours: head only, last convolutional block,
  half the backbone, or full fine-tuning.

## [2.4.0]
### Fixed
- **The "Freeze all except last layer" option produced no freezing at all.** It
  was offered in the interface but had no branch in the generator, so selecting
  it silently exported a fully trainable backbone. This is the same class of
  defect as the missing layer types fixed in 1.1.0: an interface option with no
  implementation behind it.
- The generator now raises on an unrecognised freezing option instead of
  emitting nothing.

### Changed
- **The default is now "Freeze base model" rather than "Train all layers".**
  The previous default fine-tuned the whole pretrained backbone at a learning
  rate chosen for a randomly initialised head, which destroys the pretrained
  features -- the most common transfer-learning mistake, and one this tool
  claims to help users avoid.
- Choosing "Train all layers" now emits an explanatory note in the exported
  script about the learning rate that choice implies.

### Added
- The regression suite covers all four freezing options and asserts that each
  one that should freeze actually emits freezing code.

## [2.3.0]
### Changed
- **Renamed to Layernaut Studio.** The previous candidate names collided with an
  existing bitmap drawing tool and, before that, with the DeepForge project. The
  new name is coined rather than a dictionary word or place name, and was checked
  against GitHub, PyPI, npm and Google Scholar before adoption.

## [2.2.0]
### Fixed
- **The AI Optimizer stopped working entirely** because the provider shut down
  `gemini-2.0-flash` on 1 June 2026. The model identifier was hardcoded in two
  separate places, so the feature broke with no change on our side.

### Changed
- The provider model is no longer hardcoded. On connect, the application queries
  the provider for the models the user's key can actually use, filters out those
  that cannot serve a text prompt (embedding, image, video, speech), and selects
  a current one. The user can change it from a dropdown without waiting for a
  release of this software.
- The second, duplicated API call in the connected-projects search now routes
  through the same client, so there is one place a model is chosen.
- A failure caused by a retired model is reported as such, with the action to
  take, instead of surfacing the raw provider message.

### Added
- `tests/model_selection_test.mjs`, covering discovery, filtering, fallback when
  no preferred model is available, and the retirement error path.

## [2.1.0]
### Added
- **Skip-connection control in the Custom Builder.** Any layer can now merge the
  output of an earlier layer with Add or Concatenate, from the configuration
  modal. v2.0.0 supported this in the generator but exposed no way to set it.
- **One configuration modal for every layer type**, built from a declarative
  spec. Five layer types previously had a hand-written modal each and the other
  five had none, so BatchNorm, Flatten, GlobalAvgPool, LSTM and GRU could not be
  configured at all; LSTM and GRU now expose units and return_sequences.
- **Reproducibility capsule export** (`Download capsule (.zip)`): the seeded
  script, the notebook, the full specification, pinned requirements, an
  ENVIRONMENT.md stating explicitly what is *not* pinned, and a README with the
  commands to reproduce. Written with a dependency-free ZIP encoder so the tool
  keeps its no-build, no-dependency property.
- Exported scripts now write `environment_lock.txt` at run time, recording the
  resolved package versions, Python build, platform and visible devices.
- Configuration snapshots now carry a schema version, the tool version and the
  seed.
- `tests/ui_config_test.mjs`, `tests/capsule_test.mjs`.

### Security
- **The Gemini API key is no longer written to `localStorage`**, where it
  persisted indefinitely and was readable by any script in the origin. It is
  held in memory for the session, mirrored to `sessionStorage`, and any key
  persisted by an earlier version is deleted on load.
- The key is no longer written back into the input field on load.
- A restrictive Content-Security-Policy has been added, limiting script origins
  and restricting `connect-src` to the model provider's endpoint.
- A regression test asserts that no exported artefact contains the key.

### Changed
- The reproducibility wording in the interface no longer implies that the JSON
  configuration reproduces a run; it stores a specification.

## [2.0.0]
### Changed
- **Renamed** from DeepForge Studio to Layernaut Studio, to avoid confusion
  with the established DeepForge project (Broll et al., 2020). See README.
- **Code generation migrated from the Keras Sequential API to the functional
  API.** Sequential could not express residual connections, parallel branches or
  multi-input layers, which made whole families of architecture unreachable.
- Each architecture now declares a *modality* (image, sequence, text, tabular,
  reconstruction, segmentation) which determines the input shape, the output
  head, the loss and the data pipeline. Previously one image-folder pipeline was
  emitted for every architecture.

### Added
- Skip and residual connections in the Custom Builder: any layer may merge the
  output of an earlier layer with Add or Concatenate.
- Design-time shape tracking. Invalid stacks are now rejected before export with
  an actionable message: Dense applied to a multi-dimensional tensor, a kernel
  larger than the feature map reaching it, a merge between incompatible shapes,
  or an input too small for an architecture's downsampling stages.
- Per-architecture input constraints (minimum size, and divisibility for U-Net).
- `tests/run_all.py` — end-to-end execution matrix: every export is run against
  real data with no edit other than the data path.

### Fixed
- Transformer could not be constructed at all: `MultiHeadAttention` was placed
  inside a `Sequential`, which Keras rejects. It is now a proper encoder with
  residual attention and feed-forward sublayers.
- U-Net had no skip connections, which are the architecture. They are restored.
- The autoencoder had a softmax classifier appended to its decoder, making it
  not an autoencoder. Reconstruction models now emit no class head and train
  against their own input with MSE.
- LSTM, GRU, Autoencoder and U-Net constructed but crashed at `fit()` because
  they were fed images. Each now receives the data it expects.
- The model definition and the data pipeline read the input image size from
  different places, so any non-default size produced a model and a dataset that
  disagreed. There is now one source of truth.
- VGG16-from-scratch used a separate emitter with a hardcoded input size.

## [1.1.0]
### Fixed
- Code generation failed for seven of the ten classical ML models offered in the
  interface (XGBoost, Decision Tree, Naive Bayes, Logistic Regression, K-Means,
  PCA, Linear Regression). Each returned a placeholder comment instead of a
  model, producing a script that ran and trained nothing. All ten now generate.
- `BatchNorm` and `GlobalAvgPool` were offered in the Custom Builder but had no
  case in the generator, so they were silently discarded: a stack containing
  them produced a model that did not contain them. Both now emit code.
- SVM exports with a numeric `gamma` emitted `gamma='0.1'` (quoted), which
  scikit-learn rejects. Numeric values are no longer quoted.
- Logistic Regression could emit solver/penalty combinations scikit-learn
  rejects. `liblinear` was removed (it cannot fit more than two classes) and the
  `l1` penalty now selects `saga`.
- PCA and K-Means could request more components or clusters than the data
  allows; the generated code now clamps both against the input.
- Selecting an ML model with no parameter panel silently opened no modal and
  gave no feedback.

### Added
- Parameter panels for all ten classical ML models.
- Task-aware ML export: unsupervised models (K-Means, PCA) no longer emit a
  train/test split or classification metrics, and regression reports R2/MAE/RMSE
  instead of accuracy.
- `tests/` — a regression suite covering every UI-reachable configuration, every
  builder layer type and all 974 ML parameter combinations. See tests/README.md.

### Changed
- The generator now throws on an unknown layer type instead of dropping it, and
  refuses to export an empty Custom Builder stack. Export handlers surface the
  message rather than downloading an incomplete file.

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
