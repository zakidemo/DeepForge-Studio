# Layernaut Studio

**Layernaut Studio** is an open-source, client-side web application for rapid deep learning prototyping and reproducible code export.  
It provides a visual workflow to select pre-built architectures or build custom layer stacks, configure training hyperparameters, and export a complete runnable training pipeline.

## A note on naming

This project is unrelated to, and not derived from, **DeepForge**
(https://github.com/deepforge-dev/deepforge; Broll et al., *Scientific
Programming*, 2020), an established open-source visual environment for deep
learning. An earlier version of this software was distributed under a similar
name; it was renamed to remove any risk of confusion or misattribution. The name **Layernaut** was checked against GitHub, PyPI,
npm and Google Scholar before adoption.

The two systems differ in architecture: DeepForge is a client-server platform
built on WebGME, requiring Node.js, MongoDB and compute workers, and it executes
training on remote infrastructure with built-in version control. Layernaut Studio
is a static client-side bundle with no server, no database and no account, which
exports runnable code rather than executing it.

## Exports
- **Python script**: `.py`
- **Google Colab notebook**: `.ipynb`
- **Experiment configuration**: `.json` (export + import)

> No backend is required for core features. The optional **AI Optimizer** uses the Gemini API and requires a user-provided API key.

## Live Demo (GitHub Pages)
If GitHub Pages is enabled for this repository, the demo is available at:  
https://zakidemo.github.io/Layernaut-Studio/

## Quickstart (local)
Because this project uses ES Modules, you must serve it over HTTP (not `file://`):

```bash
python -m http.server 8000
# open http://localhost:8000
```

## Features
- Visual model gallery (**prebuilt**) + **Custom Builder**
- Drag & drop support in the Builder (reorder layers)
- Edit/reconfigure any previously added layer
- For supported models, a modal prompts **From Scratch** vs **Pretrained**
- Hyperparameter configuration (optimizer, learning rate, batch size, epochs, loss)
- **Reproducible export**: `.py`, `.ipynb`, and `.json` config snapshot
- Optional **AI Optimizer (Gemini)** for transparent suggestions (user-controlled)
- Classical ML templates (e.g., **KNN**) exportable as runnable scripts/notebooks

## Reproducibility
Layernaut Studio can export a full configuration snapshot (`.json`) and re-import it later to restore the same experiment setup.

Examples are provided in:
- `examples/configs/`
- `examples/exports/`

## AI Optimizer (Gemini)
This feature is optional (the tool works without it).

- Requires a Gemini API key
- Subject to provider quotas/rate limits
- The key is stored locally in your browser storage; use the **Clear key** option when finished

## Folder Structure
- `index.html` — app entry point
- `css/` — styles
- `js/` — ES module source code
- `examples/` — sample exported configs/exports
- `.github/workflows/` — CI and Pages deployment

## License
MIT — see [LICENSE](LICENSE)

## Citation
If you use Layernaut Studio in academic work, please use the citation metadata in [CITATION.cff](CITATION.cff).
