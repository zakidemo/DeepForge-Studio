# DeepForge Studio

**DeepForge Studio** is an open-source, client-side web application for rapid deep learning prototyping and reproducible code export.
It provides a visual workflow to select pre-built architectures or build custom layer stacks, configure training hyperparameters, and export a complete runnable training pipeline.

**Exports**
- Python script: `.py`
- Google Colab notebook: `.ipynb`
- Experiment configuration: `.json` (export + import)

> No backend is required for core features. The optional **AI Optimizer** uses the Gemini API and requires a user-provided API key.

## Live Demo (GitHub Pages)
After you enable GitHub Pages for this repository (Settings → Pages), your demo will be available at:
`https://zakidemo.github.io/DeepForge-Studio/`

## Quickstart (local)
Because this project uses ES Modules, you must serve it over HTTP:

```bash
python -m http.server 8000
# open http://localhost:8000
```

## Features
- Visual model gallery (prebuilt) + **Custom Builder**
- For supported models, a modal prompts **From Scratch vs Pretrained**
- Hyperparameter configuration (optimizer, learning rate, batch size, epochs, loss)
- **Reproducible export**: `.py`, `.ipynb`, and `.json` config snapshot
- Optional **AI Optimizer** (Gemini) for transparent suggestions (user-controlled)

## Reproducibility
DeepForge Studio can export a full configuration snapshot (`.json`) and re-import it later to restore the same experiment setup.

Example configs are provided in:
- `examples/configs/`

## AI Optimizer (Gemini)
- Optional feature (the tool works without it)
- Requires a Gemini API key
- Subject to provider quotas/rate limits
- The key is stored locally in your browser storage; use the **Clear key** option when finished

## Folder structure
- `index.html` — app entry point
- `css/` — styles
- `js/` — ES module source code
- `examples/` — sample exported configs/exports
- `.github/workflows/` — CI and Pages deployment

## License
MIT — see [LICENSE](LICENSE)

## Citation
If you use DeepForge Studio in academic work, please use the citation metadata in [CITATION.cff](CITATION.cff).
