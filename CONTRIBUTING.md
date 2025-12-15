# Contributing

Thanks for your interest in contributing to DeepForge Studio!

## Development setup
1. Fork the repository and clone your fork.
2. Run a local server (ES Modules require HTTP):
   ```bash
   python -m http.server 8000
   ```
3. Open http://localhost:8000

## Pull requests
- Keep PRs focused (one feature/fix per PR).
- Update documentation when behavior changes.
- If you add new models/layers, also add:
  - generator support in `js/code-generator.js`
  - UI validation if needed
  - an example config in `examples/configs/`

## Code style
- Prefer small modules and clear naming.
- Avoid introducing frameworks unless justified (project aims to stay lightweight).
