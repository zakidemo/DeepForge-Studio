import { init, handlers } from './handlers.js';
import { utils } from './utils.js';
import { state } from './state.js';
import { ModelVisualSystem } from './visualizations.js';

// Initialize the application
document.addEventListener('DOMContentLoaded', () => {
    console.log("🚀 DeepForge Studio Starting...");
    init();
});

// Expose necessary objects to the window for inline HTML onclick events
// (Because some of your HTML strings have onclick="DeepForgeStudio.handlers...")
window.DeepForgeStudio = {
    handlers: handlers,
    utils: utils,
    state: state
};

window.ModelVisualSystem = ModelVisualSystem;