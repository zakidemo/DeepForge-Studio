import { state } from './state.js';
import { models } from './config/models.js';
import { layerTypes } from './config/layers.js';
import { mlConfigurations } from './config/ml-config.js';
import { translations } from './config/translations.js';
import { utils } from './utils.js';
import { codeGenerator } from './code-generator.js';
import { geminiOptimizer } from './gemini.js';
import { ModelVisualSystem } from './visualizations.js';

// =====================================================
// HANDLERS OBJECT
// =====================================================
export const handlers = {
    // -------------------------------------------------
    // TAB & NAVIGATION
    // -------------------------------------------------
    switchTab(tabId) {
        document.querySelectorAll('.tab').forEach(tab => {
            tab.classList.remove('active');
            tab.setAttribute('aria-selected', 'false');
        });
        
        document.querySelectorAll('.content').forEach(content => {
            content.classList.remove('active');
        });
        
        const tab = document.getElementById(`${tabId}-tab`);
        const content = document.getElementById(tabId);
        
        if (tab && content) {
            tab.classList.add('active');
            tab.setAttribute('aria-selected', 'true');
            content.classList.add('active');
        }
    },

    switchFeatureMode(mode, originEl = null) {
        // Modes:
        // - Builder mode: prebuilt | custom
        // - Model mode: scratch | pretrained

        if (mode === 'code') {
            utils.notify('Code generation mode has been disabled', 'info');
            return;
        }

        // Scope active-state styling to the nearest .features group
        const scope = originEl?.closest('.features') || document;
        scope.querySelectorAll('.feature-card').forEach(card => card.classList.remove('active'));

        const selectedCard = originEl || scope.querySelector(`[data-mode="${mode}"]`);
        if (selectedCard) selectedCard.classList.add('active');

        // Handle builder mode (prebuilt/custom)
        if (mode === 'prebuilt' || mode === 'custom') {
            const prebuilt = document.getElementById('prebuiltSection');
            const custom = document.getElementById('customBuilderSection');
            if (prebuilt) prebuilt.style.display = 'none';
            if (custom) custom.style.display = 'none';

            if (mode === 'prebuilt') {
                if (prebuilt) prebuilt.style.display = 'block';
                state.currentMode = 'prebuilt';
            } else {
                if (custom) custom.style.display = 'block';
                state.currentMode = 'custom';

                // Switching to Custom Builder should not keep a previously selected prebuilt/ML model
                // (e.g., user selects KNN then switches to Custom Builder).
                state.model = null;
                state.mlConfig = null;
                state._awaitingModeChoice = false;
                state._pendingModelModeModel = null;

                // Clear selection UI + model visuals
                document.querySelectorAll('.model-card').forEach(card => card.classList.remove('selected'));
                const visualContainer = document.getElementById('model-visual-container');
                if (visualContainer) visualContainer.innerHTML = '';

                // Refresh mode hint / pretrained settings visibility
                if (typeof this.updateModelModeUI === 'function') this.updateModelModeUI();
            }

            codeGenerator.generateCode();
            utils.notify(`Switched to ${mode} mode`, 'success');
            return;
        }

        // Handle model mode (scratch/pretrained)
        if (mode === 'scratch' || mode === 'pretrained') {
            state.modelMode = mode;

            const pretrainedOptions = document.getElementById('pretrainedSettings');
            if (pretrainedOptions) pretrainedOptions.style.display = (mode === 'pretrained') ? 'block' : 'none';

            codeGenerator.generateCode();
            utils.notify(`Switched to ${mode} mode`, 'success');
        }
    },

    updateModelModeUI() {
        const modelKey = state.model;
        const modelConfig = modelKey ? models[modelKey] : null;
        const supportsPretrained = !!(modelConfig && modelConfig.type === 'dl' && (modelConfig.supportsPretrained === true || modelConfig.fromScratch === false));
        const awaiting = state._awaitingModeChoice === true;

        const modeHint = document.getElementById('modeHint');
        const modeHintText = document.getElementById('modeHintText');
        const currentLabel = document.getElementById('currentModelModeLabel');

        if (modeHint) modeHint.style.display = supportsPretrained ? 'flex' : 'none';

        const modeLabel = (state.modelMode === 'pretrained') ? 'Mode: Pretrained' : 'Mode: From Scratch';
        if (modeHintText) modeHintText.textContent = awaiting ? 'Choose training mode…' : modeLabel;
        if (currentLabel) currentLabel.textContent = awaiting ? 'Mode: -' : modeLabel;

        const pretrainedCard = document.getElementById('pretrainedOptions');
        const pretrainedSettings = document.getElementById('pretrainedSettings');

        // Only show pretrained settings when in pretrained mode and not awaiting a choice
        if (pretrainedCard) pretrainedCard.style.display = (supportsPretrained && !awaiting && state.modelMode === 'pretrained') ? 'block' : 'none';
        if (pretrainedSettings) pretrainedSettings.style.display = (!awaiting && state.modelMode === 'pretrained') ? 'block' : 'none';
    },


// -------------------------------------------------
// MODEL MODE MODAL (Scratch vs Pretrained)
// -------------------------------------------------
showModelModeModal(modelKey) {
    if (!modelKey) return;
    const modelConfig = models[modelKey];
    const supportsPretrained = !!(modelConfig && modelConfig.type === 'dl' && (modelConfig.supportsPretrained === true || modelConfig.fromScratch === false));
    if (!supportsPretrained) return;

    const modal = document.getElementById('modelModeModal');
    if (!modal) return;

    state._awaitingModeChoice = true;
    state._pendingModelModeModel = modelKey;

    const text = document.getElementById('modelModeModalText');
    if (text) {
        text.textContent = `${modelConfig.name} supports both training from scratch and using pretrained weights. Choose how you want to start.`;
    }

    // Highlight current/default choice inside modal
    const scratch = document.getElementById('chooseScratchBtn');
    const pre = document.getElementById('choosePretrainedBtn');
    if (scratch && pre) {
        scratch.classList.toggle('active', state.modelMode === 'scratch');
        pre.classList.toggle('active', state.modelMode !== 'scratch');
    }

    modal.classList.add('active');
    modal.setAttribute('aria-hidden', 'false');

    this.updateModelModeUI();
},

closeModelModeModal() {
    const modal = document.getElementById('modelModeModal');
    if (!modal) return;
    modal.classList.remove('active');
    modal.setAttribute('aria-hidden', 'true');
},

applyModelMode(mode) {
    if (mode !== 'scratch' && mode !== 'pretrained') return;

    state.modelMode = mode;
    state._awaitingModeChoice = false;

    this.updateModelModeUI();
    this.closeModelModeModal();

    codeGenerator.generateCode();

    const modelConfig = models[state.model];
    if (modelConfig) utils.notify(`Selected: ${modelConfig.name} (${mode === 'pretrained' ? 'pretrained' : 'from scratch'})`, 'success');
},

cancelModelModeModal() {
    // Revert model selection if we stored a previous state
    if (state._modelModePrev && (state._modelModePrev.model || state._modelModePrev.model === null)) {
        const prevModel = state._modelModePrev.model;
        const prevMode = state._modelModePrev.modelMode;

        // Clear selection UI first
        document.querySelectorAll('.model-card').forEach(card => card.classList.remove('selected'));

        state.model = prevModel || null;
        state.modelMode = prevMode || 'scratch';
    }

    state._awaitingModeChoice = false;
    state._pendingModelModeModel = null;

    // Restore selected card UI
    if (state.model) {
        const selectedCard = document.querySelector(`[data-model="${state.model}"]`);
        if (selectedCard) selectedCard.classList.add('selected');

        const visualContainer = document.getElementById('model-visual-container');
        if (visualContainer) {
            visualContainer.innerHTML = ModelVisualSystem.showVisual(state.model);
            ModelVisualSystem.initInteractive(state.model);
        }
    } else {
        const visualContainer = document.getElementById('model-visual-container');
        if (visualContainer) visualContainer.innerHTML = '';
    }

    this.updateModelModeUI();
    this.closeModelModeModal();
    codeGenerator.generateCode();
},

    copyGeneratedCode() {
        const code = document.getElementById('modelCode')?.textContent || '';
        if (!code.trim()) {
            utils.notify('No code to copy yet. Select a model first.', 'info');
            return;
        }

        // Prefer Clipboard API
        if (navigator.clipboard?.writeText) {
            navigator.clipboard.writeText(code)
                .then(() => utils.notify('Copied to clipboard!', 'success'))
                .catch(() => utils.notify('Copy failed. Try selecting and copying manually.', 'warning'));
            return;
        }

        // Fallback
        const ta = document.createElement('textarea');
        ta.value = code;
        ta.style.position = 'fixed';
        ta.style.left = '-9999px';
        document.body.appendChild(ta);
        ta.select();
        try {
            document.execCommand('copy');
            utils.notify('Copied to clipboard!', 'success');
        } catch (_) {
            utils.notify('Copy failed. Try selecting and copying manually.', 'warning');
        } finally {
            ta.remove();
        }
    },

    exportConfig() {
        const config = this.buildConfigObject();
        const blob = new Blob([JSON.stringify(config, null, 2)], { type: 'application/json' });
        utils.downloadBlob(blob, `deepforge_config_${new Date().toISOString().slice(0,10)}.json`);
        utils.notify('Config exported!', 'success');
    },

    async importConfigFromFile(file) {
        if (!file) return;
        try {
            const text = await file.text();
            const config = JSON.parse(text);
            this.applyConfigObject(config);
            utils.notify('Config imported!', 'success');
        } catch (e) {
            console.error(e);
            utils.notify('Failed to import config (invalid JSON).', 'danger');
        }
    },

    buildConfigObject() {
        // Gather UI values (if present)
        const getVal = (id) => document.getElementById(id)?.value ?? null;

        return {
            version: 1,
            exportedAt: new Date().toISOString(),
            model: state.model,
            currentMode: state.currentMode,
            modelMode: state.modelMode,
            freezeLayers: getVal('freezeLayers') ?? state.freezeLayers,
            customTop: getVal('customTop') ?? state.customTop,
            inputSize: getVal('inputSize'),
            numClasses: getVal('numClasses'),
            optimizer: getVal('optimizer'),
            lr: getVal('lr'),
            batchSize: getVal('batchSize'),
            epochs: getVal('epochs'),
            lossFunction: getVal('lossFunction'),
            customLayers: state.customLayers,
            customLayerConfigs: state.customLayerConfigs,
            mlConfig: state.mlConfig,
            language: state.language
        };
    },

    applyConfigObject(config) {
        if (!config || typeof config !== 'object') return;

        // Update state
        if (config.currentMode) state.currentMode = config.currentMode;
        if (config.modelMode) state.modelMode = config.modelMode;
        if (config.language) {
            state.language = config.language;
            localStorage.setItem('deepforge_language', state.language);
        }

        // Apply basic inputs
        const setVal = (id, value) => {
            const el = document.getElementById(id);
            if (el && value !== null && value !== undefined) el.value = value;
        };

        setVal('numClasses', config.numClasses);
        setVal('optimizer', config.optimizer);
        setVal('lr', config.lr);
        setVal('batchSize', config.batchSize);
        setVal('epochs', config.epochs);
        setVal('lossFunction', config.lossFunction);
        setVal('freezeLayers', config.freezeLayers);
        setVal('customTop', config.customTop);
        setVal('inputSize', config.inputSize);

        // Update range display labels, if any
        ['lr','batchSize','epochs','inputSize'].forEach(id => {
            const el = document.getElementById(id);
            const display = document.getElementById(`${id}_value`);
            if (el && display) display.textContent = el.value;
        });

        // Restore custom layers
        if (Array.isArray(config.customLayers)) state.customLayers = config.customLayers.slice();
        if (Array.isArray(config.customLayerConfigs)) state.customLayerConfigs = config.customLayerConfigs.slice();

        // Restore ml config
        if (config.mlConfig) state.mlConfig = config.mlConfig;

        // Builder mode UI
        this.switchFeatureMode(state.currentMode, document.querySelector(`.feature-card[data-mode="${state.currentMode}"]`));

        // Model selection (if exists)
        if (config.model) this.selectModel(config.model, { skipModeModal: true });

        // Model mode UI
        this.updateModelModeUI();

        // Custom layers UI refresh
        this.updateCustomLayers();
        codeGenerator.generateCode();

        // Save config locally too
        localStorage.setItem('deepforge_config', JSON.stringify(this.buildConfigObject()));
    },

    clearApiKey() {
        localStorage.removeItem('gemini_api_key');
        const input = document.getElementById('geminiApiKey');
        if (input) input.value = '';
        utils.notify('API key cleared from this browser.', 'success');
    },

    // -------------------------------------------------
    // MODEL SELECTION & CONFIG
    // -------------------------------------------------
    filterModels(type, clickedButton) {
        const grid = document.getElementById('modelGrid');
        if (!grid) return;

        document.querySelectorAll('.model-filters .btn').forEach(btn => {
            btn.classList.remove('btn-primary');
        });
        if (clickedButton) {
            clickedButton.classList.add('btn-primary');
        }

        const filtered = type === 'all' 
            ? Object.entries(models)
            : Object.entries(models).filter(([, model]) => model.type === type);
        
        const fragment = document.createDocumentFragment();
        
        if (filtered.length === 0) {
            const p = document.createElement('p');
            p.textContent = 'No models match the filter.';
            p.className = 'text-center opacity-75';
            fragment.appendChild(p);
        } else {
            filtered.forEach(([key, model]) => {
                const card = document.createElement('div');
                card.className = 'model-card';
                card.dataset.model = key;
                card.setAttribute('role', 'listitem');
                card.setAttribute('tabindex', '0');
                
                if (model.type === 'ml') {
                    card.innerHTML = `<span class="model-badge">ML</span>`;
                }
                
                card.innerHTML += `
                    <h3><i class="${model.icon}" aria-hidden="true"></i> ${model.name}</h3>
                    <p>${model.desc}</p>
                    <small>${model.type === 'ml' ? 'Classical ML' : `${model.layers || 'N/A'} layers`}</small>
                `;
                fragment.appendChild(card);
            });
        }
        
        grid.innerHTML = '';
        grid.appendChild(fragment);
    },

    selectModel(modelKey, opts = {}) {
        const prev = { model: state.model || null, modelMode: state.modelMode || 'scratch' };

        state.model = modelKey;
        const modelConfig = models[modelKey];

        // Update selected card UI
        document.querySelectorAll('.model-card').forEach(card => card.classList.remove('selected'));
        const selectedCard = document.querySelector(`[data-model="${modelKey}"]`);
        if (selectedCard) selectedCard.classList.add('selected');

        // Show visual explanation early (so user sees something even before choosing mode)
        const visualContainer = document.getElementById('model-visual-container');
        if (visualContainer) {
            visualContainer.innerHTML = ModelVisualSystem.showVisual(modelKey);
            ModelVisualSystem.initInteractive(modelKey);
        }

        // ML models open ML config modal
        if (modelConfig?.type === 'ml') {
            state._awaitingModeChoice = false;
            this.updateModelModeUI();
            this.showMLConfigModal(modelKey);
            codeGenerator.generateCode();
            utils.notify(`Selected: ${modelConfig.name}`);
            return;
        }

        // DL models that support both pretrained and scratch -> ask via modal (no scrolling needed)
        const supportsPretrained = !!(modelConfig && modelConfig.type === 'dl' && (modelConfig.supportsPretrained === true || modelConfig.fromScratch === false));
        if (supportsPretrained && !opts.skipModeModal) {
            state._modelModePrev = prev;
            // Keep user's last choice as default highlight in modal
            if (!state.modelMode) state.modelMode = 'pretrained';
            this.showModelModeModal(modelKey);
            return;
        }

        if (supportsPretrained && opts.skipModeModal) {
            state._awaitingModeChoice = false;
            if (!state.modelMode) state.modelMode = 'pretrained';
            this.updateModelModeUI();
            codeGenerator.generateCode();
            utils.notify(`Selected: ${modelConfig?.name || modelKey}`);
            return;
        }

        // Scratch-only DL models
        state.modelMode = 'scratch';
        state._awaitingModeChoice = false;
        this.updateModelModeUI();

        codeGenerator.generateCode();
        utils.notify(`Selected: ${modelConfig?.name || modelKey}`);
    },

    // -------------------------------------------------
    // ML CONFIGURATION
    // -------------------------------------------------
    showMLConfigModal(modelType) {
        const config = mlConfigurations[modelType];
        if (!config) return;
        
        const modal = document.getElementById('mlConfigModal');
        const title = document.getElementById('mlConfigTitle');
        const form = document.getElementById('mlConfigForm');
        
        title.textContent = config.name + ' Configuration';
        
        // Generate form HTML
        let formHTML = '<div class="param-section">';
        formHTML += '<h4><i class="fas fa-sliders-h"></i> Model Parameters</h4>';
        
        Object.entries(config.params).forEach(([key, param]) => {
            formHTML += `<div class="param-group">`;
            formHTML += `<label for="ml_${key}">${param.label}</label>`;
            
            if (param.type === 'range') {
                formHTML += `
                    <div class="range-display">
                        <input type="range" class="input" id="ml_${key}" 
                               min="${param.min}" max="${param.max}" step="${param.step}" 
                               value="${param.default}" data-key="${key}">
                        <span class="range-value" id="ml_${key}_value">${param.default}</span>
                    </div>`;
            } else if (param.type === 'select') {
                formHTML += `<select class="input" id="ml_${key}">`;
                param.options.forEach(option => {
                    const selected = option === param.default ? 'selected' : '';
                    formHTML += `<option value="${option}" ${selected}>${option}</option>`;
                });
                formHTML += `</select>`;
            } else if (param.type === 'checkbox') {
                const checked = param.default ? 'checked' : '';
                formHTML += `
                    <label style="display: flex; align-items: center; gap: 10px;">
                        <input type="checkbox" id="ml_${key}" ${checked}>
                        <span>Enable</span>
                    </label>`;
            }
            
            formHTML += `<small>${param.description}</small>`;
            formHTML += `</div>`;
        });
        
        formHTML += '</div>';
        
        // Add buttons
        formHTML += `
            <div style="display: flex; justify-content: space-between; margin-top: 30px; gap: 10px;">
                <button class="btn btn-danger" id="closeMLConfigBtn">
                    <i class="fas fa-times"></i> Cancel
                </button>
                <button class="btn btn-info" id="resetMLConfigBtn">
                    <i class="fas fa-undo"></i> Reset to Defaults
                </button>
                <button class="btn btn-success" id="applyMLConfigBtn">
                    <i class="fas fa-check"></i> Apply Configuration
                </button>
            </div>`;
        
        form.innerHTML = formHTML;
        state.mlConfig = { modelType };
        
        modal.classList.add('active');
        modal.setAttribute('aria-hidden', 'false');
    },

    applyMLConfig() {
        const modelType = state.mlConfig?.modelType;
        const config = mlConfigurations[modelType];
        if (!config) return;
        
        state.mlConfig.params = {};
        
        Object.entries(config.params).forEach(([key, param]) => {
            const element = document.getElementById(`ml_${key}`);
            if (element) {
                if (param.type === 'checkbox') {
                    state.mlConfig.params[key] = element.checked;
                } else {
                    state.mlConfig.params[key] = element.value;
                }
            }
        });
        
        state.mlConfig.preprocessing = {
            scaleFeatures: document.getElementById('ml_scale_features')?.checked || true,
            testSize: document.getElementById('ml_test_size')?.value || 0.2,
            randomState: document.getElementById('ml_random_state')?.value || 42
        };
        
        codeGenerator.generateCode();
        this.closeMLConfig();
        utils.notify(`${config.name} configuration applied`, 'success');
    },

    closeMLConfig() {
        const modal = document.getElementById('mlConfigModal');
        modal.classList.remove('active');
        modal.setAttribute('aria-hidden', 'true');
    },

    // -------------------------------------------------
    // CUSTOM LAYER BUILDER
    // -------------------------------------------------
    addLayer(type) {
        const configModals = {
            'Conv2D': this.showConv2DConfig,
            'Dense': this.showDenseConfig,
            'Dropout': this.showDropoutConfig,
            'MaxPool': this.showMaxPoolConfig,
            'AvgPool': this.showAvgPoolConfig
        };
        
        if (configModals[type]) {
            state.pendingLayerType = type;
            configModals[type](); 
        } else {
            state.customLayers.push(type);
            state.customLayerConfigs.push({});
            this.updateCustomLayers();
            codeGenerator.generateCode();
        }
    },

    showConv2DConfig() {
        const modal = document.getElementById('layerConfigModal');
        const title = document.getElementById('layerConfigTitle');
        const form = document.getElementById('layerConfigForm');
        
        title.innerHTML = '<i class="fas fa-cog"></i> Configure Conv2D Layer';
        
        form.innerHTML = `
            <label>Number of Filters</label>
            <input type="number" class="input" id="conv_filters" value="32" min="1" max="512">
            
            <label>Kernel Size</label>
            <select class="input" id="conv_kernel">
                <option value="3">3x3</option>
                <option value="5">5x5</option>
                <option value="7">7x7</option>
           </select>
           
           <label>Stride</label>
           <input type="number" class="input" id="conv_stride" value="1" min="1" max="5">
           
           <label>Padding</label>
           <select class="input" id="conv_padding">
               <option value="valid">Valid</option>
               <option value="same">Same</option>
           </select>
           
           <label>Activation Function</label>
           <select class="input" id="conv_activation">
               <option value="relu">ReLU</option>
               <option value="sigmoid">Sigmoid</option>
               <option value="tanh">Tanh</option>
               <option value="linear">Linear</option>
           </select>
           
           <div style="display: flex; justify-content: space-between; margin-top: 20px;">
               <button class="btn btn-danger" id="cancelLayerConfigBtn"><i class="fas fa-times"></i> Cancel</button>
               <button class="btn btn-success" id="applyConv2DConfigBtn"><i class="fas fa-check"></i> Add Layer</button>
           </div>
       `;
       
       modal.classList.add('active');
       modal.setAttribute('aria-hidden', 'false');
    },

    showDenseConfig() {
        const modal = document.getElementById('layerConfigModal');
        const title = document.getElementById('layerConfigTitle');
        const form = document.getElementById('layerConfigForm');
        
        title.innerHTML = '<i class="fas fa-cog"></i> Configure Dense Layer';
        
        form.innerHTML = `
            <label>Number of Units</label>
            <input type="number" class="input" id="dense_units" value="128" min="1" max="4096">
            <label>Activation Function</label>
            <select class="input" id="dense_activation">
                <option value="relu">ReLU</option>
                <option value="sigmoid">Sigmoid</option>
                <option value="tanh">Tanh</option>
                <option value="softmax">Softmax</option>
                <option value="linear">Linear</option>
            </select>
            <label>Use Bias</label>
            <select class="input" id="dense_bias">
                <option value="true">Yes</option>
                <option value="false">No</option>
            </select>
            <div style="display: flex; justify-content: space-between; margin-top: 20px;">
                <button class="btn btn-danger" id="cancelLayerConfigBtn"><i class="fas fa-times"></i> Cancel</button>
                <button class="btn btn-success" id="applyDenseConfigBtn"><i class="fas fa-check"></i> Add Layer</button>
            </div>
        `;
        modal.classList.add('active');
        modal.setAttribute('aria-hidden', 'false');
    },

    showDropoutConfig() {
        const modal = document.getElementById('layerConfigModal');
        const title = document.getElementById('layerConfigTitle');
        const form = document.getElementById('layerConfigForm');
        title.innerHTML = '<i class="fas fa-cog"></i> Configure Dropout Layer';
        form.innerHTML = `
            <label>Dropout Rate</label>
            <input type="number" class="input" id="dropout_rate" value="0.5" min="0" max="0.9" step="0.1">
            <div style="display: flex; justify-content: space-between; margin-top: 20px;">
                <button class="btn btn-danger" id="cancelLayerConfigBtn"><i class="fas fa-times"></i> Cancel</button>
                <button class="btn btn-success" id="applyDropoutConfigBtn"><i class="fas fa-check"></i> Add Layer</button>
            </div>
        `;
        modal.classList.add('active');
        modal.setAttribute('aria-hidden', 'false');
    },

    showMaxPoolConfig() {
        const modal = document.getElementById('layerConfigModal');
        const title = document.getElementById('layerConfigTitle');
        const form = document.getElementById('layerConfigForm');
        title.innerHTML = '<i class="fas fa-cog"></i> Configure MaxPooling2D Layer';
        form.innerHTML = `
            <label>Pool Size</label>
            <select class="input" id="maxpool_size">
                <option value="2">2x2</option>
                <option value="3">3x3</option>
                <option value="4">4x4</option>
                <option value="5">5x5</option>
            </select>
            <label>Stride</label>
            <input type="number" class="input" id="maxpool_stride" value="2" min="1" max="5">
            <label>Padding</label>
            <select class="input" id="maxpool_padding">
                <option value="valid">Valid</option>
                <option value="same">Same</option>
            </select>
            <div style="display: flex; justify-content: space-between; margin-top: 20px;">
                <button class="btn btn-danger" id="cancelLayerConfigBtn"><i class="fas fa-times"></i> Cancel</button>
                <button class="btn btn-success" id="applyMaxPoolConfigBtn"><i class="fas fa-check"></i> Add Layer</button>
            </div>
        `;
        modal.classList.add('active');
        modal.setAttribute('aria-hidden', 'false');
    },

    showAvgPoolConfig() {
        const modal = document.getElementById('layerConfigModal');
        const title = document.getElementById('layerConfigTitle');
        const form = document.getElementById('layerConfigForm');
        title.innerHTML = '<i class="fas fa-cog"></i> Configure AveragePooling2D Layer';
        form.innerHTML = `
            <label>Pool Size</label>
            <select class="input" id="avgpool_size">
                <option value="2">2x2</option>
                <option value="3">3x3</option>
                <option value="4">4x4</option>
                <option value="5">5x5</option>
            </select>
            <label>Stride</label>
            <input type="number" class="input" id="avgpool_stride" value="2" min="1" max="5">
            <label>Padding</label>
            <select class="input" id="avgpool_padding">
                <option value="valid">Valid</option>
                <option value="same">Same</option>
            </select>
            <div style="display: flex; justify-content: space-between; margin-top: 20px;">
                <button class="btn btn-danger" id="cancelLayerConfigBtn"><i class="fas fa-times"></i> Cancel</button>
                <button class="btn btn-success" id="applyAvgPoolConfigBtn"><i class="fas fa-check"></i> Add Layer</button>
            </div>
        `;
        modal.classList.add('active');
        modal.setAttribute('aria-hidden', 'false');
    },

    applyLayerConfig(type, config) {
        state.customLayers.push(type);
        state.customLayerConfigs.push(config);
        
        this.updateCustomLayers();
        codeGenerator.generateCode();
        this.closeLayerConfig();
        
        utils.notify(`${type} layer added`);
    },

    closeLayerConfig() {
        const modal = document.getElementById('layerConfigModal');
        modal.classList.remove('active');
        modal.setAttribute('aria-hidden', 'true');
        state.pendingLayerType = null;
    },

    updateCustomLayers() {
        const layersDiv = document.getElementById('layers');
        if (!layersDiv) return;
        
        if (state.customLayers.length === 0) {
            layersDiv.innerHTML = '<p class="text-center opacity-75">Drag and drop layers here or use the buttons above</p>';
        } else {
            layersDiv.innerHTML = state.customLayers.map((layer, i) => {
                const config = state.customLayerConfigs[i] || {};
                let configText = '';
                
                if (layer === 'Conv2D') configText = ` (${config.filters || 32} filters, ${config.kernel_size || 3}x${config.kernel_size || 3})`;
                else if (layer === 'Dense') configText = ` (${config.units || 128} units)`;
                else if (layer === 'Dropout') configText = ` (${config.rate || 0.5})`;
                else if (layer === 'MaxPool' || layer === 'AvgPool') configText = ` (${config.pool_size || 2}x${config.pool_size || 2})`;
                
                const layerObj = layerTypes.find(l => l.type === layer);
                const icon = layerObj ? layerObj.icon : 'fas fa-cube';
                
                return `<span class="layer-block">
                    <i class="${icon}"></i> 
                    ${layer}${configText}
                    <span data-remove="${i}" style="cursor:pointer; color:#ff5555;">
                        <i class="fas fa-times"></i>
                    </span>
                </span>`;
            }).join('');
        }
    },

    removeLayer(index) {
        state.customLayers.splice(index, 1);
        state.customLayerConfigs.splice(index, 1);
        this.updateCustomLayers();
        codeGenerator.generateCode();
        utils.notify('Layer removed');
    },

    clearLayers() {
        state.customLayers = [];
        state.customLayerConfigs = [];
        this.updateCustomLayers();
        codeGenerator.generateCode();
        utils.notify('All layers cleared');
    },

    // -------------------------------------------------
    // AI OPTIMIZER
    // -------------------------------------------------
    async activateAI() {
        const apiKey = document.getElementById('geminiApiKey')?.value.trim();
        
        if (!apiKey) {
            utils.notify('Please enter your API key first', 'warning');
            utils.shakeElement('geminiApiKey');
            return;
        }
        
        const activateBtn = document.getElementById('activate-btn');
        const originalContent = activateBtn.innerHTML;
        
        activateBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Connecting...';
        activateBtn.disabled = true;
        
        try {
            geminiOptimizer.setApiKey(apiKey);
            const testResponse = await geminiOptimizer.makeRequest('Respond with "Connected" to confirm API is working.', 100);
            
            if (testResponse && testResponse.toLowerCase().includes('connected')) {
                state.aiOptimizerState.isConnected = true;
                document.getElementById('status-indicator').classList.add('connected');
                document.getElementById('status-text').textContent = 'Connected';
                
                this.updateWizardStep(2);
                utils.notify('🎉 AI Optimizer activated successfully!', 'success');
                localStorage.setItem('gemini_api_key', apiKey);
            } else {
                throw new Error('Connection test failed');
            }
        } catch (error) {
            utils.notify(`Connection failed: ${error.message}`, 'error');
            activateBtn.innerHTML = originalContent;
            activateBtn.disabled = false;
        }
    },

    updateWizardStep(step) {
        document.querySelectorAll('.wizard-step').forEach(s => s.classList.remove('active'));
        document.getElementById(`step-${step}`).classList.add('active');
        document.querySelectorAll('.optimizer-step').forEach(s => s.style.display = 'none');
        
        switch(step) {
            case 1: document.getElementById('api-setup-step').style.display = 'block'; break;
            case 2: document.getElementById('analysis-options-step').style.display = 'block'; break;
            case 3: document.getElementById('results-step').style.display = 'block'; break;
        }
        state.aiOptimizerState.currentStep = step;
    },

    async generateAIRecommendation() {
        const problemType = document.getElementById('problemType')?.value;
        const problemDescription = document.getElementById('problemDescription')?.value;
        const datasetSize = document.getElementById('datasetSize')?.value;
        const numFeatures = document.getElementById('numFeatures')?.value;
        const primaryGoal = document.getElementById('primaryGoal')?.value;
        const computeResources = document.getElementById('computeResources')?.value;
        const deploymentEnv = document.getElementById('deploymentEnv')?.value;
        
        if (!problemType) { utils.notify('Please select a problem type', 'warning'); utils.shakeElement('problemType'); return; }
        if (!problemDescription) { utils.notify('Please describe your problem', 'warning'); utils.shakeElement('problemDescription'); return; }
        if (!state.aiOptimizerState.isConnected) { utils.notify('Please connect to AI Optimizer first', 'warning'); this.switchTab('ai-optimizer'); return; }
        
        utils.showLoadingOverlay('🤖 AI is analyzing your requirements...');
        
        try {
            const dataTypes = [];
            if (document.getElementById('hasNumerical')?.checked) dataTypes.push('Numerical');
            if (document.getElementById('hasCategorical')?.checked) dataTypes.push('Categorical');
            if (document.getElementById('hasText')?.checked) dataTypes.push('Text');
            if (document.getElementById('hasImages')?.checked) dataTypes.push('Images');
            if (document.getElementById('hasAudio')?.checked) dataTypes.push('Audio');
            if (document.getElementById('hasDateTime')?.checked) dataTypes.push('DateTime');
            
            const prompt = `As an expert ML engineer, provide a comprehensive recommendation for this problem... (Your prompt here) ...`;
            
            const response = await geminiOptimizer.makeRequest(prompt, 4000);
            state.aiOptimizerState.lastRecommendations = response;
            
            this.displayProfessionalRecommendation(response, { problemType, problemDescription, datasetSize, dataTypes });
            utils.hideLoadingOverlay();
            if (typeof confetti !== 'undefined') confetti({ particleCount: 100, spread: 70, origin: { y: 0.6 } });
            utils.notify('✅ AI Model Recommendation Generated Successfully!', 'success');
            
        } catch (error) {
            utils.hideLoadingOverlay();
            utils.notify('Error: ' + error.message, 'error');
        }
    },

    displayProfessionalRecommendation(response, metadata) {
        const sections = this.parseRecommendationSections(response);
        
        const resultsHTML = `
            <div class="ai-recommendation-results" style="margin-top: 30px;">
                <div class="recommendation-header" style="background: linear-gradient(135deg, #667eea, #764ba2); padding: 30px; border-radius: 20px 20px 0 0; color: white;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <h2 style="margin: 0; color: white; font-size: 2rem;"><i class="fas fa-brain"></i> AI Model Recommendation</h2>
                            <p style="margin: 10px 0 0 0; opacity: 0.9;">Personalized solution for your ${metadata.problemType} problem</p>
                        </div>
                        <div style="text-align: right;">
                            <div class="stat-badge" style="background: rgba(255,255,255,0.2); padding: 8px 15px; border-radius: 20px; display: inline-block; margin: 5px;">
                                <i class="fas fa-database"></i> ${metadata.datasetSize.replace('-', ' ')}
                            </div>
                            <div class="stat-badge" style="background: rgba(255,255,255,0.2); padding: 8px 15px; border-radius: 20px; display: inline-block; margin: 5px;">
                                <i class="fas fa-layer-group"></i> ${metadata.dataTypes.join(', ')}
                            </div>
                        </div>
                    </div>
                </div>
                
                <div style="background: rgba(30, 30, 46, 0.7); border-radius: 0 0 20px 20px; overflow: hidden;">
                    <div style="background: rgba(80, 250, 123, 0.1); border-left: 4px solid #50fa7b; padding: 20px 30px; margin: 0;">
                        <h3 style="color: #50fa7b; margin: 0 0 10px 0;"><i class="fas fa-star"></i> Quick Summary</h3>
                        <div id="quick-summary" style="color: rgba(255,255,255,0.9);">
                            ${this.extractQuickSummary(response)}
                        </div>
                    </div>
                    
                    <div style="padding: 20px;">
                        <div class="recommendation-tabs" style="display: flex; gap: 10px; margin-bottom: 20px; border-bottom: 2px solid rgba(255,255,255,0.1); padding-bottom: 10px;">
                            <button class="rec-tab active" onclick="DeepForgeStudio.handlers.switchRecommendationTab('overview')" data-tab="overview" style="background: linear-gradient(135deg, #667eea, #764ba2); color: white; border: none; padding: 10px 20px; border-radius: 10px; cursor: pointer; font-weight: 600;"><i class="fas fa-chart-pie"></i> Overview</button>
                            <button class="rec-tab" onclick="DeepForgeStudio.handlers.switchRecommendationTab('architecture')" data-tab="architecture" style="background: rgba(255,255,255,0.1); color: white; border: none; padding: 10px 20px; border-radius: 10px; cursor: pointer; font-weight: 600;"><i class="fas fa-project-diagram"></i> Architecture</button>
                            <button class="rec-tab" onclick="DeepForgeStudio.handlers.switchRecommendationTab('code')" data-tab="code" style="background: rgba(255,255,255,0.1); color: white; border: none; padding: 10px 20px; border-radius: 10px; cursor: pointer; font-weight: 600;"><i class="fas fa-code"></i> Code</button>
                            <button class="rec-tab" onclick="DeepForgeStudio.handlers.switchRecommendationTab('training')" data-tab="training" style="background: rgba(255,255,255,0.1); color: white; border: none; padding: 10px 20px; border-radius: 10px; cursor: pointer; font-weight: 600;"><i class="fas fa-graduation-cap"></i> Training</button>
                        </div>
                        
                        <div class="tab-contents">
                            <div id="overview-content" class="rec-tab-content active">${this.formatOverviewTab(sections)}</div>
                            <div id="architecture-content" class="rec-tab-content" style="display: none;">${this.formatArchitectureTab(sections)}</div>
                            <div id="code-content" class="rec-tab-content" style="display: none;">${this.formatCodeTab(sections, response)}</div>
                            <div id="training-content" class="rec-tab-content" style="display: none;">${this.formatTrainingTab(sections)}</div>
                        </div>
                    </div>
                    
                    <div style="background: rgba(40, 42, 54, 0.5); padding: 20px; text-align: center; border-top: 1px solid rgba(255,255,255,0.1);">
                        <button class="btn btn-success btn-large" onclick="DeepForgeStudio.handlers.applyRecommendedSettings()" style="margin: 10px;"><i class="fas fa-magic"></i> Apply This Configuration</button>
                        <button class="btn btn-info btn-large" onclick="DeepForgeStudio.handlers.exportRecommendation()" style="margin: 10px;"><i class="fas fa-download"></i> Export Report</button>
                        <button class="btn btn-warning" onclick="DeepForgeStudio.handlers.modifyRecommendation()" style="margin: 10px;"><i class="fas fa-edit"></i> Modify Parameters</button>
                    </div>
                </div>
            </div>
        `;
        
        let resultsArea = document.getElementById('recommendation-results-area');
        if (!resultsArea) {
            const form = document.querySelector('.recommendation-form');
            if (form) {
                resultsArea = document.createElement('div');
                resultsArea.id = 'recommendation-results-area';
                form.parentElement.appendChild(resultsArea);
            }
        }
        
        if (resultsArea) {
            resultsArea.innerHTML = resultsHTML;
            resultsArea.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }
    },

    // -------------------------------------------------
    // RECOMMENDATION HELPERS
    // -------------------------------------------------
    parseRecommendationSections(response) {
        const sections = {};
        const sectionRegex = /##\s*\d*\.?\s*([A-Z\s]+)\n([\s\S]*?)(?=##|\n$)/gi;
        let match;
        while ((match = sectionRegex.exec(response)) !== null) {
            const title = match[1].trim();
            const content = match[2].trim();
            sections[title.toLowerCase().replace(/\s+/g, '_')] = { title, content };
        }
        return sections;
    },

    extractQuickSummary(response) {
        const modelMatch = response.match(/Model Name:\s*([^\n]+)/i);
        const accuracyMatch = response.match(/Accuracy.*?:\s*([^\n]+)/i);
        const epochsMatch = response.match(/Epochs:\s*([^\n]+)/i);
        const batchMatch = response.match(/Batch Size:\s*([^\n]+)/i);
        
        return `
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px;">
                <div><strong style="color: #8be9fd;">Recommended Model:</strong><br>${modelMatch ? modelMatch[1] : 'See details below'}</div>
                <div><strong style="color: #8be9fd;">Expected Accuracy:</strong><br>${accuracyMatch ? accuracyMatch[1] : '85-95%'}</div>
                <div><strong style="color: #8be9fd;">Training Epochs:</strong><br>${epochsMatch ? epochsMatch[1] : '20-50'}</div>
            </div>
        `;
    },

    formatOverviewTab(sections) {
        return `
            <div class="overview-grid" style="display: grid; gap: 20px;">
                ${Object.entries(sections).slice(0, 4).map(([key, section]) => `
                    <div style="background: rgba(40, 42, 54, 0.5); padding: 20px; border-radius: 15px; border-left: 3px solid #667eea;">
                        <h4 style="color: #8be9fd; margin: 0 0 15px 0;"><i class="fas fa-chevron-right"></i> ${section.title}</h4>
                        <div style="color: rgba(255,255,255,0.85); line-height: 1.8;">${this.formatContent(section.content)}</div>
                    </div>
                `).join('')}
            </div>
        `;
    },

    formatArchitectureTab(sections) {
        const archSection = sections['architecture_details'] || sections['model_architecture'] || {};
        return `
            <div style="background: rgba(40, 42, 54, 0.5); padding: 25px; border-radius: 15px;">
                <h3 style="color: #50fa7b; margin: 0 0 20px 0;"><i class="fas fa-sitemap"></i> Model Architecture Details</h3>
                <div style="color: rgba(255,255,255,0.9); line-height: 1.8;">
                    ${this.formatContent(archSection.content || 'Architecture details will be shown here')}
                </div>
            </div>
        `;
    },

    formatCodeTab(sections, response) {
        const codeMatch = response.match(/```python([\s\S]*?)```/);
        const code = codeMatch ? codeMatch[1] : 'No code provided';
        return `
            <div style="position: relative;">
                <div style="background: linear-gradient(135deg, #44475a, #6272a4); padding: 10px 20px; border-radius: 15px 15px 0 0; display: flex; justify-content: space-between; align-items: center;">
                    <span style="color: #50fa7b; font-weight: 600;"><i class="fas fa-code"></i> Implementation Code</span>
                    <button onclick="DeepForgeStudio.utils.copyToClipboard(\`${code.replace(/`/g, '\\`')}\`)" style="background: rgba(255,255,255,0.1); border: none; color: white; padding: 5px 15px; border-radius: 5px; cursor: pointer;"><i class="fas fa-copy"></i> Copy Code</button>
                </div>
                <pre style="background: #282a36; color: #f8f8f2; padding: 20px; margin: 0; border-radius: 0 0 15px 15px; overflow-x: auto; max-height: 500px;"><code>${this.highlightPythonCode(code)}</code></pre>
            </div>
        `;
    },

    formatTrainingTab(sections) {
        const trainingSection = sections['training_strategy'] || sections['training'] || {};
        return `
            <div style="display: grid; gap: 20px;">
                <div style="background: rgba(40, 42, 54, 0.5); padding: 20px; border-radius: 15px;">
                    <h4 style="color: #ffb86c; margin: 0 0 15px 0;"><i class="fas fa-cogs"></i> Training Configuration</h4>
                    ${this.formatContent(trainingSection.content || 'Training details here')}
                </div>
            </div>
        `;
    },

    formatContent(content) {
        return content
            .replace(/\*\*(.*?)\*\*/g, '<strong style="color: #ffb86c;">$1</strong>')
            .replace(/\*(.*?)\*/g, '<em style="color: #bd93f9;">$1</em>')
            .replace(/^-\s+(.*?)$/gm, '<li style="margin: 8px 0;">$1</li>')
            .replace(/(<li.*<\/li>\n?)+/g, '<ul style="list-style: none; padding-left: 20px;">$&</ul>')
            .replace(/\n\n/g, '</p><p style="margin: 10px 0;">')
            .replace(/^/, '<p style="margin: 10px 0;">')
            .replace(/$/, '</p>');
    },

    highlightPythonCode(code) {
        return code
            .replace(/\b(import|from|def|class|if|else|elif|for|while|return|try|except|with|as|in|is|not|and|or)\b/g, '<span style="color: #ff79c6;">$1</span>')
            .replace(/\b(True|False|None)\b/g, '<span style="color: #bd93f9;">$1</span>')
            .replace(/(#.*$)/gm, '<span style="color: #6272a4; font-style: italic;">$1</span>')
            .replace(/(['"])(.*?)\1/g, '<span style="color: #f1fa8c;">$1$2$1</span>')
            .replace(/\b(\d+)\b/g, '<span style="color: #bd93f9;">$1</span>');
    },

    switchRecommendationTab(tabName) {
        document.querySelectorAll('.rec-tab').forEach(tab => {
            tab.classList.remove('active');
            tab.style.background = 'rgba(255,255,255,0.1)';
        });
        const activeTab = document.querySelector(`[data-tab="${tabName}"]`);
        if (activeTab) {
            activeTab.classList.add('active');
            activeTab.style.background = 'linear-gradient(135deg, #667eea, #764ba2)';
        }
        document.querySelectorAll('.rec-tab-content').forEach(content => content.style.display = 'none');
        const activeContent = document.getElementById(`${tabName}-content`);
        if (activeContent) activeContent.style.display = 'block';
    },

    resetRecommendationForm() {
        document.getElementById('problemType').value = '';
        document.getElementById('problemDescription').value = '';
        document.getElementById('numFeatures').value = '';
        const resultsArea = document.getElementById('recommendation-results-area');
        if (resultsArea) resultsArea.innerHTML = '';
        utils.notify('Form reset!', 'info');
    },

    applyRecommendedSettings() {
        utils.notify('Applying recommended settings to your model...', 'info');
        this.switchTab('architecture');
    },

    exportRecommendation() {
        const recommendation = state.aiOptimizerState.lastRecommendations;
        if (recommendation) {
            const blob = new Blob([recommendation], {type: 'text/plain'});
            utils.downloadBlob(blob, 'AI_Model_Recommendation.txt');
            utils.notify('Recommendation exported successfully!', 'success');
        }
    },

    modifyRecommendation() {
        utils.notify('Feature coming soon: Manual parameter tweaking.', 'info');
    },

    // -------------------------------------------------
    // AI ANALYSIS (RESULTS)
    // -------------------------------------------------
    async startAnalysis(type) {
        const analysisTypes = {
            'recommendations': { title: 'Smart Recommendations', method: 'getRecommendations', icon: 'fas fa-lightbulb' },
            'hyperparameters': { title: 'Hyperparameter Optimization', method: 'optimizeHyperparameters', icon: 'fas fa-sliders-h' },
            'architecture': { title: 'Architecture Analysis', method: 'analyzeArchitecture', icon: 'fas fa-microscope' },
            'practices': { title: 'Best Practices', method: 'getBestPractices', icon: 'fas fa-trophy' }
        };
        
        const config = analysisTypes[type];
        if (!config) return;
        
        state.aiOptimizerState.currentAnalysis = type;
        utils.showLoadingOverlay(config.title);
        
        try {
            const response = await geminiOptimizer[config.method]();
            this.displayAnalysisResults(response, config);
            state.aiOptimizerState.analysisHistory.push({ type: type, timestamp: new Date(), results: response });
            this.updateWizardStep(3);
        } catch (error) {
            utils.hideLoadingOverlay();
            utils.notify(`Analysis failed: ${error.message}`, 'error');
        }
    },

    displayAnalysisResults(response, config) {
        utils.hideLoadingOverlay();
        document.getElementById('results-title').innerHTML = `<i class="${config.icon}"></i> ${config.title} Results`;
        const parsed = this.parseAIResponse(response);
        this.populateSummaryTab(parsed);
        document.getElementById('detailed-analysis').innerHTML = this.formatDetailedResponse(response);
        this.populateCodeTab(parsed);
    },

    parseAIResponse(response) {
        const parsed = { insights: [], quickWins: [], warnings: [], recommendations: [], code: [], metrics: {} };
        const lines = response.split('\n');
        
        lines.forEach(line => {
            const trimmed = line.trim();
            if (trimmed.startsWith('🎯') || trimmed.includes('Insight:')) parsed.insights.push(trimmed.replace(/^[🎯]*\s*Insight:\s*/i, ''));
            else if (trimmed.startsWith('⚡') || trimmed.includes('Quick win:')) parsed.quickWins.push(trimmed.replace(/^[⚡]*\s*Quick win:\s*/i, ''));
            else if (trimmed.startsWith('⚠️') || trimmed.includes('Warning:')) parsed.warnings.push(trimmed.replace(/^[⚠️]*\s*Warning:\s*/i, ''));
            else if (trimmed.startsWith('- ') || trimmed.startsWith('• ')) parsed.recommendations.push(trimmed.substring(2));
        });
        
        const codeRegex = /```(\w+)?\n?([\s\S]*?)```/g;
        let match;
        while ((match = codeRegex.exec(response)) !== null) {
            parsed.code.push({ language: match[1] || 'python', content: match[2].trim() });
        }
        return parsed;
    },

    populateSummaryTab(parsed) {
        const insightsHtml = parsed.insights.length > 0
            ? parsed.insights.map(insight => `<div class="insight-item"><i class="fas fa-lightbulb"></i><span>${insight}</span></div>`).join('')
            : '<div class="insight-item"><i class="fas fa-info-circle"></i> No specific insights found</div>';
        document.getElementById('key-insights').innerHTML = insightsHtml;
        
        const quickWinsHtml = parsed.quickWins.length > 0
            ? parsed.quickWins.map(win => `<div class="insight-item"><i class="fas fa-bolt"></i><span>${win}</span></div>`).join('')
            : '<div class="insight-item"><i class="fas fa-check"></i> Your model is well-optimized</div>';
        document.getElementById('quick-wins').innerHTML = quickWinsHtml;
        
        const warningsHtml = parsed.warnings.length > 0
            ? parsed.warnings.map(warning => `<div class="insight-item"><i class="fas fa-exclamation-triangle" style="color: #ffb86c;"></i><span>${warning}</span></div>`).join('')
            : '<div class="insight-item"><i class="fas fa-shield-alt" style="color: #50fa7b;"></i> No issues detected</div>';
        document.getElementById('warnings').innerHTML = warningsHtml;
    },

    populateCodeTab(parsed) {
        const codeElement = document.getElementById('recommended-code');
        if (!codeElement) return;
        
        if (parsed.code.length > 0) {
            codeElement.innerHTML = parsed.code.map((block, index) => `
                <div class="code-block-container">
                    <div class="code-header">
                        <span class="language-tag">${block.language.toUpperCase()}</span>
                        <button class="copy-code-btn" data-copy-index="${index}"><i class="fas fa-copy"></i> Copy</button>
                    </div>
                    <pre><code class="language-${block.language}">${utils.escapeHTML(block.content)}</code></pre>
                </div>
            `).join('');
        } else {
            codeElement.innerHTML = '<p style="text-align: center; color: rgba(255,255,255,0.5);">No code recommendations for this analysis</p>';
        }
    },

    formatDetailedResponse(response) {
        return response
            .replace(/##\s*(.*?)$/gm, '<h3 style="color: #8be9fd; margin: 20px 0;">$1</h3>')
            .replace(/\*\*(.*?)\*\*/g, '<strong style="color: #ffb86c;">$1</strong>')
            .replace(/\*(.*?)\*/g, '<em style="color: #bd93f9;">$1</em>')
            .replace(/```(\w+)?\n?([\s\S]*?)```/g, '<pre style="background: #1e1e2e; padding: 15px; border-radius: 10px;"><code>$2</code></pre>')
            .replace(/\n\n/g, '</p><p style="margin: 15px 0; line-height: 1.6;">')
            .replace(/^/, '<p style="margin: 15px 0; line-height: 1.6;">')
            .replace(/$/, '</p>');
    },

    applyRecommendations() {
        const confirmApply = confirm('This will update your model configuration. Continue?');
        if (!confirmApply) return;
        
        const applyBtn = document.getElementById('applyRecommendationsBtn');
        const originalContent = applyBtn.innerHTML;
        applyBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Applying...';
        applyBtn.disabled = true;
        
        try {
            const lastAnalysis = state.aiOptimizerState.analysisHistory[state.aiOptimizerState.analysisHistory.length - 1];
            if (lastAnalysis && lastAnalysis.results) {
                const params = this.extractParametersFromResponse(lastAnalysis.results);
                let appliedCount = 0;
                Object.entries(params).forEach(([key, value]) => {
                    const element = document.getElementById(key);
                    if (element && value !== null) {
                        element.value = value;
                        appliedCount++;
                        utils.highlightElement(element);
                    }
                });
                codeGenerator.generateCode();
                setTimeout(() => {
                    applyBtn.innerHTML = '<i class="fas fa-check"></i> Applied Successfully!';
                    applyBtn.style.background = 'linear-gradient(135deg, #11998e, #38ef7d)';
                    utils.notify(`✅ Applied ${appliedCount} recommendations to your model!`, 'success');
                    setTimeout(() => {
                        applyBtn.innerHTML = originalContent;
                        applyBtn.disabled = false;
                        applyBtn.style.background = '';
                    }, 3000);
                }, 1000);
            }
        } catch (error) {
            applyBtn.innerHTML = originalContent;
            applyBtn.disabled = false;
            utils.notify('Failed to apply recommendations', 'error');
        }
    },

    extractParametersFromResponse(response) {
        const params = {};
        const patterns = {
            lr: /learning\s*rate[:\s]+(\d+\.?\d*(?:e-?\d+)?)/i,
            batchSize: /batch\s*size[:\s]+(\d+)/i,
            epochs: /epochs?[:\s]+(\d+)/i,
            dropout: /dropout[:\s]+(\d+\.?\d*)/i,
            optimizer: /optimizer[:\s]+["']?(\w+)["']?/i
        };
        Object.entries(patterns).forEach(([param, pattern]) => {
            const match = response.match(pattern);
            if (match) params[param] = match[1];
        });
        return params;
    },

    // -------------------------------------------------
    // AI CHAT & FOLLOW UP
    // -------------------------------------------------
    async askAI() {
        const question = document.getElementById('ai-question')?.value.trim();
        if (!question) { utils.shakeElement('ai-question'); return; }
        await this.askQuestion(question);
        document.getElementById('ai-question').value = '';
    },

    async askQuestion(question) {
        utils.showLoadingOverlay('Thinking about your question...');
        try {
            const response = await geminiOptimizer.askFollowUp(question);
            this.displayAnalysisResults(response, { title: 'AI Response', icon: 'fas fa-comments' });
            this.updateWizardStep(3);
        } catch (error) {
            utils.hideLoadingOverlay();
            utils.notify('Failed to get response', 'error');
        }
    },

    async askFollowup() {
        const question = document.getElementById('followup-question')?.value.trim();
        if (!question) { utils.shakeElement('followup-question'); return; }
        await this.askQuestion(question);
        document.getElementById('followup-question').value = '';
        this.closeFollowUpModal();
    },

    closeFollowUpModal() {
        const modal = document.getElementById('followUpModal');
        modal.classList.remove('active');
        modal.setAttribute('aria-hidden', 'true');
    },

    // -------------------------------------------------
    // EXPORT
    // -------------------------------------------------
    exportModel(format) {
        switch(format) {
            case 'python':
                this.downloadCode();
                break;
            case 'colab':
                const notebook = codeGenerator.generateColabNotebook();
                const blob = new Blob([notebook], {type: 'application/json'});
                utils.downloadBlob(blob, 'deepforge_colab_notebook.ipynb');
                utils.notify('Colab notebook downloaded!');
                break;
            default:
                utils.notify(`Export to ${format} format coming soon!`);
        }
    },

    downloadCode() {
        const code = codeGenerator.generatePythonScript();
        const blob = new Blob([code], {type: 'text/plain'});
        utils.downloadBlob(blob, 'deepforge_training_script.py');
        utils.notify('Python script downloaded!');
    },

    // -------------------------------------------------
    // SEARCH CONNECTED CODES
    // -------------------------------------------------
    async searchConnectedCodes() {
        console.log('Connected Codes search initiated');
        const currentModel = state.model;
        if (!currentModel) return utils.notify('Please select a model first', 'warning');
        
        const apiKey = localStorage.getItem('gemini_api_key');
        if (!apiKey) return utils.notify('Please connect AI first (Tab 3)', 'error');
        
        window.showLoadingMessage(`🔍 Searching projects for ${currentModel}...`);
        
        try {
            const prompt = `Find 3 GitHub repositories or real-world projects that implement ${currentModel}. Return a JSON list with 'title', 'url', and 'description'.`;
            const response = await fetch(`https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key=${apiKey}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ contents: [{ parts: [{ text: prompt }] }] })
            });
            if (!response.ok) throw new Error('API Error');
            const data = await response.json();
            const text = data.candidates[0].content.parts[0].text;
            window.showConnectedResults(text, currentModel);
            window.hideLoadingMessage();
        } catch (error) {
            console.error(error);
            window.hideLoadingMessage();
            utils.notify('Search failed: ' + error.message, 'error');
        }
    }
};

// =====================================================
// HELPER FUNCTIONS FOR INIT
// =====================================================
const applyTranslations = () => {
    const currentLang = state.language;
    const trans = translations[currentLang];
    if (!trans) return;
    
    document.querySelectorAll('[data-translate]').forEach(element => {
        const key = element.dataset.translate;
        if (trans[key]) element.textContent = trans[key];
    });
    
    if (currentLang === 'ar') document.documentElement.setAttribute('dir', 'rtl');
    else document.documentElement.setAttribute('dir', 'ltr');
};

const renderModels = () => {
    const grid = document.getElementById('modelGrid');
    if (!grid) return;
    
    const fragment = document.createDocumentFragment();
    
    Object.entries(models).forEach(([key, model]) => {
        const card = document.createElement('div');
        card.className = 'model-card';
        card.dataset.model = key;
        card.setAttribute('role', 'listitem');
        card.setAttribute('tabindex', '0');
        
        if (model.type === 'ml') {
            card.innerHTML = `<span class="model-badge">ML</span>`;
        }
        
        card.innerHTML += `
            <h3><i class="${model.icon}" aria-hidden="true"></i> ${model.name}</h3>
            <p>${model.desc}</p>
            <small>${model.type === 'ml' ? 'Classical ML' : `${model.layers || 'N/A'} layers`}</small>
        `;
        fragment.appendChild(card);
    });
    
    grid.innerHTML = '';
    grid.appendChild(fragment);
};

const renderLayerButtons = () => {
    const container = document.getElementById('layerButtons');
    if (!container) return;
    
    const fragment = document.createDocumentFragment();
    
    layerTypes.forEach(layer => {
        const button = document.createElement('button');
        button.className = 'btn btn-secondary';
        button.dataset.layer = layer.type;
        button.innerHTML = `<i class="${layer.icon}"></i> ${layer.type}`;
        fragment.appendChild(button);
    });
    
    const clearButton = document.createElement('button');
    clearButton.className = 'btn btn-danger';
    clearButton.id = 'clearLayersBtn';
    clearButton.innerHTML = '<i class="fas fa-trash"></i> Clear All';
    fragment.appendChild(clearButton);
    
    container.innerHTML = '';
    container.appendChild(fragment);
};

const loadSavedConfiguration = () => {
    const saved = localStorage.getItem('deepforge_config');
    if (!saved) return;
    try {
        const config = JSON.parse(saved);
        handlers.applyConfigObject(config);
    } catch (e) {
        console.error('Failed to load saved config', e);
    }
};

const checkSavedApiKey = () => {
    const savedKey = localStorage.getItem('gemini_api_key');
    if (savedKey) document.getElementById('geminiApiKey').value = savedKey;
};

const setupEventListeners = () => {
    // Global Click Listener (Handles dynamic and static elements)
    document.addEventListener('click', (e) => {

// Model mode modal actions (Scratch vs Pretrained)
    if (e.target.closest('#changeModelModeBtn') || e.target.closest('#openModelModeModalFromCard')) {
        e.preventDefault();
        handlers.showModelModeModal(state.model);
        return;
    }
    if (e.target.closest('#chooseScratchBtn')) {
        e.preventDefault();
        handlers.applyModelMode('scratch');
        return;
    }
    if (e.target.closest('#choosePretrainedBtn')) {
        e.preventDefault();
        handlers.applyModelMode('pretrained');
        return;
    }
    if (e.target.closest('#closeModelModeModalBtn') || e.target.closest('#cancelModelModeBtn')) {
        e.preventDefault();
        handlers.cancelModelModeModal();
        return;
    }

        // Tab switching
        if (e.target.closest('.tab')) {
            handlers.switchTab(e.target.closest('.tab').dataset.tab);
        }
        // Model filters
        if (e.target.closest('[data-filter]')) {
            const btn = e.target.closest('[data-filter]');
            handlers.filterModels(btn.dataset.filter, btn);
        }
        // Model selection
        if (e.target.closest('.model-card')) {
            const card = e.target.closest('.model-card');
            if (card.dataset.model) handlers.selectModel(card.dataset.model);
            else if (card.dataset.export) handlers.exportModel(card.dataset.export);
        }
        // Feature mode (Prebuilt/Custom)
        if (e.target.closest('[data-mode]')) {
            const modeEl = e.target.closest('[data-mode]');
            handlers.switchFeatureMode(modeEl.dataset.mode, modeEl);
        }
        // Open Layer Config
        if (e.target.closest('[data-layer]')) {
            handlers.addLayer(e.target.closest('[data-layer]').dataset.layer);
        }
        // Remove Layer
        if (e.target.closest('[data-remove]')) {
            handlers.removeLayer(parseInt(e.target.closest('[data-remove]').dataset.remove));
        }
        // AI Analysis Buttons
        if (e.target.closest('[data-action]')) {
            handlers.startAnalysis(e.target.closest('[data-action]').dataset.action);
        }
        // AI Analysis Cards
        if (e.target.closest('[data-analysis]')) {
            handlers.startAnalysis(e.target.closest('[data-analysis]').dataset.analysis);
        }

        // --- Dynamic Modal Button Handlers ---
        if (e.target.closest('#cancelLayerConfigBtn')) handlers.closeLayerConfig();
        
        // Handle generic close buttons (The "X" in top right of modals)
        if (e.target.closest('.close-modal')) {
            document.querySelectorAll('.modal').forEach(m => {
                m.classList.remove('active');
                m.setAttribute('aria-hidden', 'true');
            });
        }

        // Apply Layer Configs
        if (e.target.closest('#applyConv2DConfigBtn')) {
            handlers.applyLayerConfig('Conv2D', {
                filters: document.getElementById('conv_filters').value,
                kernel_size: document.getElementById('conv_kernel').value,
                stride: document.getElementById('conv_stride').value,
                padding: document.getElementById('conv_padding').value,
                activation: document.getElementById('conv_activation').value
            });
        }
        if (e.target.closest('#applyDenseConfigBtn')) {
            handlers.applyLayerConfig('Dense', {
                units: document.getElementById('dense_units').value,
                activation: document.getElementById('dense_activation').value,
                use_bias: document.getElementById('dense_bias').value
            });
        }
        if (e.target.closest('#applyDropoutConfigBtn')) {
            handlers.applyLayerConfig('Dropout', { rate: document.getElementById('dropout_rate').value });
        }
        if (e.target.closest('#applyMaxPoolConfigBtn')) {
            handlers.applyLayerConfig('MaxPool', {
                pool_size: document.getElementById('maxpool_size').value,
                stride: document.getElementById('maxpool_stride').value,
                padding: document.getElementById('maxpool_padding').value
            });
        }
        if (e.target.closest('#applyAvgPoolConfigBtn')) {
            handlers.applyLayerConfig('AvgPool', {
                pool_size: document.getElementById('avgpool_size').value,
                stride: document.getElementById('avgpool_stride').value,
                padding: document.getElementById('avgpool_padding').value
            });
        }

        // ============================================
        // ML CONFIGURATION BUTTONS (Dynamic)
        // ============================================
        if (e.target.closest('#applyMLConfigBtn')) {
            e.preventDefault(); // Prevent accidental form submission
            handlers.applyMLConfig();
        }
        if (e.target.closest('#closeMLConfigBtn')) {
            e.preventDefault();
            handlers.closeMLConfig();
        }
        if (e.target.closest('#resetMLConfigBtn')) {
            e.preventDefault();
            // Re-open the modal to reset to default values
            if (state.mlConfig && state.mlConfig.modelType) {
                handlers.showMLConfigModal(state.mlConfig.modelType);
            }
        }

        // Result Tabs
        if (e.target.closest('[data-result-tab]')) {
            const btn = e.target.closest('[data-result-tab]');
            document.querySelectorAll('.result-tab').forEach(t => t.classList.remove('active'));
            btn.classList.add('active');
            document.querySelectorAll('.result-tab-content').forEach(c => c.classList.remove('active'));
            document.getElementById(`${btn.dataset.resultTab}-tab`).classList.add('active');
        }
        
        // AI Suggestions
        if (e.target.closest('[data-question]')) {
            document.getElementById('ai-question').value = e.target.closest('[data-question]').dataset.question;
            handlers.askAI();
        }

        // Copy Code
        if (e.target.closest('[data-copy-index]')) {
            const btn = e.target.closest('[data-copy-index]');
            const codeContainer = btn.parentElement.nextElementSibling.querySelector('code');
            utils.copyToClipboard(codeContainer.textContent);
        }

        // Connected Codes
        if (e.target.id === 'searchConnectedBtn' || e.target.closest('#searchConnectedBtn')) {
            handlers.searchConnectedCodes();
        }
        
        // Recommendation Buttons
        if (e.target.id === 'generateRecommendationBtn' || e.target.closest('#generateRecommendationBtn')) {
            e.preventDefault();
            handlers.generateAIRecommendation();
        }
        if (e.target.id === 'resetFormBtn' || e.target.closest('#resetFormBtn')) {
            e.preventDefault();
            handlers.resetRecommendationForm();
        }
    });

    // Static Button Listeners
    document.getElementById('clearLayersBtn')?.addEventListener('click', () => handlers.clearLayers());
    document.getElementById('activate-btn')?.addEventListener('click', () => handlers.activateAI());
    document.getElementById('askAIBtn')?.addEventListener('click', () => handlers.askAI());
    document.getElementById('applyRecommendationsBtn')?.addEventListener('click', () => handlers.applyRecommendations());
    document.getElementById('newAnalysisBtn')?.addEventListener('click', () => handlers.updateWizardStep(2));
    document.getElementById('downloadResultsBtn')?.addEventListener('click', () => {
        const results = state.aiOptimizerState.analysisHistory[state.aiOptimizerState.analysisHistory.length - 1];
        if (results) {
            const blob = new Blob([results.results], {type: 'text/plain'});
            utils.downloadBlob(blob, `analysis_results_${Date.now()}.txt`);
            utils.notify('Results downloaded');
        }
    });

    // API Key Toggle
    document.getElementById('toggleApiKeyBtn')?.addEventListener('click', () => {
        const input = document.getElementById('geminiApiKey');
        const icon = document.getElementById('api-visibility-icon');
        if (input.type === 'password') {
            input.type = 'text';
            icon.className = 'fas fa-eye-slash';
        } else {
            input.type = 'password';
            icon.className = 'fas fa-eye';
        }
    });

    
    // Clear saved API key
    document.getElementById('clearApiKeyBtn')?.addEventListener('click', () => {
        handlers.clearApiKey();
    });

    // Code preview / export toolbar
    document.getElementById('copyCodeBtn')?.addEventListener('click', () => handlers.copyGeneratedCode());
    document.getElementById('downloadPyBtn')?.addEventListener('click', () => handlers.downloadCode());
    document.getElementById('downloadIpynbBtn')?.addEventListener('click', () => {
        const notebook = codeGenerator.generateColabNotebook();
        const blob = new Blob([notebook], {type: 'application/json'});
        utils.downloadBlob(blob, 'deepforge_colab_notebook.ipynb');
        utils.notify('Colab notebook downloaded!', 'success');
    });
    document.getElementById('exportConfigBtn')?.addEventListener('click', () => handlers.exportConfig());
    document.getElementById('importConfigBtn')?.addEventListener('click', () => {
        document.getElementById('importConfigInput')?.click();
    });
    document.getElementById('importConfigInput')?.addEventListener('change', (e) => {
        const file = e.target.files?.[0];
        handlers.importConfigFromFile(file);
        e.target.value = '';
    });

    // Language selector
    document.getElementById('languageSelect')?.addEventListener('change', (e) => {
        state.language = e.target.value;
        localStorage.setItem('deepforge_language', state.language);
        applyTranslations();
    });

    // Real-time code updates
    ['numClasses', 'optimizer', 'lr', 'batchSize', 'epochs', 'freezeLayers', 'inputSize', 'customTop'].forEach(id => {
        document.getElementById(id)?.addEventListener('change', () => codeGenerator.generateCode());
        document.getElementById(id)?.addEventListener('input', () => codeGenerator.generateCode());
    });

    // Range inputs display update
    document.querySelectorAll('input[type="range"]').forEach(range => {
        range.addEventListener('input', (e) => {
            const display = document.getElementById(`${e.target.id}_value`);
            if (display) display.textContent = e.target.value;
        });
    });
};

// =====================================================
// GLOBAL HELPERS EXPOSED TO WINDOW (For legacy support)
// =====================================================
window.showConnectedResults = function(response, modelName) {
    const existing = document.getElementById('connectedCodesModal');
    if (existing) existing.remove();
    const formatted = response.replace(/```json/g, '').replace(/```/g, '');
    const modalHTML = `
    <div id="connectedCodesModal" class="modal active" style="display:flex">
        <div class="modal-content">
            <div class="modal-header">
                <h2>Similar Projects: ${modelName}</h2>
                <button class="close-modal" onclick="document.getElementById('connectedCodesModal').remove()">&times;</button>
            </div>
            <div class="modal-body" style="white-space: pre-wrap; font-family: monospace;">${formatted}</div>
        </div>
    </div>`;
    document.body.insertAdjacentHTML('beforeend', modalHTML);
};

window.showLoadingMessage = function(msg) {
    window.hideLoadingMessage();
    const loader = document.createElement('div');
    loader.id = 'globalLoader';
    loader.style.cssText = 'position:fixed;top:0;left:0;width:100%;height:100%;background:rgba(0,0,0,0.85);z-index:9999;display:flex;flex-direction:column;align-items:center;justify-content:center;color:white;';
    loader.innerHTML = `<i class="fas fa-spinner fa-spin fa-3x"></i><p style="margin-top:15px">${msg}</p>`;
    document.body.appendChild(loader);
};

window.hideLoadingMessage = function() {
    const loader = document.getElementById('globalLoader');
    if (loader) loader.remove();
};

// =====================================================
// INIT EXPORT
// =====================================================
export const init = () => {
    applyTranslations();
    renderModels();
    renderLayerButtons();
    setupEventListeners();
    codeGenerator.generateCode();
    loadSavedConfiguration();
    checkSavedApiKey();
};