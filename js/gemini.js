import { state } from './state.js';
import { models } from './config/models.js';
import { utils } from './utils.js';

export class GeminiOptimizer {
    constructor() {
        this.apiKey = '';
        this.apiBase = 'https://generativelanguage.googleapis.com/v1beta';
        // The provider retires models on its own schedule: the previously
        // hardcoded gemini-2.0-flash was shut down on 1 June 2026, which broke
        // the feature with no change on our side. Rather than hardcode another
        // identifier that will expire in turn, the model is configurable and
        // the list of usable models is discovered from the provider at connect
        // time. PREFERRED_MODELS is only an ordering hint for choosing a
        // sensible default from whatever the account actually has access to.
        this.PREFERRED_MODELS = [
            'gemini-3.5-flash',
            'gemini-3.6-flash',
            'gemini-3.5-flash-lite',
            'gemini-2.5-flash'
        ];
        this.model = '';
        this.availableModels = [];
        this.lastRecommendations = null;
        this.responseLanguage = 'English';
    }

    get baseURL() {
        const model = this.model || this.PREFERRED_MODELS[0];
        return `${this.apiBase}/models/${model}:generateContent`;
    }

    setModel(model) {
        this.model = model;
        try { sessionStorage.setItem('gemini_model', model); } catch (_) {}
    }

    getModel() {
        if (!this.model) {
            try { this.model = sessionStorage.getItem('gemini_model') || ''; } catch (_) {}
        }
        return this.model || this.PREFERRED_MODELS[0];
    }

    // Ask the provider which models this key can actually use, so a retirement
    // on their side degrades to a different choice rather than a dead feature.
    async listModels() {
        const apiKey = this.getApiKey();
        if (!apiKey) throw new Error('Please enter your Google Gemini API key first');

        const response = await fetch(`${this.apiBase}/models?key=${apiKey}`);
        if (!response.ok) {
            const err = await response.json().catch(() => ({}));
            throw new Error(err.error?.message || `Could not list models: ${response.status}`);
        }
        const data = await response.json();
        this.availableModels = (data.models || [])
            .filter(m => (m.supportedGenerationMethods || []).includes('generateContent'))
            .map(m => m.name.replace(/^models\//, ''))
            // Exclude specialised endpoints that cannot answer a text prompt.
            .filter(n => !/embedding|aqa|imagen|veo|tts|image|音/i.test(n));

        if (!this.availableModels.length) {
            throw new Error('This API key has no models available for text generation.');
        }

        // Keep the current choice if it is still offered; otherwise take the
        // highest-ranked preference that is, and fall back to whatever exists.
        const current = this.getModel();
        if (!this.availableModels.includes(current)) {
            const pick = this.PREFERRED_MODELS.find(m => this.availableModels.includes(m))
                || this.availableModels.find(n => /flash/i.test(n))
                || this.availableModels[0];
            this.setModel(pick);
        }
        return this.availableModels;
    }

    // Reviewer 3, comment 3.4b: the key was written to localStorage, where it
    // persisted indefinitely and was readable by any script running in this
    // origin. It is now held in memory for the session only, mirrored to
    // sessionStorage so a page reload does not lose it, and cleared when the
    // tab closes. This narrows the window; it does not eliminate the risk, and
    // the documentation says so.
    setApiKey(key) {
        this.apiKey = key;
        try { sessionStorage.setItem('gemini_api_key', key); } catch (_) {}
        // Remove any key persisted by an earlier version of the application.
        try { localStorage.removeItem('gemini_api_key'); } catch (_) {}
    }

    getApiKey() {
        if (!this.apiKey) {
            try { this.apiKey = sessionStorage.getItem('gemini_api_key') || ''; } catch (_) { this.apiKey = ''; }
        }
        return this.apiKey;
    }

    clearApiKey() {
        this.apiKey = '';
        try { sessionStorage.removeItem('gemini_api_key'); } catch (_) {}
        try { localStorage.removeItem('gemini_api_key'); } catch (_) {}
    }

    async makeRequest(prompt, maxTokens = 2000) {
        const languageInstruction = this.responseLanguage === 'Arabic' 
            ? 'Please respond in Arabic language (العربية). Use Arabic technical terms when available.'
            : 'Please respond in English language.';
        
        const enhancedPrompt = `${languageInstruction}\n\n${prompt}`;
        
        const apiKey = this.getApiKey();
        if (!apiKey) {
            throw new Error('Please enter your Google Gemini API key first');
        }

        try {
            const response = await fetch(`${this.baseURL}?key=${apiKey}`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    contents: [{
                        parts: [{ text: enhancedPrompt }]
                    }],
                    generationConfig: {
                        maxOutputTokens: maxTokens,
                        temperature: 0.7,
                        topP: 0.8,
                        topK: 40
                    }
                })
            });

            if (!response.ok) {
                const error = await response.json().catch(() => ({}));
                const message = error.error?.message || `API request failed: ${response.status}`;
                // A retired model is the most likely cause of a sudden failure
                // on a key that previously worked; say so and say what to do.
                if (/no longer available|not found|deprecated|shut down/i.test(message)) {
                    throw new Error(`The model \"${this.getModel()}\" is no longer available from the provider. Reconnect to refresh the model list and choose a current model. (Provider message: ${message})`);
                }
                throw new Error(message);
            }

            const data = await response.json();
            
            if (!data.candidates || !data.candidates[0] || !data.candidates[0].content) {
                throw new Error('No response generated');
            }
            
            return data.candidates[0].content.parts[0].text;
        } catch (error) {
            console.error('Gemini API Error:', error);
            throw error;
        }
    }

    getCurrentModelConfig() {
        return {
            architecture: state.model || 'custom',
            modelMode: state.modelMode || 'scratch',
            customLayers: state.customLayers || [],
            hyperparameters: {
                numClasses: document.getElementById('numClasses')?.value || 10,
                optimizer: document.getElementById('optimizer')?.value || 'adam',
                learningRate: document.getElementById('lr')?.value || 0.001,
                batchSize: document.getElementById('batchSize')?.value || 32,
                epochs: document.getElementById('epochs')?.value || 50,
            },
            enabledFeatures: utils.getEnabledFeatures()
        };
    }

    async getRecommendations() {
        const config = this.getCurrentModelConfig();
        const modelCode = document.getElementById('modelCode')?.textContent || '';
        
        const prompt = `As an expert ML engineer, analyze this model configuration and provide specific recommendations:

**Current Configuration:**
- Architecture: ${config.architecture}
- Mode: ${config.modelMode}
- Classes: ${config.hyperparameters.numClasses}
- Optimizer: ${config.hyperparameters.optimizer}
- Learning Rate: ${config.hyperparameters.learningRate}
- Batch Size: ${config.hyperparameters.batchSize}
- Epochs: ${config.hyperparameters.epochs}
- Custom Layers: ${config.customLayers.join(', ') || 'None'}

**Model Code:**
\`\`\`python
${modelCode.slice(0, 1500)}
\`\`\`

Please provide:
1. **Architecture Recommendations**: Suggest improvements to the model structure
2. **Hyperparameter Optimization**: Recommend better values with reasoning
3. **Performance Tips**: Specific techniques to improve accuracy/speed
4. **Potential Issues**: Identify problems and solutions
5. **Best Practices**: Industry standards for this type of model

Format your response with clear headings and actionable suggestions. Include specific parameter values where applicable.`;

        const response = await this.makeRequest(prompt, 3000);
        this.lastRecommendations = response;
        return response;
    }

    async optimizeHyperparameters() {
        const config = this.getCurrentModelConfig();
        
        const prompt = `As an ML optimization expert, provide hyperparameter tuning recommendations for this configuration:

**Current Setup:**
- Model: ${config.architecture}
- Classes: ${config.hyperparameters.numClasses}
- Current LR: ${config.hyperparameters.learningRate}
- Current Batch Size: ${config.hyperparameters.batchSize}
- Current Epochs: ${config.hyperparameters.epochs}
- Optimizer: ${config.hyperparameters.optimizer}

**Task**: Suggest optimal hyperparameters with the following format:

## 🎯 Recommended Hyperparameters
- **Learning Rate**: [value] - [reasoning]
- **Batch Size**: [value] - [reasoning]
- **Epochs**: [value] - [reasoning]
- **Optimizer**: [choice] - [reasoning]
- **Dropout**: [value] - [reasoning]
- **Regularization**: [recommendations]

## 📊 Learning Rate Schedule
[Specific schedule recommendations]

## 🔧 Advanced Optimizations
[Additional tuning suggestions]

Provide specific values I can apply directly to the interface.`;

        return await this.makeRequest(prompt, 2500);
    }

    async analyzeArchitecture() {
        const config = this.getCurrentModelConfig();
        const modelCode = document.getElementById('modelCode')?.textContent || '';
        
        const prompt = `Perform a deep architecture analysis of this model:

**Model Details:**
- Type: ${config.architecture}
- Custom Layers: ${JSON.stringify(config.customLayers)}
- Classes: ${config.hyperparameters.numClasses}

**Code:**
\`\`\`python
${modelCode.slice(0, 2000)}
\`\`\`

Analyze:

## 🔍 Architecture Strengths
[What's working well]

## ⚠️ Potential Weaknesses
[Issues and bottlenecks]

## 🚀 Optimization Opportunities
[Specific improvements]

## 📈 Performance Predictions
[Expected accuracy, training time, etc.]

## 🛠️ Alternative Architectures
[Better options for this use case]

Be specific and technical. Include parameter counts, computational complexity, and memory usage considerations.`;

        return await this.makeRequest(prompt, 3000);
    }

    async getBestPractices() {
        const config = this.getCurrentModelConfig();
        
        const prompt = `Provide comprehensive best practices for this ML setup:

**Configuration:**
- Model: ${config.architecture}
- Type: ${models[config.architecture]?.type || 'unknown'}
- Classes: ${config.hyperparameters.numClasses}

Share best practices for:

## 📚 Data Preparation
- Preprocessing steps
- Augmentation strategies
- Train/validation splits

## 🏗️ Model Architecture
- Layer design principles
- Activation functions
- Regularization techniques

## ⚙️ Training Strategy
- Learning rate scheduling
- Early stopping criteria
- Checkpoint management

## 📊 Evaluation & Monitoring
- Metrics to track
- Visualization techniques
- Overfitting detection

## 🚀 Production Considerations
- Model deployment
- Performance optimization
- Monitoring in production

Include specific code snippets and parameter recommendations where relevant.`;

        return await this.makeRequest(prompt, 3500);
    }

    async askFollowUp(question) {
        const config = this.getCurrentModelConfig();
        const context = this.lastRecommendations ? `\n\nPrevious Analysis:\n${this.lastRecommendations.slice(0, 500)}...` : '';
        
        const prompt = `Context: I'm working on a ${config.architecture} model with ${config.hyperparameters.numClasses} classes.${context}

Question: ${question}

Please provide a detailed, actionable answer specific to my model configuration.`;

        return await this.makeRequest(prompt, 2000);
    }
}

// THIS IS THE LINE YOU WERE LIKELY MISSING:
export const geminiOptimizer = new GeminiOptimizer();