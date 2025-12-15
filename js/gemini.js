import { state } from './state.js';
import { models } from './config/models.js';
import { utils } from './utils.js';

export class GeminiOptimizer {
    constructor() {
        this.apiKey = '';
        this.baseURL = 'https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent';
        this.lastRecommendations = null;
        this.responseLanguage = 'English';
    }

    setApiKey(key) {
        this.apiKey = key;
        localStorage.setItem('gemini_api_key', key);
    }

    getApiKey() {
        if (!this.apiKey) {
            this.apiKey = localStorage.getItem('gemini_api_key') || '';
        }
        return this.apiKey;
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
                const error = await response.json();
                throw new Error(error.error?.message || `API request failed: ${response.status}`);
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