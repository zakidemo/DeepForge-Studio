export const state = {
    model: null,
    customLayers: [],
    customLayerConfigs: [],
    currentMode: 'prebuilt',
    pendingLayerType: null,
    modelMode: 'pretrained',
    freezeLayers: 'base',
    customTop: 'default',
    mlConfig: null,
    language: localStorage.getItem('deepforge_language') || 'en',
    aiOptimizerState: {
        isConnected: false,
        currentStep: 1,
        currentAnalysis: null,
        analysisHistory: [],
        lastRecommendations: null
    }
};