export const models = {
    'alexnet': { name: 'AlexNet', desc: 'Classic CNN for Image Classification', layers: 11, icon: 'fas fa-chess-rook', type: 'dl', fromScratch: true },
    'vgg16': { name: 'VGG16', desc: 'Deep Convolutional Network', layers: 16, icon: 'fas fa-layer-group', type: 'dl', fromScratch: true , supportsPretrained: true },
    'lstm': { name: 'LSTM', desc: 'Long Short-Term Memory for Sequences', layers: 4, icon: 'fas fa-stream', type: 'dl', fromScratch: true },
    'gru': { name: 'GRU', desc: 'Gated Recurrent Unit for Sequences', layers: 3, icon: 'fas fa-water', type: 'dl', fromScratch: true },
    'simple_cnn': { name: 'Simple CNN', desc: 'Basic Convolutional Network', layers: 7, icon: 'fas fa-layer-group', type: 'dl', fromScratch: true },
    'resnet50': { name: 'ResNet-50', desc: 'Residual Network with Skip Connections', layers: 50, icon: 'fas fa-link', type: 'dl', fromScratch: false , supportsPretrained: true },
    'mobilenet': { name: 'MobileNetV2', desc: 'Lightweight Mobile-Optimized Network', layers: 28, icon: 'fas fa-mobile-alt', type: 'dl', fromScratch: false , supportsPretrained: true },
    'efficientnet': { name: 'EfficientNetB0', desc: 'Efficient Scaling Architecture', layers: 18, icon: 'fas fa-bolt', type: 'dl', fromScratch: false , supportsPretrained: true },
    'inceptionv3': { name: 'InceptionV3', desc: 'Multi-scale Feature Extraction', layers: 48, icon: 'fas fa-sitemap', type: 'dl', fromScratch: false , supportsPretrained: true },
    'densenet': { name: 'DenseNet121', desc: 'Densely Connected Network', layers: 121, icon: 'fas fa-network-wired', type: 'dl', fromScratch: false , supportsPretrained: true },
    'transformer': { name: 'Transformer', desc: 'Attention-based Architecture', layers: 12, icon: 'fas fa-exchange-alt', type: 'dl', fromScratch: true, pretrainedAvailable: false },
    'unet': { name: 'U-Net', desc: 'Segmentation Architecture (Simplified)', layers: 23, icon: 'fas fa-project-diagram', type: 'dl', fromScratch: true, pretrainedAvailable: false },
    'autoencoder': { name: 'Autoencoder', desc: 'Encoder-Decoder Architecture', layers: 8, icon: 'fas fa-compress-alt', type: 'dl', fromScratch: true, pretrainedAvailable: false },
    
    // ML Models
    'svm': { name: 'SVM', desc: 'Support Vector Machine for Classification', type: 'ml', icon: 'fas fa-vector-square' },
    'knn': { name: 'KNN', desc: 'K-Nearest Neighbors Classifier', type: 'ml', icon: 'fas fa-project-diagram' },
    'randomforest': { name: 'Random Forest', desc: 'Ensemble of Decision Trees', type: 'ml', icon: 'fas fa-tree' },
    'xgboost': { name: 'XGBoost', desc: 'Gradient Boosting Algorithm', type: 'ml', icon: 'fas fa-rocket' },
    'decisiontree': { name: 'Decision Tree', desc: 'Tree-based Classification', type: 'ml', icon: 'fas fa-sitemap' },
    'naivebayes': { name: 'Naive Bayes', desc: 'Probabilistic Classifier', type: 'ml', icon: 'fas fa-chart-pie' },
    'logisticregression': { name: 'Logistic Regression', desc: 'Linear Classification Model', type: 'ml', icon: 'fas fa-chart-line' },
    'kmeans': { name: 'K-Means', desc: 'Clustering Algorithm', type: 'ml', icon: 'fas fa-circle-nodes' },
    'pca': { name: 'PCA', desc: 'Principal Component Analysis', type: 'ml', icon: 'fas fa-compress-arrows-alt' },
    'linearregression': { name: 'Linear Regression', desc: 'Linear Prediction Model', type: 'ml', icon: 'fas fa-chart-line' }
};