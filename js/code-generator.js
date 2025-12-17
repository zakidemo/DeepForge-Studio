import { state } from './state.js';
import { models } from './config/models.js';

export const codeGenerator = {
    generateCode() {
        const code = this.generateModelCode();
        const el = document.getElementById('modelCode');
        if (el) {
            el.textContent = code;
            if (window.Prism && typeof window.Prism.highlightElement === 'function') {
                try { window.Prism.highlightElement(el); } catch (_) {}
            }
        }
        return code;
    },

    generateModelCode() {
        const numClasses = document.getElementById('numClasses')?.value || 10;

        // If the user is in Custom Builder mode, always generate from the custom layer stack.
        // This prevents previously selected ML/prebuilt models (e.g., KNN) from "sticking" into exports.
        if (state.currentMode === 'custom') {
            return this.generateDLCode(numClasses);
        }

        if (state.model && models[state.model]?.type === 'ml') {
            return this.generateMLCode(state.model, numClasses);
        }


        // Keras Applications architectures (from scratch)
        if (
            state.modelMode === 'scratch' &&
            state.model &&
            ['vgg16', 'resnet50', 'mobilenet', 'efficientnet', 'inceptionv3', 'densenet'].includes(state.model) &&
            state.currentMode !== 'custom'
        ) {
            return this.generateApplicationScratchCode(state.model, numClasses);
        }

        if (
            state.modelMode === 'pretrained' &&
            state.model &&
            ['vgg16', 'resnet50', 'mobilenet', 'efficientnet', 'inceptionv3', 'densenet'].includes(state.model)
        ) {
            return this.generatePretrainedCode(state.model, numClasses);
        }

        return this.generateDLCode(numClasses);
    },

    generateDLCode(numClasses) {
        let code = `import tensorflow as tf\nfrom tensorflow.keras import layers, models\n\n# Model architecture\nmodel = models.Sequential([\n`;

        if (state.currentMode === 'custom' && state.customLayers.length > 0) {
            let hasInput = false;
            state.customLayers.forEach((layer, index) => {
                const config = state.customLayerConfigs[index] || {};
                
                switch(layer) {
                    case 'Conv2D':
                        if (!hasInput) {
                            code += `    layers.Conv2D(${config.filters || 32}, (${config.kernel_size || 3}, ${config.kernel_size || 3}), strides=${config.stride || 1}, padding='${config.padding || 'valid'}', activation='${config.activation || 'relu'}', input_shape=(224, 224, 3)),\n`;
                            hasInput = true;
                        } else {
                            code += `    layers.Conv2D(${config.filters || 32}, (${config.kernel_size || 3}, ${config.kernel_size || 3}), strides=${config.stride || 1}, padding='${config.padding || 'valid'}', activation='${config.activation || 'relu'}'),\n`;
                        }
                        break;
                    case 'Dense':
                        code += `    layers.Dense(${config.units || 128}, activation='${config.activation || 'relu'}'),\n`;
                        break;
                    case 'MaxPool':
                        code += `    layers.MaxPooling2D((${config.pool_size || 2}, ${config.pool_size || 2}), strides=${config.stride || 2}, padding='${config.padding || 'valid'}'),\n`;
                        break;
                    case 'AvgPool':
                        code += `    layers.AveragePooling2D((${config.pool_size || 2}, ${config.pool_size || 2}), strides=${config.stride || 2}, padding='${config.padding || 'valid'}'),\n`;
                        break;
                    case 'Dropout':
                        code += `    layers.Dropout(${config.rate || 0.5}),\n`;
                        break;
                    case 'Flatten':
                        code += `    layers.Flatten(),\n`;
                        break;
                    case 'LSTM':
                        if(!hasInput) { code += `    layers.LSTM(128, input_shape=(100, 1)),\n`; hasInput=true; } else { code += `    layers.LSTM(64),\n`; }
                        break;
                    case 'GRU':
                         if(!hasInput) { code += `    layers.GRU(64, input_shape=(100, 1)),\n`; hasInput=true; } else { code += `    layers.GRU(32),\n`; }
                        break;
                }
            });
        } else if (state.model) {
            code += this.getModelArchitecture(state.model);
        }
        
        code += `    layers.Dense(${numClasses}, activation='softmax')\n])\n\n# Summary\nmodel.summary()\n`;
        code += `\n# NOTE: Compilation and training steps are included in the exported training pipeline.\n`;
        return code;
    },

    getModelArchitecture(modelName) {
        // ... (Paste the architectures object from original script here) ...
        const architectures = {
             'vgg16': `    # VGG16 Architecture
    layers.Conv2D(64, (3,3), activation='relu', padding='same', input_shape=(224,224,3)),
    layers.Conv2D(64, (3,3), activation='relu', padding='same'),
    layers.MaxPooling2D((2,2), strides=2),
    
    layers.Conv2D(128, (3,3), activation='relu', padding='same'),
    layers.Conv2D(128, (3,3), activation='relu', padding='same'),
    layers.MaxPooling2D((2,2), strides=2),
    
    layers.Conv2D(256, (3,3), activation='relu', padding='same'),
    layers.Conv2D(256, (3,3), activation='relu', padding='same'),
    layers.Conv2D(256, (3,3), activation='relu', padding='same'),
    layers.MaxPooling2D((2,2), strides=2),
    
    layers.Conv2D(512, (3,3), activation='relu', padding='same'),
    layers.Conv2D(512, (3,3), activation='relu', padding='same'),
    layers.Conv2D(512, (3,3), activation='relu', padding='same'),
    layers.MaxPooling2D((2,2), strides=2),
    
    layers.Conv2D(512, (3,3), activation='relu', padding='same'),
    layers.Conv2D(512, (3,3), activation='relu', padding='same'),
    layers.Conv2D(512, (3,3), activation='relu', padding='same'),
    layers.MaxPooling2D((2,2), strides=2),
    
    layers.Flatten(),
    layers.Dense(4096, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(4096, activation='relu'),
    layers.Dropout(0.5),\n`,
                        'alexnet': `    # AlexNet Architecture
    layers.Conv2D(96, (11,11), strides=4, activation='relu', input_shape=(224,224,3)),
    layers.MaxPooling2D((3,3), strides=2),
    layers.Conv2D(256, (5,5), activation='relu', padding='same'),
    layers.MaxPooling2D((3,3), strides=2),
    layers.Conv2D(384, (3,3), activation='relu', padding='same'),
    layers.Conv2D(384, (3,3), activation='relu', padding='same'),
    layers.Conv2D(256, (3,3), activation='relu', padding='same'),
    layers.MaxPooling2D((3,3), strides=2),
    layers.Flatten(),
    layers.Dense(4096, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(4096, activation='relu'),
    layers.Dropout(0.5),\n`,
                        'lstm': `    # LSTM Architecture for Sequences
    layers.LSTM(128, return_sequences=True, input_shape=(100, 1)),
    layers.Dropout(0.2),
    layers.LSTM(128, return_sequences=True),
    layers.Dropout(0.2),
    layers.LSTM(64),
    layers.Dropout(0.2),
    layers.Dense(32, activation='relu'),\n`,
                        'gru': `    # GRU Architecture for Sequences
    layers.GRU(128, return_sequences=True, input_shape=(100, 1)),
    layers.Dropout(0.2),
    layers.GRU(64, return_sequences=True),
    layers.Dropout(0.2),
    layers.GRU(32),
    layers.Dropout(0.2),
    layers.Dense(16, activation='relu'),\n`,
                        'simple_cnn': `    # Simple CNN Architecture
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(224,224,3)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),\n`,

    'unet': `    # U-Net Architecture for Segmentation
    # Encoder
    layers.Conv2D(64, (3,3), activation='relu', padding='same', input_shape=(256,256,3)),
    layers.Conv2D(64, (3,3), activation='relu', padding='same'),
    layers.MaxPooling2D((2,2)),
    
    layers.Conv2D(128, (3,3), activation='relu', padding='same'),
    layers.Conv2D(128, (3,3), activation='relu', padding='same'),
    layers.MaxPooling2D((2,2)),
    
    layers.Conv2D(256, (3,3), activation='relu', padding='same'),
    layers.Conv2D(256, (3,3), activation='relu', padding='same'),
    layers.MaxPooling2D((2,2)),
    
    # Bridge
    layers.Conv2D(512, (3,3), activation='relu', padding='same'),
    layers.Conv2D(512, (3,3), activation='relu', padding='same'),
    
    # Decoder
    layers.Conv2DTranspose(256, (2,2), strides=2, padding='same'),
    layers.Conv2D(256, (3,3), activation='relu', padding='same'),
    layers.Conv2D(256, (3,3), activation='relu', padding='same'),
    
    layers.Conv2DTranspose(128, (2,2), strides=2, padding='same'),
    layers.Conv2D(128, (3,3), activation='relu', padding='same'),
    layers.Conv2D(128, (3,3), activation='relu', padding='same'),
    
    layers.Conv2DTranspose(64, (2,2), strides=2, padding='same'),
    layers.Conv2D(64, (3,3), activation='relu', padding='same'),
    layers.Conv2D(64, (3,3), activation='relu', padding='same'),\n`,

    'autoencoder': `    # Autoencoder Architecture
    # Encoder
    layers.Dense(512, activation='relu', input_shape=(784,)),
    layers.Dense(256, activation='relu'),
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    
    # Latent space
    layers.Dense(32, activation='relu', name='latent_space'),
    
    # Decoder
    layers.Dense(64, activation='relu'),
    layers.Dense(128, activation='relu'),
    layers.Dense(256, activation='relu'),
    layers.Dense(512, activation='relu'),
    layers.Dense(784, activation='sigmoid'),\n`,

    'transformer': `    # Simplified Transformer Architecture
    # Input embedding
    layers.Input(shape=(100,)),
    layers.Embedding(input_dim=10000, output_dim=512),
    
    # Positional encoding would go here (custom implementation needed)
    
    # Multi-head attention blocks (simplified)
    layers.MultiHeadAttention(num_heads=8, key_dim=64),
    layers.Dropout(0.1),
    layers.LayerNormalization(epsilon=1e-6),
    
    # Feed forward network
    layers.Dense(2048, activation='relu'),
    layers.Dense(512),
    layers.Dropout(0.1),
    layers.LayerNormalization(epsilon=1e-6),
    
    # Output layers
    layers.GlobalAveragePooling1D(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),\n`
        };
        return architectures[modelName] || '';
    },

    
    generateApplicationScratchCode(modelType, numClasses) {
        const appMap = {
            'vgg16': 'VGG16',
            'resnet50': 'ResNet50',
            'mobilenet': 'MobileNetV2',
            'efficientnet': 'EfficientNetB0',
            'inceptionv3': 'InceptionV3',
            'densenet': 'DenseNet121'
        };

        const inputSizeMap = {
            'inceptionv3': 299
        };

        const appClass = appMap[modelType];
        const inputSize = inputSizeMap[modelType] || 224;

        // Special case: export fully expanded VGG16 blocks (layer-by-layer) for educational transparency
        if (modelType === 'vgg16') {
            return this.generateVGG16ScratchExpanded(numClasses, inputSize);
        }

        let code = `import tensorflow as tf\nfrom tensorflow.keras import layers, models\n\n`;
        code += `# ${appClass} (from scratch: weights=None)\n`;
        code += `base_model = tf.keras.applications.${appClass}(\n`;
        code += `    include_top=False,\n`;
        code += `    weights=None,\n`;
        code += `    input_shape=(${inputSize}, ${inputSize}, 3)\n`;
        code += `)\n\n`;

        code += `inputs = layers.Input(shape=(${inputSize}, ${inputSize}, 3))\n`;
        code += `x = base_model(inputs)\n`;
        code += `x = layers.GlobalAveragePooling2D()(x)\n`;
        code += `x = layers.Dropout(0.2)(x)\n`;
        code += `outputs = layers.Dense(${numClasses}, activation='softmax')(x)\n`;
        code += `model = models.Model(inputs, outputs)\n\n`;

        code += `# Summary\nmodel.summary()\n`;
        code += `\n# NOTE: Compilation and training steps are included in the exported training pipeline.\n`;

        return code;
    },

    generateVGG16ScratchExpanded(numClasses, inputSize = 224) {
        // Fully expanded VGG16 backbone (conv/pool blocks) for transparent exports.
        // Head is kept lightweight by default (GAP + Dropout + Dense) to suit many tasks.
        let code = `import tensorflow as tf\nfrom tensorflow.keras import layers, models\n\n`;
        code += `# VGG16 (from scratch, expanded layer-by-layer)\n`;
        code += `inputs = layers.Input(shape=(${inputSize}, ${inputSize}, 3))\n`;
        code += `x = inputs\n\n`;

        // Block 1
        code += `# Block 1\n`;
        code += `x = layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(x)\n`;
        code += `x = layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)\n`;
        code += `x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)\n\n`;

        // Block 2
        code += `# Block 2\n`;
        code += `x = layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)\n`;
        code += `x = layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)\n`;
        code += `x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)\n\n`;

        // Block 3
        code += `# Block 3\n`;
        code += `x = layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)\n`;
        code += `x = layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)\n`;
        code += `x = layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)\n`;
        code += `x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)\n\n`;

        // Block 4
        code += `# Block 4\n`;
        code += `x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)\n`;
        code += `x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)\n`;
        code += `x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)\n`;
        code += `x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)\n\n`;

        // Block 5
        code += `# Block 5\n`;
        code += `x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)\n`;
        code += `x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)\n`;
        code += `x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)\n`;
        code += `x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)\n\n`;

        // Head
        code += `# Classification head\n`;
        code += `x = layers.GlobalAveragePooling2D()(x)\n`;
        code += `x = layers.Dropout(0.2)(x)\n`;
        code += `outputs = layers.Dense(${numClasses}, activation='softmax', name='predictions')(x)\n`;
        code += `model = models.Model(inputs, outputs, name='vgg16_from_scratch')\n\n`;

        code += `# Summary\nmodel.summary()\n`;
        code += `\n# NOTE: Compilation and training steps are included in the exported training pipeline.\n`;

        return code;
    },

generatePretrainedCode(modelType, numClasses) {
                        const freezeLayers = document.getElementById('freezeLayers')?.value || 'base';
                    const inputSize = document.getElementById('inputSize')?.value || '224';
                    const customTop = document.getElementById('customTop')?.value || 'default';
                    
                    let code = `import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import `;
                    
                    const appMap = {
                        'vgg16': 'VGG16',
                        'resnet50': 'ResNet50',
                        'mobilenet': 'MobileNetV2',
                        'efficientnet': 'EfficientNetB0',
                        'inceptionv3': 'InceptionV3',
                        'densenet': 'DenseNet121'
                    };

                    const preprocessImportMap = {
                        vgg16: "from tensorflow.keras.applications.vgg16 import preprocess_input",
                        resnet50: "from tensorflow.keras.applications.resnet50 import preprocess_input",
                        mobilenet: "from tensorflow.keras.applications.mobilenet_v2 import preprocess_input",
                        efficientnet: "from tensorflow.keras.applications.efficientnet import preprocess_input",
                        inceptionv3: "from tensorflow.keras.applications.inception_v3 import preprocess_input",
                        densenet: "from tensorflow.keras.applications.densenet import preprocess_input"
                    };

                    
                    code += `${appMap[modelType]}\n\n`;
                    code += `${preprocessImportMap[modelType]}\n\n`;
                    code += `# Load pretrained ${appMap[modelType]} model
base_model = ${appMap[modelType]}(
    input_shape=(${inputSize}, ${inputSize}, 3),
    include_top=False,
    weights='imagenet'
)\n\n`;
                    
                    // Handle freezing
                    if (freezeLayers === 'base') {
                        code += `# Freeze the base model layers
base_model.trainable = False\n\n`;
                    } else if (freezeLayers === 'partial') {
                        code += `# Freeze first 50% of layers
for layer in base_model.layers[:len(base_model.layers)//2]:
    layer.trainable = False\n\n`;
                    }
                    
                    code += `# Create the complete model
inputs = tf.keras.Input(shape=(${inputSize}, ${inputSize}, 3))

# Data augmentation
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2),
])

x = data_augmentation(inputs)
x = preprocess_input(x)
x = base_model(x, training=False)\n\n`;
                    
                    // Add custom top
                    if (customTop === 'custom_dense') {
                        code += `# Custom classification head
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(${numClasses}, activation='softmax')(x)\n\n`;
                    } else {
                        code += `# Default classification head
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(${numClasses}, activation='softmax')(x)\n\n`;
                    }
                    
                    code += `# Build the model
model = tf.keras.Model(inputs, outputs)

# Model summary
print(f"Total layers: {len(model.layers)}")
print(f"Trainable layers: {sum([layer.trainable for layer in model.layers])}")
model.summary()`;
                    
                    return code;
    },

    generateMLCode(modelType, numClasses) {
        const mlConfig = state.mlConfig || {};
        const params = mlConfig.params || {};
        const preprocessing = mlConfig.preprocessing || { scaleFeatures: true, testSize: 0.2, randomState: 42 };

        const useScaling = preprocessing.scaleFeatures !== false;

        if (modelType === 'knn') {
            return `from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# K-Nearest Neighbors (KNN)
knn = KNeighborsClassifier(
    n_neighbors=${params.n_neighbors || 5},
    weights='${params.weights || 'uniform'}',
    algorithm='${params.algorithm || 'auto'}',
    metric='${params.metric || 'euclidean'}',
    n_jobs=-1
)

${useScaling ? `pipeline = Pipeline([("scaler", StandardScaler()), ("knn", knn)])` : `pipeline = knn`}

# Train
pipeline.fit(X_train, y_train)

# Predict
y_pred = pipeline.predict(X_test)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
print("\\nClassification Report:")
print(classification_report(y_test, y_pred))`;
        }

        if (modelType === 'svm') {
            return `from sklearn import svm
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# Support Vector Machine (SVM)
svc = svm.SVC(
    kernel='${params.kernel || 'rbf'}',
    C=${params.C || 1.0},
    gamma='${params.gamma || 'scale'}',
    probability=${params.probability === false ? 'False' : 'True'},
    random_state=${preprocessing.randomState || 42}
)

${useScaling ? `pipeline = Pipeline([("scaler", StandardScaler()), ("svm", svc)])` : `pipeline = svc`}

# Train
pipeline.fit(X_train, y_train)

# Predict
y_pred = pipeline.predict(X_test)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
print("\\nClassification Report:")
print(classification_report(y_test, y_pred))`;
        }

        if (modelType === 'randomforest') {
            return `from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Random Forest
model = RandomForestClassifier(
    n_estimators=${params.n_estimators || 100},
    max_depth=${params.max_depth || 'None'},
    min_samples_split=${params.min_samples_split || 2},
    min_samples_leaf=${params.min_samples_leaf || 1},
    max_features='${params.max_features || 'sqrt'}',
    bootstrap=True,
    random_state=${preprocessing.randomState || 42},
    n_jobs=-1
)

# Train
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
print("\\nClassification Report:")
print(classification_report(y_test, y_pred))`;
        }

        return '# Unsupported ML model type';
    },
    
    generatePythonScript() {
        const numClasses = parseInt(document.getElementById('numClasses')?.value || '10', 10);
        const epochs = parseInt(document.getElementById('epochs')?.value || '10', 10);
        const batchSize = parseInt(document.getElementById('batchSize')?.value || '32', 10);
        const lr = parseFloat(document.getElementById('lr')?.value || '0.001');
        const optimizer = (document.getElementById('optimizer')?.value || 'adam').toLowerCase();
        const lossFunction = document.getElementById('lossFunction')?.value || 'categorical_crossentropy';

        const modelCode = this.generateModelCode();
        const isML = state.currentMode !== 'custom' && state.model && models[state.model]?.type === 'ml';

        const header = `# DeepForge Studio - Exported Training Pipeline
# Generated: ${new Date().toISOString()}

`;

        const seedBlockDL = `import os, random
import numpy as np

def set_seed(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except Exception:
        pass

set_seed(42)

`;
const seedBlockML = `import os, random
import numpy as np

def set_seed(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

set_seed(42)

`;


                if (isML) {
                    const mlTestSize = parseFloat(state.mlConfig?.preprocessing?.testSize ?? 0.2);
                    const mlSeed = parseInt(state.mlConfig?.preprocessing?.randomState ?? 42, 10);

                    return header + seedBlockML + `
# ============================
# DATA LOADING (EDIT THIS)
# ============================
# Provide X (features) and y (labels) as numpy arrays.
# Example:
#   import pandas as pd
#   df = pd.read_csv("your_data.csv")
#   X = df.drop("label", axis=1).values
#   y = df["label"].values
#
# X = ...
# y = ...

# Guard: ensure X and y are defined before proceeding
try:
    X
    y
except NameError as e:
    raise NameError("Please define X and y before running. See the DATA LOADING section.") from e

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=${mlTestSize},
    random_state=${mlSeed}
)

` + modelCode + `

print("Done.")
`;                }

        // DL / Pretrained
        
        let inputSize = 224;
                if (state.modelMode === 'pretrained') {
                    inputSize = parseInt(document.getElementById('inputSize')?.value || '224', 10);
                } else if (state.model && state.currentMode !== 'custom') {
                    // Match default input sizes used by Keras Applications when training from scratch
                    if (state.model === 'inceptionv3') inputSize = 299;
                }

        const labelMode = (lossFunction.includes('sparse')) ? 'int' : 'categorical';

        const optExpr = (() => {
            if (optimizer === 'sgd') return `tf.keras.optimizers.SGD(learning_rate=${lr}, momentum=0.9)`;
            if (optimizer === 'rmsprop') return `tf.keras.optimizers.RMSprop(learning_rate=${lr})`;
            if (optimizer === 'adamw') return `tf.keras.optimizers.AdamW(learning_rate=${lr})`;
            return `tf.keras.optimizers.Adam(learning_rate=${lr})`;
        })();

        return header + seedBlockDL + modelCode + `

# ============================
# TRAINING PIPELINE (EDIT DATA PATH)
# ============================
import tensorflow as tf

# Expected folder structure:
# DATA_DIR/
#   class_a/
#   class_b/
#   ...
DATA_DIR = "path/to/your/image_dataset"
IMG_SIZE = (${inputSize}, ${inputSize})
BATCH_SIZE = ${batchSize}
EPOCHS = ${epochs}

train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="${labelMode}",
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="${labelMode}",
)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# Re-compile using UI-selected hyperparams (overrides template defaults safely)
model.compile(
    optimizer=${optExpr},
    loss="${lossFunction}",
    metrics=["accuracy"],
)

history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)

model.save("deepforge_model.keras")
print("Saved model to deepforge_model.keras")
`;
    },

        generateColabNotebook() {
        const pythonScript = this.generatePythonScript();

        const modelLabel = (() => {
            if (state.currentMode === 'custom') return 'Custom Model';
            if (state.model && models[state.model]?.name) return models[state.model].name;
            return state.model || 'Model';
        })();

        const modeLabel = (() => {
            if (state.currentMode === 'custom') return 'From Scratch';
            if (state.model && models[state.model]?.type === 'ml') return 'Classical ML';
            return (state.modelMode === 'pretrained') ? 'Pretrained' : 'From Scratch';
        })();

        const safe = (s) => String(s)
            .replace(/[^a-zA-Z0-9]+/g, '_')
            .replace(/^_+|_+$/g, '');

        const notebookName = `DeepForge_${safe(modelLabel)}_${safe(modeLabel.toLowerCase())}.ipynb`;

        const cells = [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    `# 🚀 DeepForge Studio - ${modelLabel} (${modeLabel})\n`,
                    "\n",
                    "**Auto-generated Training Notebook**\n",
                    "\n",
                    "### ⚡ Quick Start:\n",
                    "1. **Enable GPU:** `Runtime → Change runtime type → GPU`\n",
                    "2. **Run this notebook:** `Runtime → Run all`\n",
                    "3. **Set your dataset path** where indicated in the code\n4. Ensure your dataset follows the folder structure: `DATA_DIR/class_name/images...`\n",
                    "4. **Download the trained model** using the last cell\n",
                    "\n",
                    "---"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": null,
                "metadata": {},
                "outputs": [],
                "source": pythonScript
            },
            {
                "cell_type": "code",
                "execution_count": null,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Download the trained model (Google Colab only)\n",
                    "try:\n",
                    "    from google.colab import files\n",
                    "    files.download('deepforge_model.keras')\n",
                    "except Exception as e:\n",
                    "    print('Download is supported in Google Colab only:', e)\n"
                ]
            }
        ];

        return JSON.stringify({
            "nbformat": 4,
            "nbformat_minor": 0,
            "metadata": {
                "colab": {
                    "name": notebookName,
                    "provenance": []
                },
                "kernelspec": {
                    "name": "python3",
                    "display_name": "Python 3"
                },
                "accelerator": "GPU"
            },
            "cells": cells
        }, null, 2);
    }
};
