import { state } from './state.js';
import { models } from './config/models.js';
import { mlConfigurations, mlTask } from './config/ml-config.js';
import { modalityOf, inputShapeExpr, inputShapeNumeric, buildCustomGraph, headCode, checkInputSize, MODALITIES } from './architecture.js';
import { ARCHITECTURES } from './config/architectures.js';

export const codeGenerator = {
    generateCode() {
        let code;
        try {
            code = this.generateModelCode();
        } catch (err) {
            // The preview is refreshed on almost every UI event, so a generation
            // error must be shown rather than thrown: an uncaught error here
            // would leave the previous (wrong) code on screen.
            code = `# Code generation stopped\n# ${err.message}`;
        }
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
            return this.generateMLCode(state.model);
        }


        // Keras Applications architectures (from scratch)
        if (
            state.modelMode === 'scratch' &&
            state.model &&
            ['resnet50', 'mobilenet', 'efficientnet', 'inceptionv3', 'densenet'].includes(state.model) &&
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

    // Emits the model using the Keras functional API. Sequential could not
    // express residual connections, parallel branches, or multi-input layers,
    // which ruled out ResNet-style blocks, U-Net and Transformer alike.
    generateDLCode(numClasses) {
        const inputSize = this.imageInputSize();
        const modality = this.currentModality();
        if (state.currentMode !== 'custom' && state.model) {
            const problem = checkInputSize(state.model, inputSize);
            if (problem) throw new Error(problem);
        }
        const isCustom = state.currentMode === 'custom';

        let bodyCode, outputShape;
        if (isCustom) {
            const graph = buildCustomGraph(state.customLayers, state.customLayerConfigs, modality, inputSize);
            bodyCode = graph.code;
            outputShape = graph.outputShape;
        } else {
            const template = ARCHITECTURES[state.model];
            if (!template) {
                throw new Error(`No architecture template for "${state.model}". Please report this as a bug.`);
            }
            bodyCode = template();
            outputShape = null;
        }

        const head = headCode(modality, numClasses, outputShape);
        const shapeExpr = inputShapeExpr(modality, inputSize);

        let code = `import tensorflow as tf\nfrom tensorflow.keras import layers, models\n\n`;
        if (modality !== 'image' && modality !== 'segmentation') {
            code += `# Shapes marked in CAPITALS are defined by the data section of the\n`;
            code += `# exported script, so this model adapts to your dataset.\n`;
        }
        code += `inputs = layers.Input(shape=${shapeExpr})\nx = inputs\n\n`;
        code += bodyCode.trimEnd() + `\n\n`;
        if (head.note) code += `# ${head.note}\n`;
        code += head.code + `\n`;
        code += `model = models.Model(inputs, outputs, name='${isCustom ? 'custom_model' : state.model}')\n\n`;
        code += `# Summary\nmodel.summary()\n`;
        code += `\n# NOTE: Compilation and training steps are included in the exported training pipeline.\n`;
        return code;
    },

    currentModality() {
        return modalityOf(state.model, state.customLayers, state.currentMode);
    },

    // Single source of truth for the image input size. The model definition and
    // the data pipeline previously read it from different places, so any value
    // other than the default produced a model and a dataset that disagreed.
    imageInputSize() {
        if (state.model === 'inceptionv3') {
            return Math.max(75, parseInt(document.getElementById('inputSize')?.value || '299', 10));
        }
        return parseInt(document.getElementById('inputSize')?.value || '224', 10);
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

        const appClass = appMap[modelType];
        const inputSize = this.imageInputSize();
        const problem = checkInputSize(modelType, inputSize);
        if (problem) throw new Error(problem);

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

generatePretrainedCode(modelType, numClasses) {
                        const freezeLayers = document.getElementById('freezeLayers')?.value || 'base';
                    const inputSize = this.imageInputSize();
                    const sizeProblem = checkInputSize(modelType, inputSize);
                    if (sizeProblem) throw new Error(sizeProblem);
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
                    
                    // Handle freezing. Every option offered in the interface must
                    // appear here: 'all_but_last' was previously absent, so selecting
                    // it silently produced a fully trainable backbone.
                    if (freezeLayers === 'base') {
                        code += `# Freeze the base model layers
base_model.trainable = False\n\n`;
                    } else if (freezeLayers === 'partial') {
                        code += `# Freeze first 50% of layers
for layer in base_model.layers[:len(base_model.layers)//2]:
    layer.trainable = False\n\n`;
                    } else if (freezeLayers === 'all_but_last') {
                        code += `# Freeze every layer of the backbone except the last one that has
# trainable weights. Taking layers[:-1] literally would be a no-op for
# most backbones, whose final layer is a pooling layer with no weights.
_weighted = [i for i, l in enumerate(base_model.layers) if l.weights]
for layer in base_model.layers[:_weighted[-1]]:
    layer.trainable = False
print(f"Unfrozen backbone layer: {base_model.layers[_weighted[-1]].name}")\n\n`;
                    } else if (freezeLayers === 'none') {
                        // Training a pretrained backbone end to end at a head-sized
                        // learning rate destroys the features that make it useful.
                        // Say so in the export rather than letting it fail silently.
                        code += `# NOTE: no layers are frozen, so the pretrained backbone will be
# fine-tuned end to end. Pretrained weights are easily destroyed at a
# learning rate chosen for a randomly initialised head. If this is
# intentional, use a much smaller learning rate (of the order of 1e-5);
# otherwise select "Freeze base model" in the interface.\n\n`;
                    } else {
                        throw new Error(`Freezing option "${freezeLayers}" is offered in the interface but has no implementation. Please report this as a bug.`);
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

    // -------------------------------------------------
    // CLASSICAL ML
    // -------------------------------------------------
    // One builder per model exposed in js/config/models.js. Adding a model to
    // that file without adding a builder here is a defect; mlBuildersCoverModels()
    // below is asserted by the test harness so the two cannot drift apart again.
    mlBuilders: {
        knn(p, scale) {
            return {
                imports: ['from sklearn.neighbors import KNeighborsClassifier'],
                estimator: `KNeighborsClassifier(
    n_neighbors=${p.n_neighbors || 5},
    weights='${p.weights || 'uniform'}',
    algorithm='${p.algorithm || 'auto'}',
    metric='${p.metric || 'euclidean'}',
    n_jobs=-1
)`,
                scaler: scale ? 'standard' : null
            };
        },
        svm(p, scale, seed) {
            return {
                imports: ['from sklearn.svm import SVC'],
                estimator: `SVC(
    kernel='${p.kernel || 'rbf'}',
    C=${p.C || 1.0},
    gamma=${(g => (g === 'scale' || g === 'auto') ? `'${g}'` : g)(p.gamma || 'scale')},
    probability=${p.probability === false ? 'False' : 'True'},
    random_state=${seed}
)`,
                scaler: scale ? 'standard' : null
            };
        },
        randomforest(p, scale, seed) {
            return {
                imports: ['from sklearn.ensemble import RandomForestClassifier'],
                estimator: `RandomForestClassifier(
    n_estimators=${p.n_estimators || 100},
    max_depth=${p.max_depth || 'None'},
    min_samples_split=${p.min_samples_split || 2},
    random_state=${seed},
    n_jobs=-1
)`,
                // Tree ensembles are scale-invariant; scaling is deliberately skipped.
                scaler: null
            };
        },
        decisiontree(p, scale, seed) {
            return {
                imports: ['from sklearn.tree import DecisionTreeClassifier'],
                estimator: `DecisionTreeClassifier(
    criterion='${p.criterion || 'gini'}',
    max_depth=${p.max_depth || 'None'},
    min_samples_split=${p.min_samples_split || 2},
    min_samples_leaf=${p.min_samples_leaf || 1},
    random_state=${seed}
)`,
                scaler: null
            };
        },
        xgboost(p, scale, seed) {
            return {
                note: '# Requires the xgboost package:  pip install xgboost',
                imports: ['from xgboost import XGBClassifier'],
                estimator: `XGBClassifier(
    n_estimators=${p.n_estimators || 100},
    max_depth=${p.max_depth || 6},
    learning_rate=${p.learning_rate || 0.3},
    subsample=${p.subsample || 1.0},
    random_state=${seed},
    eval_metric='mlogloss',
    n_jobs=-1
)`,
                scaler: null
            };
        },
        naivebayes(p, scale) {
            const variant = p.variant || 'gaussian';
            const cls = { gaussian: 'GaussianNB', multinomial: 'MultinomialNB', bernoulli: 'BernoulliNB' }[variant];
            const args = variant === 'gaussian' ? '' : `alpha=${p.alpha || 1.0}`;
            return {
                imports: [`from sklearn.naive_bayes import ${cls}`],
                estimator: `${cls}(${args})`,
                // MultinomialNB requires non-negative features, so standardisation
                // would break it; min-max keeps the data in [0, 1].
                scaler: scale ? (variant === 'gaussian' ? 'standard' : 'minmax') : null
            };
        },
        logisticregression(p, scale, seed) {
            const penalty = p.penalty === 'none' ? 'None' : `'${p.penalty || 'l2'}'`;
            let solver = p.solver || 'lbfgs';
            // lbfgs does not support the l1 penalty; saga does and, unlike
            // liblinear, also handles more than two classes.
            if (p.penalty === 'l1' && solver !== 'saga') solver = 'saga';
            return {
                imports: ['from sklearn.linear_model import LogisticRegression'],
                estimator: `LogisticRegression(
    C=${p.C || 1.0},
    penalty=${penalty},
    solver='${solver}',
    max_iter=${p.max_iter || 1000},
    random_state=${seed}
)`,
                scaler: scale ? 'standard' : null
            };
        },
        linearregression(p, scale) {
            return {
                imports: ['from sklearn.linear_model import LinearRegression'],
                estimator: `LinearRegression(
    fit_intercept=${p.fit_intercept === false ? 'False' : 'True'},
    positive=${p.positive === true ? 'True' : 'False'}
)`,
                scaler: scale ? 'standard' : null
            };
        },
        kmeans(p, scale, seed) {
            return {
                imports: ['from sklearn.cluster import KMeans'],
                estimator: `KMeans(
    # clamped: cannot form more clusters than there are samples
    n_clusters=min(${p.n_clusters || 3}, len(X)),
    init='${p.init || 'k-means++'}',
    n_init=${p.n_init || 10},
    max_iter=${p.max_iter || 300},
    random_state=${seed}
)`,
                scaler: scale ? 'standard' : null
            };
        },
        pca(p, scale) {
            return {
                imports: ['from sklearn.decomposition import PCA'],
                estimator: `PCA(
    # clamped: n_components cannot exceed the number of input features
    n_components=min(${p.n_components || 2}, X.shape[1]),
    svd_solver='${p.svd_solver || 'auto'}',
    whiten=${p.whiten === true ? 'True' : 'False'}
)`,
                scaler: scale ? 'standard' : null
            };
        }
    },

    generateMLCode(modelType) {
        const build = this.mlBuilders[modelType];
        if (!build) {
            // Reaching here means models.js offers something mlBuilders does not
            // implement. Fail loudly rather than emitting a placeholder comment.
            throw new Error(`No code generator for the model "${modelType}". Please report this as a bug.`);
        }

        const mlConfig = state.mlConfig || {};
        const params = mlConfig.params || {};
        const preprocessing = mlConfig.preprocessing || {};
        const seed = parseInt(preprocessing.randomState ?? 42, 10);
        const scale = preprocessing.scaleFeatures !== false;
        const task = mlTask(modelType);

        const spec = build(params, scale, seed);
        const scalerImport = spec.scaler === 'minmax'
            ? 'from sklearn.preprocessing import MinMaxScaler'
            : 'from sklearn.preprocessing import StandardScaler';
        const scalerCall = spec.scaler === 'minmax' ? 'MinMaxScaler()' : 'StandardScaler()';

        const metricImports = {
            classification: 'from sklearn.metrics import accuracy_score, classification_report, confusion_matrix',
            regression: 'from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error',
            clustering: 'from sklearn.metrics import silhouette_score',
            decomposition: null
        }[task];

        const imports = [...spec.imports];
        if (spec.scaler) imports.push('from sklearn.pipeline import Pipeline', scalerImport);
        if (metricImports) imports.push(metricImports);
        if (task === 'decomposition') imports.push('import numpy as np');

        let code = '';
        if (spec.note) code += `${spec.note}\n`;
        code += imports.join('\n') + '\n\n';
        code += `# ${mlConfigurations[modelType]?.name || modelType}\nestimator = ${spec.estimator}\n\n`;
        code += spec.scaler
            ? `model = Pipeline([("scaler", ${scalerCall}), ("estimator", estimator)])\n\n`
            : `model = estimator\n\n`;

        if (task === 'classification') {
            code += `# Train
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\\nClassification report:")
print(classification_report(y_test, y_pred))
print("Confusion matrix:")
print(confusion_matrix(y_test, y_pred))`;
        } else if (task === 'regression') {
            code += `# Train
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
rmse = mean_squared_error(y_test, y_pred) ** 0.5
print(f"R2:   {r2_score(y_test, y_pred):.4f}")
print(f"MAE:  {mean_absolute_error(y_test, y_pred):.4f}")
print(f"RMSE: {rmse:.4f}")`;
        } else if (task === 'clustering') {
            code += `# Fit (unsupervised: no labels are used)
labels = model.fit_predict(X)

# Evaluate
inertia = model[-1].inertia_ if hasattr(model, "__getitem__") else model.inertia_
print(f"Inertia: {inertia:.4f}")
if len(set(labels)) > 1:
    print(f"Silhouette score: {silhouette_score(X, labels):.4f}")
else:
    print("Silhouette score undefined: all samples fell into one cluster.")`;
        } else {
            code += `# Fit and transform (unsupervised: no labels are used)
X_reduced = model.fit_transform(X)

# Evaluate
pca_step = model[-1] if hasattr(model, "__getitem__") else model
ratios = pca_step.explained_variance_ratio_
print(f"Reduced shape: {X_reduced.shape}")
for i, r in enumerate(ratios):
    print(f"  PC{i + 1}: {r:.4f} of variance")
print(f"Total variance retained: {np.sum(ratios):.4f}")`;
        }

        return code;
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

        const header = `# Layernaut Studio - Exported Training Pipeline
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



const envLockBlock = `
# ============================
# ENVIRONMENT RECORD
# ============================
# Seeding constrains but does not eliminate variation (see ENVIRONMENT.md), so
# the run records the environment it actually executed in.
def _write_environment_lock(path="environment_lock.txt"):
    import platform, sys
    lines = [f"python: {sys.version.split()[0]} ({platform.platform()})"]
    from importlib.metadata import version, PackageNotFoundError
    # A distribution may be installed under any of several names
    # (tensorflow / tensorflow-cpu / tensorflow-macos), so try each, then fall
    # back to the imported module's own __version__.
    for module, dists in (("tensorflow", ("tensorflow", "tensorflow-cpu", "tensorflow-macos", "tensorflow-gpu")),
                          ("numpy", ("numpy",)),
                          ("sklearn", ("scikit-learn",)),
                          ("xgboost", ("xgboost",))):
        found = None
        for dist in dists:
            try:
                found = version(dist)
                break
            except PackageNotFoundError:
                continue
        if found is None:
            try:
                found = __import__(module).__version__
            except Exception:
                found = "not installed"
        lines.append(f"{module}: {found}")
    try:
        import tensorflow as _tf
        lines.append(f"devices: {[d.name for d in _tf.config.list_physical_devices()]}")
    except Exception:
        pass
    with open(path, "w") as fh:
        fh.write("\\n".join(lines) + "\\n")
    print("Environment recorded in", path)

_write_environment_lock()

`;
                if (isML) {
                    const task = mlTask(state.model);
                    const mlTestSize = parseFloat(state.mlConfig?.preprocessing?.testSize ?? 0.2);
                    const mlSeed = parseInt(state.mlConfig?.preprocessing?.randomState ?? 42, 10);
                    const supervised = (task === 'classification' || task === 'regression');

                    // Unsupervised models (K-Means, PCA) have no labels, so a
                    // train/test split and accuracy metrics do not apply to them.
                    const dataBlock = supervised
                        ? `
# ============================
# DATA LOADING (EDIT THIS)
# ============================
# Provide X (features) and y (${task === 'regression' ? 'targets' : 'labels'}) as numpy arrays.
# Example:
#   import pandas as pd
#   df = pd.read_csv("your_data.csv")
#   X = df.drop("${task === 'regression' ? 'target' : 'label'}", axis=1).values
#   y = df["${task === 'regression' ? 'target' : 'label'}"].values
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

`
                        : `
# ============================
# DATA LOADING (EDIT THIS)
# ============================
# ${task === 'clustering' ? 'Clustering' : 'Dimensionality reduction'} is unsupervised:
# provide X (features) only. No labels and no train/test split are required.
# Example:
#   import pandas as pd
#   df = pd.read_csv("your_data.csv")
#   X = df.values
#
# X = ...

# Guard: ensure X is defined before proceeding
try:
    X
except NameError as e:
    raise NameError("Please define X before running. See the DATA LOADING section.") from e

`;

                    return header + seedBlockML + envLockBlock + dataBlock + modelCode + `

print("Done.")
`;                }

        // DL / Pretrained
        const modality = (state.modelMode === 'pretrained') ? 'image' : this.currentModality();

        const inputSize = this.imageInputSize();

        const labelMode = (lossFunction.includes('sparse')) ? 'int' : 'categorical';

        const optExpr = (() => {
            if (optimizer === 'sgd') return `tf.keras.optimizers.SGD(learning_rate=${lr}, momentum=0.9)`;
            if (optimizer === 'rmsprop') return `tf.keras.optimizers.RMSprop(learning_rate=${lr})`;
            if (optimizer === 'adamw') return `tf.keras.optimizers.AdamW(learning_rate=${lr})`;
            return `tf.keras.optimizers.Adam(learning_rate=${lr})`;
        })();

        // v1.0.x emitted an image-folder pipeline for every architecture, so the
        // sequence, reconstruction and segmentation models generated code that
        // constructed and then failed at fit(). Each modality now gets the data
        // section it actually needs, and the section precedes the model so the
        // model can be defined in terms of the data's real shape.
        const dataSections = {
            image: `# ============================
# DATA (EDIT DATA_DIR)
# ============================
# Expected folder structure:
#   DATA_DIR/class_a/... , DATA_DIR/class_b/... , ...
DATA_DIR = "path/to/your/image_dataset"
IMG_SIZE = (${inputSize}, ${inputSize})
BATCH_SIZE = ${batchSize}
EPOCHS = ${epochs}

train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR, validation_split=0.2, subset="training", seed=42,
    image_size=IMG_SIZE, batch_size=BATCH_SIZE, label_mode="${labelMode}",
)
val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR, validation_split=0.2, subset="validation", seed=42,
    image_size=IMG_SIZE, batch_size=BATCH_SIZE, label_mode="${labelMode}",
)

# The classification head is built for the class count set in the interface.
# The tool cannot see your dataset, so the check happens here, where the data
# is: without it a mismatch surfaces as an opaque shape error inside the loss
# function several minutes into training.
NUM_CLASSES = ${numClasses}
_found = len(train_ds.class_names)
if _found != NUM_CLASSES:
    raise ValueError(
        f"This script was exported for {NUM_CLASSES} classes, but {DATA_DIR} "
        f"contains {_found}: {train_ds.class_names}. "
        f"Re-export with 'Number of Classes' set to {_found}, or point DATA_DIR "
        f"at a dataset with {NUM_CLASSES} classes."
    )
print(f"{_found} classes: {train_ds.class_names}")

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
`,

            segmentation: `# ============================
# DATA (EDIT THE TWO PATHS)
# ============================
# Segmentation needs one mask per image, not one label per image.
# Masks must be single-channel, with pixel values 0..${numClasses - 1}.
IMAGE_DIR = "path/to/images"
MASK_DIR = "path/to/masks"
IMG_SIZE = (${inputSize}, ${inputSize})
BATCH_SIZE = ${batchSize}
EPOCHS = ${epochs}

import pathlib
image_paths = sorted(str(p) for p in pathlib.Path(IMAGE_DIR).glob("*"))
mask_paths = sorted(str(p) for p in pathlib.Path(MASK_DIR).glob("*"))
assert len(image_paths) == len(mask_paths), "Each image needs exactly one mask."

def _load(image_path, mask_path):
    image = tf.image.resize(tf.io.decode_image(tf.io.read_file(image_path), channels=3, expand_animations=False), IMG_SIZE) / 255.0
    mask = tf.image.resize(tf.io.decode_image(tf.io.read_file(mask_path), channels=1, expand_animations=False), IMG_SIZE, method="nearest")
    return image, tf.one_hot(tf.cast(tf.squeeze(mask, -1), tf.int32), ${numClasses})

ds = tf.data.Dataset.from_tensor_slices((image_paths, mask_paths)).map(_load, num_parallel_calls=tf.data.AUTOTUNE)
n_val = max(1, int(0.2 * len(image_paths)))
val_ds = ds.take(n_val).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
train_ds = ds.skip(n_val).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
`,

            sequence: `# ============================
# DATA (EDIT THIS)
# ============================
# Provide X of shape (n_samples, timesteps, features) and y of shape (n_samples,).
# Example, windowing a univariate series:
#   import numpy as np
#   series = np.load("your_series.npy")
#   X = np.stack([series[i:i + 100] for i in range(len(series) - 100)])[..., None]
#   y = labels[100:]
#
# X = ...
# y = ...

try:
    X, y
except NameError as e:
    raise NameError("Please define X and y before running. See the DATA section.") from e

import numpy as np
from sklearn.model_selection import train_test_split

X = np.asarray(X)
SEQ_LEN, N_FEATURES = X.shape[1], X.shape[2]
BATCH_SIZE = ${batchSize}
EPOCHS = ${epochs}

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
`,

            tabular: `# ============================
# DATA (EDIT THIS)
# ============================
# Provide X of shape (n_samples, n_features) and y of shape (n_samples,).
# Example:
#   import pandas as pd
#   df = pd.read_csv("your_data.csv")
#   X = df.drop("label", axis=1).values
#   y = df["label"].values
#
# X = ...
# y = ...

try:
    X, y
except NameError as e:
    raise NameError("Please define X and y before running. See the DATA section.") from e

import numpy as np
from sklearn.model_selection import train_test_split

X = np.asarray(X, dtype="float32")
N_FEATURES = X.shape[1]
BATCH_SIZE = ${batchSize}
EPOCHS = ${epochs}

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
`,

            reconstruction: `# ============================
# DATA (EDIT THIS)
# ============================
# An autoencoder is trained to reproduce its own input, so no labels are needed.
# Provide X of shape (n_samples, n_features), scaled to [0, 1].
# Example:
#   (X, _), _ = tf.keras.datasets.mnist.load_data()
#   X = X.reshape(len(X), -1) / 255.0
#
# X = ...

try:
    X
except NameError as e:
    raise NameError("Please define X before running. See the DATA section.") from e

import numpy as np
from sklearn.model_selection import train_test_split

X = np.asarray(X, dtype="float32")
N_FEATURES = X.shape[1]
LATENT_DIM = ${Math.max(2, Math.min(64, numClasses * 4))}
BATCH_SIZE = ${batchSize}
EPOCHS = ${epochs}

X_train, X_val = train_test_split(X, test_size=0.2, random_state=42)
`
        };

        // Transformer needs vocabulary and block sizes alongside the sequence.
        const transformerConsts = (state.model === 'transformer' && state.currentMode !== 'custom')
            ? `
# Transformer hyperparameters
VOCAB_SIZE = 10000   # set to your tokenizer's vocabulary size
EMBED_DIM = 128
N_HEADS = 4
FF_DIM = 256
N_BLOCKS = 2
`
            : '';

        // Token sequences are integer ids of shape (samples, timesteps), so the
        // generic sequence loader's feature axis does not apply.
        const dataSection = (state.model === 'transformer' && state.currentMode !== 'custom')
            ? dataSections.sequence.replace('SEQ_LEN, N_FEATURES = X.shape[1], X.shape[2]', 'SEQ_LEN = X.shape[1]')
                                   .replace('# Provide X of shape (n_samples, timesteps, features) and y of shape (n_samples,).',
                                            '# Provide X of integer token ids, shape (n_samples, timesteps), and y of shape (n_samples,).')
            : dataSections[modality];

        const fitCall = {
            image: 'history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)',
            segmentation: 'history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)',
            sequence: 'history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=EPOCHS, batch_size=BATCH_SIZE)',
            tabular: 'history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=EPOCHS, batch_size=BATCH_SIZE)',
            text: 'history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=EPOCHS, batch_size=BATCH_SIZE)',
            reconstruction: '# The target is the input itself.\nhistory = model.fit(X_train, X_train, validation_data=(X_val, X_val), epochs=EPOCHS, batch_size=BATCH_SIZE)'
        }[modality];
        if (!fitCall) throw new Error(`No training pipeline defined for modality "${modality}". Please report this as a bug.`);

        // A reconstruction model has no classes, so accuracy is meaningless.
        const effectiveLoss = (modality === 'reconstruction') ? 'mse' : lossFunction;
        const metrics = (modality === 'reconstruction') ? '["mae"]' : '["accuracy"]';

        return header + seedBlockDL + `import tensorflow as tf
` + envLockBlock + dataSection + transformerConsts + `
` + modelCode + `
# ============================
# TRAINING
# ============================
model.compile(
    optimizer=${optExpr},
    loss="${effectiveLoss}",
    metrics=${metrics},
)

${fitCall}

model.save("layernaut_model.keras")
print("Saved model to layernaut_model.keras")
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

        const notebookName = `Layernaut_${safe(modelLabel)}_${safe(modeLabel.toLowerCase())}.ipynb`;

        const cells = [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    `# 🚀 Layernaut Studio - ${modelLabel} (${modeLabel})\n`,
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
                    "    files.download('layernaut_model.keras')\n",
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
