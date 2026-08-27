// Architecture model for the code generator.
//
// v1.0.x emitted `models.Sequential([...])`, which made three things impossible:
// residual/skip connections, any parallel branch, and layers that take more than
// one input tensor (MultiHeadAttention). It also emitted a single image-folder
// data pipeline for every architecture, so sequence and reconstruction models
// generated code that constructed but crashed at fit().
//
// This module replaces that with a small typed graph: every layer declares how
// it emits functional-API code and how it transforms the tensor shape, so the
// shape is known at design time and invalid stacks are rejected before export.

// ---------------------------------------------------------------------------
// Modalities
// ---------------------------------------------------------------------------
// A modality fixes the input shape, the output head, the loss and the data
// pipeline. Every architecture must declare one.
export const MODALITIES = {
    image: { label: 'Image classification', head: 'softmax', defaultLoss: 'categorical_crossentropy' },
    sequence: { label: 'Sequence classification', head: 'softmax', defaultLoss: 'categorical_crossentropy' },
    text: { label: 'Token-sequence classification', head: 'softmax', defaultLoss: 'categorical_crossentropy' },
    tabular: { label: 'Tabular classification', head: 'softmax', defaultLoss: 'categorical_crossentropy' },
    reconstruction: { label: 'Reconstruction (autoencoder)', head: 'none', defaultLoss: 'mse' },
    segmentation: { label: 'Semantic segmentation', head: 'pixelwise', defaultLoss: 'categorical_crossentropy' }
};

const MODEL_MODALITY = {
    alexnet: 'image', vgg16: 'image', simple_cnn: 'image', resnet50: 'image',
    mobilenet: 'image', efficientnet: 'image', inceptionv3: 'image', densenet: 'image',
    lstm: 'sequence', gru: 'sequence', transformer: 'text',
    autoencoder: 'reconstruction',
    unet: 'segmentation'
};

// Custom stacks have no declared modality, so infer it from the first layer that
// implies one. Conv2D means spatial input; a recurrent layer means sequential.
export function inferModality(customLayers) {
    for (const layer of customLayers) {
        if (layer === 'Conv2D' || layer === 'MaxPool' || layer === 'AvgPool' || layer === 'GlobalAvgPool') return 'image';
        if (layer === 'LSTM' || layer === 'GRU') return 'sequence';
    }
    return 'tabular';
}

export function modalityOf(modelKey, customLayers, currentMode) {
    if (currentMode === 'custom') return inferModality(customLayers || []);
    return MODEL_MODALITY[modelKey] || 'image';
}

// Symbolic input shapes. Image size is a UI choice, so it is substituted; the
// other dimensions are defined by the exported data section, which always
// precedes the model, so the generated code stays correct for any dataset.
export function inputShapeExpr(modality, inputSize) {
    switch (modality) {
        case 'image':
        case 'segmentation': return `(${inputSize}, ${inputSize}, 3)`;
        case 'sequence': return `(SEQ_LEN, N_FEATURES)`;
        case 'text': return `(SEQ_LEN,)`;
        case 'reconstruction': return `(N_FEATURES,)`;
        default: return `(N_FEATURES,)`;
    }
}

// Numeric counterpart used for design-time shape tracking. Symbolic dimensions
// are represented as null and simply propagate.
export function inputShapeNumeric(modality, inputSize) {
    switch (modality) {
        case 'image':
        case 'segmentation': return [Number(inputSize) || 224, Number(inputSize) || 224, 3];
        case 'sequence': return [null, null];
        case 'text': return [null];
        default: return [null];
    }
}

// ---------------------------------------------------------------------------
// Layer definitions
// ---------------------------------------------------------------------------
// emit(config)  -> the Keras expression, applied functionally as expr(input)
// shape(in, cfg) -> the output shape, or an Error message string if invalid
const out = (h, k, s, padding) => {
    if (h === null) return null;
    const eff = padding === 'same' ? Math.ceil(h / s) : Math.floor((h - k) / s) + 1;
    return eff > 0 ? eff : NaN;
};

export const LAYERS = {
    Conv2D: {
        rank: 4,
        emit: (c) => `layers.Conv2D(${c.filters || 32}, (${c.kernel_size || 3}, ${c.kernel_size || 3}), strides=${c.stride || 1}, padding='${c.padding || 'valid'}', activation='${c.activation || 'relu'}')`,
        shape: (s, c) => {
            if (s.length !== 3) return 'Conv2D needs a 2-D feature map with channels. Add it before any Flatten or Dense layer.';
            const k = Number(c.kernel_size || 3), st = Number(c.stride || 1), p = c.padding || 'valid';
            const h = out(s[0], k, st, p), w = out(s[1], k, st, p);
            if (Number.isNaN(h) || Number.isNaN(w)) return `Conv2D with kernel ${k} is larger than the ${s[0]}x${s[1]} feature map reaching it. Use a smaller kernel, padding='same', or fewer pooling layers.`;
            return [h, w, Number(c.filters || 32)];
        }
    },
    MaxPool: {
        rank: 4,
        emit: (c) => `layers.MaxPooling2D((${c.pool_size || 2}, ${c.pool_size || 2}), strides=${c.stride || 2}, padding='${c.padding || 'valid'}')`,
        shape: (s, c) => poolShape(s, c, 'MaxPool')
    },
    AvgPool: {
        rank: 4,
        emit: (c) => `layers.AveragePooling2D((${c.pool_size || 2}, ${c.pool_size || 2}), strides=${c.stride || 2}, padding='${c.padding || 'valid'}')`,
        shape: (s, c) => poolShape(s, c, 'AvgPool')
    },
    GlobalAvgPool: {
        emit: () => `layers.GlobalAveragePooling2D()`,
        shape: (s) => s.length === 3 ? [s[2]] : 'GlobalAvgPool needs a 2-D feature map. It only follows convolutional or pooling layers.'
    },
    Flatten: {
        emit: () => `layers.Flatten()`,
        shape: (s) => s.length === 1 ? s : [s.every(d => d !== null) ? s.reduce((a, b) => a * b, 1) : null]
    },
    Dense: {
        emit: (c) => `layers.Dense(${c.units || 128}, activation='${c.activation || 'relu'}')`,
        shape: (s, c) => s.length > 1
            ? 'Dense would be applied to every position of a multi-dimensional tensor. Add Flatten or GlobalAvgPool before it.'
            : [Number(c.units || 128)]
    },
    Dropout: {
        emit: (c) => `layers.Dropout(${c.rate || 0.5})`,
        shape: (s) => s
    },
    BatchNorm: {
        emit: (c) => `layers.BatchNormalization(momentum=${c.momentum || 0.99}, epsilon=${c.epsilon || 0.001})`,
        shape: (s) => s
    },
    LSTM: {
        emit: (c) => `layers.LSTM(${c.units || 128}, return_sequences=${c.return_sequences ? 'True' : 'False'})`,
        shape: (s, c) => s.length !== 2
            ? 'LSTM needs a sequence input of shape (timesteps, features).'
            : (c.return_sequences ? [s[0], Number(c.units || 128)] : [Number(c.units || 128)])
    },
    GRU: {
        emit: (c) => `layers.GRU(${c.units || 64}, return_sequences=${c.return_sequences ? 'True' : 'False'})`,
        shape: (s, c) => s.length !== 2
            ? 'GRU needs a sequence input of shape (timesteps, features).'
            : (c.return_sequences ? [s[0], Number(c.units || 64)] : [Number(c.units || 64)])
    }
};

function poolShape(s, c, name) {
    if (s.length !== 3) return `${name} needs a 2-D feature map with channels.`;
    const k = Number(c.pool_size || 2), st = Number(c.stride || 2), p = c.padding || 'valid';
    const h = out(s[0], k, st, p), w = out(s[1], k, st, p);
    if (Number.isNaN(h) || Number.isNaN(w)) return `${name} cannot reduce a ${s[0]}x${s[1]} feature map any further. Remove a pooling layer or enlarge the input.`;
    return [h, w, s[2]];
}

const sameShape = (a, b) => a.length === b.length && a.every((d, i) => d === null || b[i] === null || d === b[i]);
const fmt = (s) => `(${s.join(', ')})`;

// ---------------------------------------------------------------------------
// Graph builder
// ---------------------------------------------------------------------------
// Emits functional-API code for a custom stack. Each layer may declare
// `skipFrom`: the index of an earlier layer whose output is merged back in with
// Add or Concatenate — the mechanism residual networks are built from.
//
// Returns { code, outputShape } or throws with a message the UI can display.
export function buildCustomGraph(customLayers, customLayerConfigs, modality, inputSize) {
    if (!customLayers.length) {
        throw new Error('Add at least one layer in the Custom Builder before generating code.');
    }

    let shape = inputShapeNumeric(modality, inputSize);
    const lines = [];
    const shapes = [shape];   // shapes[i] is the tensor available *before* layer i
    let prev = 'x';

    customLayers.forEach((type, i) => {
        const def = LAYERS[type];
        if (!def) {
            throw new Error(`Layer type "${type}" is offered in the builder but has no code generator. Please report this as a bug.`);
        }
        const cfg = customLayerConfigs[i] || {};
        const next = def.shape(shape, cfg);
        if (typeof next === 'string') {
            throw new Error(`Layer ${i + 1} (${type}): ${next}`);
        }
        lines.push(`x = ${def.emit(cfg)}(${prev})`);
        shape = next;
        prev = 'x';

        // Skip connection: merge the output of an earlier layer back in.
        const from = cfg.skipFrom;
        if (from !== undefined && from !== null && from !== '' && Number(from) >= 0) {
            const j = Number(from);
            if (j >= i) throw new Error(`Layer ${i + 1} (${type}): a skip connection must come from an earlier layer.`);
            const srcShape = shapes[j + 1];
            const merge = cfg.skipMerge === 'concat' ? 'Concatenate' : 'Add';
            if (merge === 'Add' && !sameShape(shape, srcShape)) {
                throw new Error(`Layer ${i + 1} (${type}): cannot add the output of layer ${j + 1}. Shapes ${fmt(srcShape)} and ${fmt(shape)} differ. Use Concatenate, or match the shapes.`);
            }
            if (merge === 'Concatenate' && (shape.length !== srcShape.length || !sameShape(shape.slice(0, -1), srcShape.slice(0, -1)))) {
                throw new Error(`Layer ${i + 1} (${type}): cannot concatenate with layer ${j + 1}. All dimensions except the last must match; got ${fmt(srcShape)} and ${fmt(shape)}.`);
            }
            lines.push(`x = layers.${merge}(name='merge_${i + 1}_${j + 1}')([x, skip_${j + 1}])`);
            if (merge === 'Concatenate') shape = [...shape.slice(0, -1), shape[shape.length - 1] + srcShape[srcShape.length - 1]];
        }

        shapes.push(shape);

        // Anything referenced by a later skip must be kept in a named variable.
        const referencedLater = customLayers.some((_, k) => k > i && Number(customLayerConfigs[k]?.skipFrom) === i);
        if (referencedLater) lines.push(`skip_${i + 1} = x`);
    });

    return { code: lines.join('\n'), outputShape: shape };
}

// ---------------------------------------------------------------------------
// Output head
// ---------------------------------------------------------------------------
// v1.0.x appended a softmax classifier to every architecture, which turned the
// autoencoder into a classifier and destroyed the U-Net's pixel-wise output.
export function headCode(modality, numClasses, outputShape) {
    switch (modality) {
        case 'reconstruction':
            return { code: `outputs = layers.Dense(N_FEATURES, activation='sigmoid', name='reconstruction')(x)`, note: 'Reconstruction output: the model is trained to reproduce its input, so there is no class head.' };
        case 'segmentation':
            return { code: `outputs = layers.Conv2D(${numClasses}, (1, 1), activation='softmax', name='segmentation')(x)`, note: 'Pixel-wise output: one class distribution per pixel.' };
        default: {
            const needsPool = outputShape && outputShape.length > 1;
            const pre = needsPool ? `x = layers.Flatten()(x)\n` : '';
            return { code: `${pre}outputs = layers.Dense(${numClasses}, activation='softmax', name='predictions')(x)`, note: null };
        }
    }
}

// ---------------------------------------------------------------------------
// Architecture input constraints
// ---------------------------------------------------------------------------
// Downsampling stacks cannot accept arbitrarily small inputs: AlexNet's three
// 3x3/stride-2 pools exhaust a 64px image, and U-Net's encoder/decoder only
// realign if the input divides by 8. Checking this here produces an actionable
// message instead of a Keras error from inside the model.
export const ARCH_CONSTRAINTS = {
    alexnet: { minInput: 128 },
    vgg16: { minInput: 32 },
    simple_cnn: { minInput: 32 },
    unet: { minInput: 32, multipleOf: 8 },
    resnet50: { minInput: 32 },
    mobilenet: { minInput: 32 },
    efficientnet: { minInput: 32 },
    densenet: { minInput: 32 },
    inceptionv3: { minInput: 75 }
};

export function checkInputSize(modelKey, inputSize) {
    const c = ARCH_CONSTRAINTS[modelKey];
    if (!c) return null;
    const n = Number(inputSize);
    if (c.minInput && n < c.minInput) {
        return `${modelKey} needs an input of at least ${c.minInput}x${c.minInput} pixels; its downsampling stages exhaust a ${n}x${n} image. Increase the input size.`;
    }
    if (c.multipleOf && n % c.multipleOf !== 0) {
        return `${modelKey} needs an input size that is a multiple of ${c.multipleOf} so the decoder realigns with the encoder; ${n} is not. Try ${Math.round(n / c.multipleOf) * c.multipleOf}.`;
    }
    return null;
}
