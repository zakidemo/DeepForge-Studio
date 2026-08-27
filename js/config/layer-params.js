// Parameter specification for the Custom Builder's layer configuration modal.
//
// Previously five layer types had a hand-written modal each and the other five
// had none, so BatchNorm, Flatten, GlobalAvgPool, LSTM and GRU could not be
// configured at all. One spec-driven modal covers every type and gives all of
// them the skip-connection control uniformly.
export const LAYER_PARAMS = {
    Conv2D: [
        { key: 'filters', label: 'Number of Filters', type: 'number', min: 1, max: 512, default: 32 },
        { key: 'kernel_size', label: 'Kernel Size', type: 'select', options: [['3', '3x3'], ['5', '5x5'], ['7', '7x7']], default: '3' },
        { key: 'stride', label: 'Stride', type: 'number', min: 1, max: 5, default: 1 },
        { key: 'padding', label: 'Padding', type: 'select', options: [['valid', 'Valid'], ['same', 'Same']], default: 'valid' },
        { key: 'activation', label: 'Activation Function', type: 'select', options: [['relu', 'ReLU'], ['sigmoid', 'Sigmoid'], ['tanh', 'Tanh'], ['linear', 'Linear']], default: 'relu' }
    ],
    Dense: [
        { key: 'units', label: 'Number of Units', type: 'number', min: 1, max: 4096, default: 128 },
        { key: 'activation', label: 'Activation Function', type: 'select', options: [['relu', 'ReLU'], ['sigmoid', 'Sigmoid'], ['tanh', 'Tanh'], ['linear', 'Linear']], default: 'relu' }
    ],
    Dropout: [
        { key: 'rate', label: 'Dropout Rate', type: 'number', min: 0, max: 0.9, step: 0.05, default: 0.5 }
    ],
    MaxPool: [
        { key: 'pool_size', label: 'Pool Size', type: 'select', options: [['2', '2x2'], ['3', '3x3']], default: '2' },
        { key: 'stride', label: 'Stride', type: 'number', min: 1, max: 5, default: 2 },
        { key: 'padding', label: 'Padding', type: 'select', options: [['valid', 'Valid'], ['same', 'Same']], default: 'valid' }
    ],
    AvgPool: [
        { key: 'pool_size', label: 'Pool Size', type: 'select', options: [['2', '2x2'], ['3', '3x3']], default: '2' },
        { key: 'stride', label: 'Stride', type: 'number', min: 1, max: 5, default: 2 },
        { key: 'padding', label: 'Padding', type: 'select', options: [['valid', 'Valid'], ['same', 'Same']], default: 'valid' }
    ],
    BatchNorm: [
        { key: 'momentum', label: 'Momentum', type: 'number', min: 0.1, max: 0.999, step: 0.001, default: 0.99 },
        { key: 'epsilon', label: 'Epsilon', type: 'number', min: 0.000001, max: 0.1, step: 0.0001, default: 0.001 }
    ],
    LSTM: [
        { key: 'units', label: 'Number of Units', type: 'number', min: 1, max: 1024, default: 128 },
        { key: 'return_sequences', label: 'Return Sequences', type: 'checkbox', default: false, help: 'Required when another recurrent layer follows this one.' }
    ],
    GRU: [
        { key: 'units', label: 'Number of Units', type: 'number', min: 1, max: 1024, default: 64 },
        { key: 'return_sequences', label: 'Return Sequences', type: 'checkbox', default: false, help: 'Required when another recurrent layer follows this one.' }
    ],
    Flatten: [],
    GlobalAvgPool: []
};
