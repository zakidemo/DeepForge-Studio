export const mlConfigurations = {
    'svm': {
        name: 'Support Vector Machine',
        params: {
            kernel: { label: 'Kernel Type', type: 'select', options: ['rbf', 'linear', 'poly', 'sigmoid'], default: 'rbf', description: 'Kernel function' },
            C: { label: 'Regularization (C)', type: 'range', min: 0.01, max: 100, step: 0.01, default: 1.0, description: 'Regularization parameter' },
            gamma: { label: 'Gamma', type: 'select', options: ['scale', 'auto', '0.001', '0.01', '0.1', '1'], default: 'scale', description: 'Kernel coefficient' },
            probability: { label: 'Enable Probability', type: 'checkbox', default: true, description: 'Enable probability estimates' }
        }
    },
    'knn': {
        name: 'K-Nearest Neighbors',
        params: {
            n_neighbors: { label: 'Number of Neighbors', type: 'range', min: 1, max: 50, step: 1, default: 5, description: 'Number of neighbors' },
            weights: { label: 'Weight Function', type: 'select', options: ['uniform', 'distance'], default: 'uniform', description: 'Weight function' },
            algorithm: { label: 'Algorithm', type: 'select', options: ['auto', 'ball_tree', 'kd_tree', 'brute'], default: 'auto', description: 'Algorithm to compute nearest neighbors' },
            metric: { label: 'Distance Metric', type: 'select', options: ['euclidean', 'manhattan', 'chebyshev', 'minkowski'], default: 'euclidean', description: 'Distance metric' }
        }
    },
    'randomforest': {
        name: 'Random Forest',
        params: {
            n_estimators: { label: 'Number of Trees', type: 'range', min: 10, max: 500, step: 10, default: 100, description: 'Number of trees' },
            max_depth: { label: 'Max Depth', type: 'range', min: 1, max: 50, step: 1, default: 10, description: 'Maximum depth' },
            min_samples_split: { label: 'Min Samples Split', type: 'range', min: 2, max: 20, step: 1, default: 2, description: 'Minimum samples required to split' }
        }
    }
};