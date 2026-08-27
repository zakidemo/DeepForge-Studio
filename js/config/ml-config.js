// Parameter panels for the classical-ML models exposed in js/config/models.js.
// Every model listed there MUST have an entry here, otherwise showMLConfigModal()
// has nothing to render. The `task` field drives which data pipeline and which
// evaluation metrics the exporter emits:
//   classification -> X/y, train/test split, accuracy + classification report
//   regression     -> X/y, train/test split, R2 / MAE / RMSE
//   clustering     -> X only, no split, silhouette + inertia
//   decomposition  -> X only, no split, explained variance
export const mlConfigurations = {
    'svm': {
        name: 'Support Vector Machine',
        task: 'classification',
        params: {
            kernel: { label: 'Kernel Type', type: 'select', options: ['rbf', 'linear', 'poly', 'sigmoid'], default: 'rbf', description: 'Kernel function' },
            C: { label: 'Regularization (C)', type: 'range', min: 0.01, max: 100, step: 0.01, default: 1.0, description: 'Regularization parameter' },
            gamma: { label: 'Gamma', type: 'select', options: ['scale', 'auto', '0.001', '0.01', '0.1', '1'], default: 'scale', description: 'Kernel coefficient' },
            probability: { label: 'Enable Probability', type: 'checkbox', default: true, description: 'Enable probability estimates' }
        }
    },
    'knn': {
        name: 'K-Nearest Neighbors',
        task: 'classification',
        params: {
            n_neighbors: { label: 'Number of Neighbors', type: 'range', min: 1, max: 50, step: 1, default: 5, description: 'Number of neighbors' },
            weights: { label: 'Weight Function', type: 'select', options: ['uniform', 'distance'], default: 'uniform', description: 'Weight function' },
            algorithm: { label: 'Algorithm', type: 'select', options: ['auto', 'ball_tree', 'kd_tree', 'brute'], default: 'auto', description: 'Algorithm to compute nearest neighbors' },
            metric: { label: 'Distance Metric', type: 'select', options: ['euclidean', 'manhattan', 'chebyshev', 'minkowski'], default: 'euclidean', description: 'Distance metric' }
        }
    },
    'randomforest': {
        name: 'Random Forest',
        task: 'classification',
        params: {
            n_estimators: { label: 'Number of Trees', type: 'range', min: 10, max: 500, step: 10, default: 100, description: 'Number of trees' },
            max_depth: { label: 'Max Depth', type: 'range', min: 1, max: 50, step: 1, default: 10, description: 'Maximum depth' },
            min_samples_split: { label: 'Min Samples Split', type: 'range', min: 2, max: 20, step: 1, default: 2, description: 'Minimum samples required to split' }
        }
    },
    'xgboost': {
        name: 'XGBoost',
        task: 'classification',
        requires: ['xgboost'],
        params: {
            n_estimators: { label: 'Number of Boosting Rounds', type: 'range', min: 10, max: 1000, step: 10, default: 100, description: 'Number of trees to fit' },
            max_depth: { label: 'Max Depth', type: 'range', min: 1, max: 20, step: 1, default: 6, description: 'Maximum tree depth' },
            learning_rate: { label: 'Learning Rate', type: 'range', min: 0.01, max: 1, step: 0.01, default: 0.3, description: 'Step size shrinkage (eta)' },
            subsample: { label: 'Subsample Ratio', type: 'range', min: 0.1, max: 1, step: 0.05, default: 1.0, description: 'Fraction of rows sampled per tree' }
        }
    },
    'decisiontree': {
        name: 'Decision Tree',
        task: 'classification',
        params: {
            criterion: { label: 'Split Criterion', type: 'select', options: ['gini', 'entropy', 'log_loss'], default: 'gini', description: 'Function measuring split quality' },
            max_depth: { label: 'Max Depth', type: 'range', min: 1, max: 50, step: 1, default: 10, description: 'Maximum depth of the tree' },
            min_samples_split: { label: 'Min Samples Split', type: 'range', min: 2, max: 20, step: 1, default: 2, description: 'Minimum samples required to split a node' },
            min_samples_leaf: { label: 'Min Samples Leaf', type: 'range', min: 1, max: 20, step: 1, default: 1, description: 'Minimum samples required at a leaf' }
        }
    },
    'naivebayes': {
        name: 'Naive Bayes',
        task: 'classification',
        params: {
            variant: { label: 'Variant', type: 'select', options: ['gaussian', 'multinomial', 'bernoulli'], default: 'gaussian', description: 'Distribution assumed for the features' },
            alpha: { label: 'Smoothing (alpha)', type: 'range', min: 0.01, max: 10, step: 0.01, default: 1.0, description: 'Additive smoothing (multinomial / bernoulli only)' }
        }
    },
    'logisticregression': {
        name: 'Logistic Regression',
        task: 'classification',
        params: {
            C: { label: 'Inverse Regularization (C)', type: 'range', min: 0.01, max: 100, step: 0.01, default: 1.0, description: 'Smaller values mean stronger regularization' },
            penalty: { label: 'Penalty', type: 'select', options: ['l2', 'l1', 'none'], default: 'l2', description: 'Regularization norm' },
            solver: { label: 'Solver', type: 'select', options: ['lbfgs', 'saga'], default: 'lbfgs', description: 'Optimization algorithm (liblinear is excluded: it cannot fit more than two classes)' },
            max_iter: { label: 'Max Iterations', type: 'range', min: 100, max: 5000, step: 100, default: 1000, description: 'Maximum solver iterations' }
        }
    },
    'linearregression': {
        name: 'Linear Regression',
        task: 'regression',
        params: {
            fit_intercept: { label: 'Fit Intercept', type: 'checkbox', default: true, description: 'Whether to estimate the intercept term' },
            positive: { label: 'Force Positive Coefficients', type: 'checkbox', default: false, description: 'Constrain all coefficients to be positive' }
        }
    },
    'kmeans': {
        name: 'K-Means Clustering',
        task: 'clustering',
        params: {
            n_clusters: { label: 'Number of Clusters (k)', type: 'range', min: 2, max: 50, step: 1, default: 3, description: 'Number of clusters to form' },
            init: { label: 'Initialization', type: 'select', options: ['k-means++', 'random'], default: 'k-means++', description: 'Centroid initialization method' },
            n_init: { label: 'Number of Initializations', type: 'range', min: 1, max: 50, step: 1, default: 10, description: 'Runs with different seeds; best inertia is kept' },
            max_iter: { label: 'Max Iterations', type: 'range', min: 50, max: 1000, step: 50, default: 300, description: 'Maximum iterations per run' }
        }
    },
    'pca': {
        name: 'Principal Component Analysis',
        task: 'decomposition',
        params: {
            n_components: { label: 'Number of Components', type: 'range', min: 1, max: 50, step: 1, default: 2, description: 'Dimensions to keep' },
            svd_solver: { label: 'SVD Solver', type: 'select', options: ['auto', 'full', 'randomized'], default: 'auto', description: 'Algorithm used for the decomposition' },
            whiten: { label: 'Whiten Components', type: 'checkbox', default: false, description: 'Scale components to unit variance' }
        }
    }
};

// Task lookup used by the exporter. Kept as a separate export so the code
// generator does not have to defensively handle a missing configuration.
export const mlTask = (modelType) => mlConfigurations[modelType]?.task || 'classification';
