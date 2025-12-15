export const ModelVisualSystem = {
    visualizations: {
        'simple_cnn': {
            type: 'architecture',
            title: 'How CNN Recognizes Images',
            diagram: `<div class="visual-container">...CNN Visual HTML...</div>`,
            animation: true
        },
        'bilstm': {
            type: 'sequence',
            title: 'BiLSTM',
            diagram: `<div class="visual-container">...LSTM Visual HTML...</div>`
        },
        'gnn': {
            type: 'graph',
            title: 'Graph Neural Network',
            diagram: `<div class="visual-container">...GNN Visual HTML...</div>`
        }
    },

    showVisual(modelKey) {
        const visual = this.visualizations[modelKey];
        if (!visual) return '';
        return `
            <div class="model-visual-panel">
                <div class="visual-header">
                    <h3>${visual.title}</h3>
                    <button class="btn-icon" onclick="ModelVisualSystem.toggleAnimation('${modelKey}')">
                        <i class="fas fa-play"></i> Animate
                    </button>
                </div>
                ${visual.diagram}
            </div>
        `;
    },
    
    toggleAnimation(modelKey) {
        const container = document.querySelector('.visual-container');
        if (container) container.classList.toggle('animated');
    },

    initInteractive(modelKey) {
        if (modelKey === 'gnn') this.drawGraphNetwork();
    },

    drawGraphNetwork() {
        const canvas = document.getElementById('gnn-canvas');
        if (!canvas) return;
        const ctx = canvas.getContext('2d');
        
        // Node positions
        const nodes = [
            {x: 100, y: 200, label: 'A', color: '#50fa7b'},
            {x: 250, y: 100, label: 'B', color: '#8be9fd'},
            {x: 250, y: 300, label: 'C', color: '#ffb86c'},
            {x: 400, y: 200, label: 'D', color: '#ff79c6'},
            {x: 400, y: 100, label: 'E', color: '#bd93f9'}
        ];
        
        // Edges
        const edges = [
            [0, 1], [0, 2], [1, 3], [2, 3], [1, 4], [3, 4]
        ];
        
        // Draw edges
        ctx.strokeStyle = '#667eea';
        ctx.lineWidth = 2;
        edges.forEach(([from, to]) => {
            ctx.beginPath();
            ctx.moveTo(nodes[from].x, nodes[from].y);
            ctx.lineTo(nodes[to].x, nodes[to].y);
            ctx.stroke();
        });
        
        // Draw nodes
        nodes.forEach(node => {
            ctx.fillStyle = node.color;
            ctx.beginPath();
            ctx.arc(node.x, node.y, 25, 0, Math.PI * 2);
            ctx.fill();
            
            ctx.fillStyle = 'white';
            ctx.font = 'bold 16px Arial';
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.fillText(node.label, node.x, node.y);
        });
    }
};