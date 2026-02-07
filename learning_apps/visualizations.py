"""
D3.js Algorithm Visualizations - Interactive Learning Experience

Generates beautiful, animated visualizations for algorithms:
- Tree structures (BST, optimal BST, heaps)
- Array operations (sorting, searching, LIS)
- Graph algorithms (BFS, DFS, shortest paths)
- DP tables with step highlighting
- Neural networks with gradient flow
- Concept maps and knowledge graphs

All visualizations are interactive and can be stepped through.
"""
import json
from typing import Dict, List, Any, Optional


def get_d3_visualization_html() -> str:
    """Return the complete D3.js visualization component."""
    return '''
<!-- D3.js Algorithm Visualization Container -->
<div id="viz-container" class="viz-container" style="display:none;">
  <div class="viz-header">
    <h3 id="viz-title">Visualization</h3>
    <div class="viz-controls">
      <button id="viz-prev" class="viz-btn" disabled>⏮ Previous</button>
      <button id="viz-play" class="viz-btn">▶ Play</button>
      <button id="viz-next" class="viz-btn">⏭ Next</button>
      <button id="viz-close" class="viz-btn viz-close">✕</button>
    </div>
  </div>
  <div class="viz-step-info">
    <span id="viz-step-number">Step 1</span>
    <span id="viz-step-desc">Description</span>
  </div>
  <svg id="viz-svg" width="100%" height="400"></svg>
  <div id="viz-legend" class="viz-legend"></div>
</div>

<style>
.viz-container {
  background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
  border-radius: 16px;
  padding: 20px;
  margin: 20px 0;
  border: 1px solid #0f3460;
  box-shadow: 0 10px 40px rgba(0,0,0,0.3);
}
.viz-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 16px;
}
.viz-header h3 {
  color: #e94560;
  margin: 0;
  font-size: 1.2rem;
}
.viz-controls {
  display: flex;
  gap: 8px;
}
.viz-btn {
  background: #0f3460;
  border: 1px solid #e94560;
  color: #e94560;
  padding: 8px 16px;
  border-radius: 8px;
  cursor: pointer;
  font-size: 0.85rem;
  transition: all 0.2s;
}
.viz-btn:hover:not(:disabled) {
  background: #e94560;
  color: white;
}
.viz-btn:disabled {
  opacity: 0.4;
  cursor: not-allowed;
}
.viz-close {
  background: transparent;
  border-color: #94a3b8;
  color: #94a3b8;
}
.viz-step-info {
  background: rgba(233, 69, 96, 0.1);
  border-left: 3px solid #e94560;
  padding: 12px 16px;
  margin-bottom: 16px;
  border-radius: 0 8px 8px 0;
}
#viz-step-number {
  color: #e94560;
  font-weight: bold;
  margin-right: 12px;
}
#viz-step-desc {
  color: #e2e8f0;
}
#viz-svg {
  background: rgba(0,0,0,0.2);
  border-radius: 8px;
}
.viz-legend {
  display: flex;
  gap: 16px;
  margin-top: 12px;
  flex-wrap: wrap;
}
.viz-legend-item {
  display: flex;
  align-items: center;
  gap: 6px;
  color: #94a3b8;
  font-size: 0.85rem;
}
.viz-legend-color {
  width: 12px;
  height: 12px;
  border-radius: 50%;
}
/* Tree nodes */
.tree-node circle {
  fill: #0f3460;
  stroke: #e94560;
  stroke-width: 2;
  transition: all 0.3s;
}
.tree-node.highlight circle {
  fill: #e94560;
  stroke: #fff;
}
.tree-node text {
  fill: #e2e8f0;
  font-size: 14px;
  font-weight: bold;
}
.tree-link {
  fill: none;
  stroke: #475569;
  stroke-width: 2;
}
/* Array elements */
.array-element {
  transition: all 0.3s;
}
.array-element rect {
  fill: #0f3460;
  stroke: #475569;
  rx: 4;
}
.array-element.highlight rect {
  fill: #e94560;
  stroke: #fff;
}
.array-element.selected rect {
  fill: #22c55e;
  stroke: #fff;
}
.array-element.comparing rect {
  fill: #f59e0b;
  stroke: #fff;
}
.array-element text {
  fill: #e2e8f0;
  font-size: 14px;
  font-weight: bold;
  text-anchor: middle;
}
/* DP Table */
.dp-cell {
  transition: all 0.3s;
}
.dp-cell rect {
  fill: #0f3460;
  stroke: #334155;
}
.dp-cell.highlight rect {
  fill: #e94560;
  stroke: #fff;
  stroke-width: 2;
}
.dp-cell.optimal rect {
  fill: #22c55e;
}
.dp-cell text {
  fill: #e2e8f0;
  font-size: 12px;
  text-anchor: middle;
}
/* Neural network */
.neuron circle {
  fill: #0f3460;
  stroke: #6366f1;
  stroke-width: 2;
}
.neuron.active circle {
  fill: #6366f1;
}
.neuron.gradient circle {
  fill: #e94560;
  animation: pulse 0.5s ease-in-out;
}
.synapse {
  stroke: #475569;
  stroke-width: 1;
  opacity: 0.6;
}
.synapse.active {
  stroke: #6366f1;
  stroke-width: 2;
  opacity: 1;
}
@keyframes pulse {
  0%, 100% { transform: scale(1); }
  50% { transform: scale(1.2); }
}
</style>

<script src="https://d3js.org/d3.v7.min.js"></script>
<script>
// D3.js Visualization Engine
class VizEngine {
  constructor(containerId) {
    this.container = document.getElementById(containerId);
    this.svg = d3.select('#viz-svg');
    this.currentStep = 0;
    this.steps = [];
    this.playing = false;
    this.playInterval = null;
    this.vizType = null;
    
    this.initControls();
  }
  
  initControls() {
    document.getElementById('viz-prev').addEventListener('click', () => this.prevStep());
    document.getElementById('viz-next').addEventListener('click', () => this.nextStep());
    document.getElementById('viz-play').addEventListener('click', () => this.togglePlay());
    document.getElementById('viz-close').addEventListener('click', () => this.hide());
  }
  
  show(title) {
    this.container.style.display = 'block';
    document.getElementById('viz-title').textContent = title;
    this.svg.selectAll('*').remove();
  }
  
  hide() {
    this.container.style.display = 'none';
    this.stopPlay();
  }
  
  setSteps(steps) {
    this.steps = steps;
    this.currentStep = 0;
    this.updateStepInfo();
    this.updateControls();
  }
  
  updateStepInfo() {
    const step = this.steps[this.currentStep];
    if (step) {
      document.getElementById('viz-step-number').textContent = `Step ${step.step || this.currentStep + 1}`;
      document.getElementById('viz-step-desc').textContent = step.description || '';
    }
  }
  
  updateControls() {
    document.getElementById('viz-prev').disabled = this.currentStep === 0;
    document.getElementById('viz-next').disabled = this.currentStep >= this.steps.length - 1;
  }
  
  nextStep() {
    if (this.currentStep < this.steps.length - 1) {
      this.currentStep++;
      this.renderStep();
    } else {
      this.stopPlay();
    }
  }
  
  prevStep() {
    if (this.currentStep > 0) {
      this.currentStep--;
      this.renderStep();
    }
  }
  
  togglePlay() {
    if (this.playing) {
      this.stopPlay();
    } else {
      this.startPlay();
    }
  }
  
  startPlay() {
    this.playing = true;
    document.getElementById('viz-play').textContent = '⏸ Pause';
    this.playInterval = setInterval(() => this.nextStep(), 1500);
  }
  
  stopPlay() {
    this.playing = false;
    document.getElementById('viz-play').textContent = '▶ Play';
    if (this.playInterval) {
      clearInterval(this.playInterval);
      this.playInterval = null;
    }
  }
  
  renderStep() {
    this.updateStepInfo();
    this.updateControls();
    
    const step = this.steps[this.currentStep];
    if (!step) return;
    
    switch (this.vizType) {
      case 'tree':
        this.renderTree(step);
        break;
      case 'array':
      case 'array_sequence':
        this.renderArray(step);
        break;
      case 'dp_table':
        this.renderDPTable(step);
        break;
      case 'neural_network':
        this.renderNeuralNetwork(step);
        break;
    }
  }
  
  // === TREE VISUALIZATION ===
  initTree(data, steps) {
    this.vizType = 'tree';
    this.show('Binary Search Tree');
    this.setSteps(steps);
    this.treeData = data;
    this.renderTree(steps[0]);
  }
  
  renderTree(step) {
    const svg = this.svg;
    const width = svg.node().getBoundingClientRect().width;
    const height = 400;
    
    svg.selectAll('*').remove();
    
    // If we have a tree structure from the step
    const treeData = step.treeData || this.buildTreeFromRootTable(step);
    if (!treeData) return;
    
    const root = d3.hierarchy(treeData);
    const treeLayout = d3.tree().size([width - 80, height - 100]);
    treeLayout(root);
    
    const g = svg.append('g').attr('transform', 'translate(40, 50)');
    
    // Draw links
    g.selectAll('.tree-link')
      .data(root.links())
      .join('path')
      .attr('class', 'tree-link')
      .attr('d', d3.linkVertical()
        .x(d => d.x)
        .y(d => d.y));
    
    // Draw nodes
    const nodes = g.selectAll('.tree-node')
      .data(root.descendants())
      .join('g')
      .attr('class', d => {
        const highlight = step.highlight || [];
        const isHighlight = highlight.some(h => 
          (typeof h === 'string' && h === d.data.name) ||
          (Array.isArray(h) && this.isInRange(d, h))
        );
        return `tree-node ${isHighlight ? 'highlight' : ''}`;
      })
      .attr('transform', d => `translate(${d.x}, ${d.y})`);
    
    nodes.append('circle').attr('r', 22);
    nodes.append('text')
      .attr('dy', 5)
      .attr('text-anchor', 'middle')
      .text(d => d.data.name);
    
    // Legend
    this.setLegend([
      { color: '#e94560', label: 'Current focus' },
      { color: '#0f3460', label: 'Computed' }
    ]);
  }
  
  buildTreeFromRootTable(step) {
    // Build tree from OBST root table if available
    if (!step.root_table) return null;
    const keys = step.keys || ['A', 'B', 'C', 'D', 'E'].slice(0, step.root_table.length);
    return this.buildSubtree(keys, step.root_table, 0, keys.length - 1);
  }
  
  buildSubtree(keys, rootTable, i, j) {
    if (i > j) return null;
    const r = rootTable[i][j];
    return {
      name: keys[r],
      children: [
        this.buildSubtree(keys, rootTable, i, r - 1),
        this.buildSubtree(keys, rootTable, r + 1, j)
      ].filter(c => c !== null)
    };
  }
  
  isInRange(node, range) {
    return true; // Simplified
  }
  
  // === ARRAY VISUALIZATION ===
  initArray(data, steps) {
    this.vizType = 'array';
    this.show('Array Visualization');
    this.setSteps(steps);
    this.renderArray(steps[0]);
  }
  
  renderArray(step) {
    const svg = this.svg;
    const width = svg.node().getBoundingClientRect().width;
    const height = 400;
    
    svg.selectAll('*').remove();
    
    const arr = step.array || [];
    const dp = step.dp || [];
    const cellWidth = Math.min(60, (width - 80) / arr.length);
    const startX = (width - cellWidth * arr.length) / 2;
    
    const g = svg.append('g').attr('transform', `translate(0, 50)`);
    
    // Main array
    g.append('text')
      .attr('x', 40)
      .attr('y', 20)
      .attr('fill', '#94a3b8')
      .text('Array:');
    
    const elements = g.selectAll('.array-element')
      .data(arr)
      .join('g')
      .attr('class', (d, i) => {
        let cls = 'array-element';
        if (i === step.current_index) cls += ' highlight';
        if (step.current_lis && step.current_lis.includes(d)) cls += ' selected';
        if (step.comparing_with && step.comparing_with.some(c => c[0] === i)) cls += ' comparing';
        return cls;
      })
      .attr('transform', (d, i) => `translate(${startX + i * cellWidth}, 40)`);
    
    elements.append('rect')
      .attr('width', cellWidth - 4)
      .attr('height', 50)
      .attr('rx', 4);
    
    elements.append('text')
      .attr('x', cellWidth / 2 - 2)
      .attr('y', 32)
      .text(d => d);
    
    elements.append('text')
      .attr('x', cellWidth / 2 - 2)
      .attr('y', 65)
      .attr('fill', '#64748b')
      .attr('font-size', 10)
      .text((d, i) => `[${i}]`);
    
    // DP array if available
    if (dp.length > 0) {
      g.append('text')
        .attr('x', 40)
        .attr('y', 130)
        .attr('fill', '#94a3b8')
        .text('DP:');
      
      const dpElements = g.selectAll('.dp-element')
        .data(dp)
        .join('g')
        .attr('class', 'array-element')
        .attr('transform', (d, i) => `translate(${startX + i * cellWidth}, 150)`);
      
      dpElements.append('rect')
        .attr('width', cellWidth - 4)
        .attr('height', 40)
        .attr('fill', '#1e3a5f')
        .attr('stroke', '#6366f1')
        .attr('rx', 4);
      
      dpElements.append('text')
        .attr('x', cellWidth / 2 - 2)
        .attr('y', 27)
        .text(d => d);
    }
    
    // Current LIS
    if (step.current_lis && step.current_lis.length > 0) {
      g.append('text')
        .attr('x', 40)
        .attr('y', 240)
        .attr('fill', '#94a3b8')
        .text('Current LIS:');
      
      g.append('text')
        .attr('x', 140)
        .attr('y', 240)
        .attr('fill', '#22c55e')
        .attr('font-weight', 'bold')
        .text(`[${step.current_lis.join(', ')}] (length ${step.current_lis.length})`);
    }
    
    this.setLegend([
      { color: '#e94560', label: 'Current element' },
      { color: '#22c55e', label: 'In LIS' },
      { color: '#f59e0b', label: 'Comparing' }
    ]);
  }
  
  // === DP TABLE VISUALIZATION ===
  initDPTable(data, steps) {
    this.vizType = 'dp_table';
    this.show('Dynamic Programming Table');
    this.setSteps(steps);
    this.renderDPTable(steps[0]);
  }
  
  renderDPTable(step) {
    const svg = this.svg;
    const width = svg.node().getBoundingClientRect().width;
    const height = 400;
    
    svg.selectAll('*').remove();
    
    const costTable = step.cost_table || step.dp || [];
    if (!costTable.length) return;
    
    const rows = costTable.length;
    const cols = costTable[0]?.length || costTable.length;
    const cellSize = Math.min(50, (width - 100) / cols, (height - 100) / rows);
    const startX = (width - cellSize * cols) / 2;
    const startY = 60;
    
    const g = svg.append('g');
    
    g.append('text')
      .attr('x', width / 2)
      .attr('y', 30)
      .attr('text-anchor', 'middle')
      .attr('fill', '#e2e8f0')
      .attr('font-size', 14)
      .text(step.description || 'DP Table');
    
    const highlights = step.highlight || [];
    
    for (let i = 0; i < rows; i++) {
      for (let j = 0; j < cols; j++) {
        const value = Array.isArray(costTable[i]) ? costTable[i][j] : costTable[i];
        const isHighlight = highlights.some(h => h[0] === i && h[1] === j);
        
        const cell = g.append('g')
          .attr('class', `dp-cell ${isHighlight ? 'highlight' : ''}`)
          .attr('transform', `translate(${startX + j * cellSize}, ${startY + i * cellSize})`);
        
        cell.append('rect')
          .attr('width', cellSize - 2)
          .attr('height', cellSize - 2)
          .attr('rx', 4);
        
        cell.append('text')
          .attr('x', cellSize / 2 - 1)
          .attr('y', cellSize / 2 + 4)
          .text(value === Infinity || value === 'Infinity' ? '∞' : 
                (typeof value === 'number' ? value.toFixed(1) : value));
      }
    }
    
    // Row and column labels
    const keys = step.keys || ['A', 'B', 'C', 'D', 'E'];
    for (let i = 0; i < Math.min(rows, keys.length); i++) {
      g.append('text')
        .attr('x', startX - 20)
        .attr('y', startY + i * cellSize + cellSize / 2 + 4)
        .attr('fill', '#94a3b8')
        .attr('text-anchor', 'middle')
        .attr('font-size', 12)
        .text(keys[i]);
      
      g.append('text')
        .attr('x', startX + i * cellSize + cellSize / 2)
        .attr('y', startY - 10)
        .attr('fill', '#94a3b8')
        .attr('text-anchor', 'middle')
        .attr('font-size', 12)
        .text(keys[i]);
    }
    
    this.setLegend([
      { color: '#e94560', label: 'Current computation' },
      { color: '#22c55e', label: 'Optimal path' }
    ]);
  }
  
  // === NEURAL NETWORK VISUALIZATION ===
  initNeuralNetwork(layers, steps) {
    this.vizType = 'neural_network';
    this.show('Neural Network - Backpropagation');
    this.layers = layers;
    this.setSteps(steps);
    this.renderNeuralNetwork(steps[0]);
  }
  
  renderNeuralNetwork(step) {
    const svg = this.svg;
    const width = svg.node().getBoundingClientRect().width;
    const height = 400;
    
    svg.selectAll('*').remove();
    
    const layers = this.layers || [3, 4, 4, 2];
    const maxNeurons = Math.max(...layers);
    const layerSpacing = (width - 100) / (layers.length - 1);
    const neuronSpacing = (height - 100) / (maxNeurons + 1);
    const startX = 50;
    const startY = 50;
    
    const g = svg.append('g');
    
    // Draw synapses (connections)
    for (let l = 0; l < layers.length - 1; l++) {
      for (let i = 0; i < layers[l]; i++) {
        for (let j = 0; j < layers[l + 1]; j++) {
          const x1 = startX + l * layerSpacing;
          const y1 = startY + (height - 100 - layers[l] * neuronSpacing) / 2 + i * neuronSpacing;
          const x2 = startX + (l + 1) * layerSpacing;
          const y2 = startY + (height - 100 - layers[l + 1] * neuronSpacing) / 2 + j * neuronSpacing;
          
          g.append('line')
            .attr('class', `synapse ${step.active_layer === l ? 'active' : ''}`)
            .attr('x1', x1)
            .attr('y1', y1)
            .attr('x2', x2)
            .attr('y2', y2);
        }
      }
    }
    
    // Draw neurons
    for (let l = 0; l < layers.length; l++) {
      const layerY = startY + (height - 100 - layers[l] * neuronSpacing) / 2;
      
      for (let i = 0; i < layers[l]; i++) {
        const isActive = step.active_layer === l;
        const isGradient = step.gradient_layer === l;
        
        const neuron = g.append('g')
          .attr('class', `neuron ${isActive ? 'active' : ''} ${isGradient ? 'gradient' : ''}`)
          .attr('transform', `translate(${startX + l * layerSpacing}, ${layerY + i * neuronSpacing})`);
        
        neuron.append('circle')
          .attr('r', 15);
        
        // Show activation value if available
        if (step.activations && step.activations[l]) {
          neuron.append('text')
            .attr('dy', 4)
            .attr('text-anchor', 'middle')
            .attr('fill', '#fff')
            .attr('font-size', 10)
            .text(step.activations[l][i]?.toFixed(1) || '');
        }
      }
      
      // Layer label
      g.append('text')
        .attr('x', startX + l * layerSpacing)
        .attr('y', height - 30)
        .attr('text-anchor', 'middle')
        .attr('fill', '#94a3b8')
        .attr('font-size', 12)
        .text(l === 0 ? 'Input' : l === layers.length - 1 ? 'Output' : `Hidden ${l}`);
    }
    
    // Direction arrow
    const arrowY = height - 60;
    if (step.direction === 'forward') {
      g.append('text')
        .attr('x', width / 2)
        .attr('y', arrowY)
        .attr('text-anchor', 'middle')
        .attr('fill', '#6366f1')
        .attr('font-size', 14)
        .text('→ Forward Pass');
    } else if (step.direction === 'backward') {
      g.append('text')
        .attr('x', width / 2)
        .attr('y', arrowY)
        .attr('text-anchor', 'middle')
        .attr('fill', '#e94560')
        .attr('font-size', 14)
        .text('← Backward Pass (Gradients)');
    }
    
    this.setLegend([
      { color: '#6366f1', label: 'Forward activation' },
      { color: '#e94560', label: 'Gradient flow' }
    ]);
  }
  
  // === HELPER METHODS ===
  setLegend(items) {
    const legend = document.getElementById('viz-legend');
    legend.innerHTML = items.map(item => `
      <div class="viz-legend-item">
        <div class="viz-legend-color" style="background: ${item.color}"></div>
        <span>${item.label}</span>
      </div>
    `).join('');
  }
}

// Global instance
window.vizEngine = new VizEngine('viz-container');
</script>
'''


def get_concept_map_html() -> str:
    """Return the concept map / knowledge graph visualization."""
    return '''
<!-- Concept Map / Knowledge Graph -->
<div id="concept-map-container" class="concept-map-container" style="display:none;">
  <div class="concept-map-header">
    <h3>🧠 Concept Map</h3>
    <div class="concept-map-controls">
      <input type="text" id="concept-search" placeholder="Search concepts..." class="concept-search">
      <button id="concept-map-close" class="viz-btn viz-close">✕</button>
    </div>
  </div>
  <svg id="concept-map-svg" width="100%" height="500"></svg>
  <div id="concept-details" class="concept-details"></div>
</div>

<style>
.concept-map-container {
  background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
  border-radius: 16px;
  padding: 20px;
  margin: 20px 0;
  border: 1px solid #0f3460;
}
.concept-map-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 16px;
}
.concept-map-header h3 {
  color: #6366f1;
  margin: 0;
}
.concept-search {
  padding: 8px 16px;
  border-radius: 8px;
  border: 1px solid #475569;
  background: #1e293b;
  color: #e2e8f0;
  width: 200px;
}
#concept-map-svg {
  background: rgba(0,0,0,0.2);
  border-radius: 8px;
}
.concept-node {
  cursor: pointer;
}
.concept-node circle {
  fill: #0f3460;
  stroke: #6366f1;
  stroke-width: 2;
  transition: all 0.3s;
}
.concept-node:hover circle {
  fill: #6366f1;
  stroke: #fff;
}
.concept-node.active circle {
  fill: #e94560;
  stroke: #fff;
}
.concept-node.prerequisite circle {
  fill: #22c55e;
}
.concept-node.related circle {
  fill: #f59e0b;
}
.concept-node text {
  fill: #e2e8f0;
  font-size: 11px;
  text-anchor: middle;
}
.concept-link {
  stroke: #475569;
  stroke-width: 1.5;
  fill: none;
}
.concept-link.prerequisite {
  stroke: #22c55e;
  stroke-dasharray: 5,5;
}
.concept-link.related {
  stroke: #f59e0b;
}
.concept-details {
  margin-top: 16px;
  padding: 16px;
  background: rgba(99, 102, 241, 0.1);
  border-radius: 8px;
  display: none;
}
.concept-details h4 {
  color: #6366f1;
  margin: 0 0 8px 0;
}
.concept-details p {
  color: #94a3b8;
  margin: 4px 0;
  font-size: 0.9rem;
}
.concept-tag {
  display: inline-block;
  padding: 4px 8px;
  background: #1e293b;
  border-radius: 4px;
  font-size: 0.75rem;
  margin: 2px;
  color: #e2e8f0;
}
.concept-tag.difficulty-basics { border-left: 3px solid #22c55e; }
.concept-tag.difficulty-intermediate { border-left: 3px solid #f59e0b; }
.concept-tag.difficulty-advanced { border-left: 3px solid #e94560; }
</style>

<script>
class ConceptMap {
  constructor(containerId) {
    this.container = document.getElementById(containerId);
    this.svg = d3.select('#concept-map-svg');
    this.concepts = [];
    this.links = [];
    this.simulation = null;
    
    document.getElementById('concept-map-close').addEventListener('click', () => this.hide());
    document.getElementById('concept-search').addEventListener('input', (e) => this.search(e.target.value));
  }
  
  show() {
    this.container.style.display = 'block';
  }
  
  hide() {
    this.container.style.display = 'none';
  }
  
  setData(concepts, links) {
    this.concepts = concepts;
    this.links = links;
    this.render();
  }
  
  render() {
    const svg = this.svg;
    const width = svg.node().getBoundingClientRect().width;
    const height = 500;
    
    svg.selectAll('*').remove();
    
    // Create force simulation
    this.simulation = d3.forceSimulation(this.concepts)
      .force('link', d3.forceLink(this.links).id(d => d.id).distance(100))
      .force('charge', d3.forceManyBody().strength(-300))
      .force('center', d3.forceCenter(width / 2, height / 2))
      .force('collision', d3.forceCollide().radius(40));
    
    // Draw links
    const link = svg.append('g')
      .selectAll('line')
      .data(this.links)
      .join('line')
      .attr('class', d => `concept-link ${d.type || ''}`);
    
    // Draw nodes
    const node = svg.append('g')
      .selectAll('.concept-node')
      .data(this.concepts)
      .join('g')
      .attr('class', 'concept-node')
      .call(d3.drag()
        .on('start', (event, d) => {
          if (!event.active) this.simulation.alphaTarget(0.3).restart();
          d.fx = d.x;
          d.fy = d.y;
        })
        .on('drag', (event, d) => {
          d.fx = event.x;
          d.fy = event.y;
        })
        .on('end', (event, d) => {
          if (!event.active) this.simulation.alphaTarget(0);
          d.fx = null;
          d.fy = null;
        }))
      .on('click', (event, d) => this.showDetails(d));
    
    node.append('circle')
      .attr('r', d => 20 + (d.importance || 0) * 5);
    
    node.append('text')
      .attr('dy', 35)
      .text(d => d.name);
    
    // Update positions on tick
    this.simulation.on('tick', () => {
      link
        .attr('x1', d => d.source.x)
        .attr('y1', d => d.source.y)
        .attr('x2', d => d.target.x)
        .attr('y2', d => d.target.y);
      
      node.attr('transform', d => `translate(${d.x}, ${d.y})`);
    });
  }
  
  showDetails(concept) {
    const details = document.getElementById('concept-details');
    details.style.display = 'block';
    
    const difficultyClass = `difficulty-${concept.difficulty || 'intermediate'}`;
    
    details.innerHTML = `
      <h4>${concept.name}</h4>
      <p>${concept.description || 'No description available.'}</p>
      <div style="margin-top: 8px;">
        <span class="concept-tag ${difficultyClass}">${concept.difficulty || 'intermediate'}</span>
        <span class="concept-tag">${concept.category || 'General'}</span>
        ${(concept.books || []).map(b => `<span class="concept-tag">${b}</span>`).join('')}
      </div>
      ${concept.prerequisites ? `
        <p style="margin-top: 12px;"><strong>Prerequisites:</strong> ${concept.prerequisites.join(', ')}</p>
      ` : ''}
    `;
    
    // Highlight related nodes
    this.svg.selectAll('.concept-node')
      .classed('active', d => d.id === concept.id)
      .classed('prerequisite', d => (concept.prerequisites || []).includes(d.id))
      .classed('related', d => (concept.related || []).includes(d.id));
  }
  
  search(query) {
    if (!query) {
      this.svg.selectAll('.concept-node').style('opacity', 1);
      return;
    }
    
    const q = query.toLowerCase();
    this.svg.selectAll('.concept-node')
      .style('opacity', d => d.name.toLowerCase().includes(q) ? 1 : 0.2);
  }
}

window.conceptMap = new ConceptMap('concept-map-container');
</script>
'''


def get_full_visualization_bundle() -> str:
    """Return all visualization components bundled together."""
    return get_d3_visualization_html() + get_concept_map_html()


# =============================================================================
# CONCEPT MAP DATA GENERATOR
# =============================================================================

def generate_concept_map_data(lab_id: str = None) -> Dict:
    """Generate concept map data for visualization."""
    
    # Core concepts that apply across labs
    all_concepts = [
        # Fundamentals
        {"id": "algorithms", "name": "Algorithms", "category": "Fundamentals", "difficulty": "basics", 
         "importance": 3, "description": "Step-by-step procedures for solving problems."},
        {"id": "data_structures", "name": "Data Structures", "category": "Fundamentals", "difficulty": "basics",
         "importance": 3, "description": "Ways to organize and store data efficiently."},
        {"id": "complexity", "name": "Complexity Analysis", "category": "Fundamentals", "difficulty": "intermediate",
         "importance": 2, "description": "Analyzing time and space requirements of algorithms."},
        
        # Dynamic Programming
        {"id": "dynamic_programming", "name": "Dynamic Programming", "category": "Algorithms", "difficulty": "intermediate",
         "importance": 3, "prerequisites": ["recursion", "memoization"],
         "description": "Solving problems by breaking them into overlapping subproblems."},
        {"id": "memoization", "name": "Memoization", "category": "Algorithms", "difficulty": "intermediate",
         "importance": 2, "prerequisites": ["recursion"],
         "description": "Caching results of expensive function calls."},
        {"id": "recursion", "name": "Recursion", "category": "Fundamentals", "difficulty": "basics",
         "importance": 2, "description": "Functions that call themselves to solve smaller instances."},
        
        # Trees and Graphs
        {"id": "binary_search_tree", "name": "Binary Search Tree", "category": "Data Structures", "difficulty": "intermediate",
         "importance": 2, "prerequisites": ["trees", "binary_search"],
         "description": "Tree with left < root < right property for efficient search."},
        {"id": "trees", "name": "Trees", "category": "Data Structures", "difficulty": "basics",
         "importance": 2, "description": "Hierarchical data structure with root and children."},
        {"id": "graphs", "name": "Graphs", "category": "Data Structures", "difficulty": "intermediate",
         "importance": 3, "description": "Nodes connected by edges, modeling relationships."},
        {"id": "binary_search", "name": "Binary Search", "category": "Algorithms", "difficulty": "basics",
         "importance": 2, "description": "Efficiently find elements in sorted data."},
        
        # Machine Learning
        {"id": "neural_networks", "name": "Neural Networks", "category": "Deep Learning", "difficulty": "intermediate",
         "importance": 3, "prerequisites": ["linear_algebra", "calculus"],
         "description": "Computational models inspired by biological neurons."},
        {"id": "backpropagation", "name": "Backpropagation", "category": "Deep Learning", "difficulty": "intermediate",
         "importance": 3, "prerequisites": ["chain_rule", "neural_networks"],
         "description": "Algorithm for computing gradients in neural networks."},
        {"id": "gradient_descent", "name": "Gradient Descent", "category": "Optimization", "difficulty": "intermediate",
         "importance": 3, "prerequisites": ["calculus"],
         "description": "Optimization by following the negative gradient."},
        
        # Math foundations
        {"id": "linear_algebra", "name": "Linear Algebra", "category": "Math", "difficulty": "intermediate",
         "importance": 2, "description": "Study of vectors, matrices, and linear transformations."},
        {"id": "calculus", "name": "Calculus", "category": "Math", "difficulty": "intermediate",
         "importance": 2, "description": "Study of continuous change and rates."},
        {"id": "chain_rule", "name": "Chain Rule", "category": "Math", "difficulty": "intermediate",
         "importance": 2, "prerequisites": ["calculus"],
         "description": "Derivative of composed functions."},
        {"id": "probability", "name": "Probability", "category": "Math", "difficulty": "intermediate",
         "importance": 2, "description": "Study of random events and uncertainty."},
        
        # Information Theory
        {"id": "entropy", "name": "Entropy", "category": "Information Theory", "difficulty": "intermediate",
         "importance": 2, "prerequisites": ["probability"],
         "description": "Measure of uncertainty or information content."},
        {"id": "information_gain", "name": "Information Gain", "category": "Information Theory", "difficulty": "intermediate",
         "importance": 2, "prerequisites": ["entropy"],
         "description": "Reduction in entropy from learning new information."},
        
        # RL
        {"id": "reinforcement_learning", "name": "Reinforcement Learning", "category": "ML", "difficulty": "advanced",
         "importance": 2, "prerequisites": ["mdp", "probability"],
         "description": "Learning through interaction and rewards."},
        {"id": "mdp", "name": "Markov Decision Process", "category": "ML", "difficulty": "advanced",
         "importance": 2, "prerequisites": ["probability"],
         "description": "Mathematical framework for sequential decisions."},
    ]
    
    # Generate links based on prerequisites and related concepts
    links = []
    concept_ids = {c["id"] for c in all_concepts}
    
    for concept in all_concepts:
        for prereq in concept.get("prerequisites", []):
            if prereq in concept_ids:
                links.append({
                    "source": prereq,
                    "target": concept["id"],
                    "type": "prerequisite"
                })
        
        for related in concept.get("related", []):
            if related in concept_ids:
                links.append({
                    "source": concept["id"],
                    "target": related,
                    "type": "related"
                })
    
    return {
        "concepts": all_concepts,
        "links": links
    }


# Flask route registration for visualizations
def register_visualization_routes(app):
    """Register visualization API routes."""
    from flask import jsonify
    
    @app.route("/api/concept-map")
    def api_concept_map():
        """Get concept map data for D3 visualization."""
        lab_id = app.config.get("lab_id")
        data = generate_concept_map_data(lab_id)
        return jsonify({"ok": True, **data})
    
    @app.route("/api/concept-map/<concept_id>")
    def api_concept_detail(concept_id):
        """Get detailed info about a specific concept."""
        data = generate_concept_map_data()
        concept = next((c for c in data["concepts"] if c["id"] == concept_id), None)
        if not concept:
            return jsonify({"ok": False, "error": "Concept not found"}), 404
        return jsonify({"ok": True, "concept": concept})
