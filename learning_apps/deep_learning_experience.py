"""
Deep Learning Experience - Interactive Educational Module

A complete, immersive learning experience for deep learning concepts:
- Interactive neural network builder
- Live backpropagation visualization
- Gradient flow animations
- Math breakdowns with LaTeX
- Real-time training visualization
- Concept interconnection maps
"""
import json
import math
from typing import Dict, List, Any, Optional

def get_deep_learning_experience_html() -> str:
    """Return the complete deep learning interactive experience."""
    return '''
<!-- Deep Learning Interactive Experience -->
<div id="dl-experience" class="dl-experience" style="display:none;">
  <div class="dl-header">
    <h2>🧠 Deep Learning Lab</h2>
    <p class="dl-subtitle">Interactive Neural Network Exploration</p>
    <button id="dl-close" class="viz-btn viz-close">✕</button>
  </div>
  
  <div class="dl-tabs">
    <button class="dl-tab active" data-tab="build">🔨 Build</button>
    <button class="dl-tab" data-tab="train">🎯 Train</button>
    <button class="dl-tab" data-tab="backprop">⚡ Backprop</button>
    <button class="dl-tab" data-tab="math">📐 Math</button>
  </div>
  
  <!-- BUILD TAB: Neural Network Builder -->
  <div id="dl-tab-build" class="dl-panel active">
    <div class="dl-builder-controls">
      <div class="dl-control-group">
        <label>Input Size</label>
        <input type="number" id="nn-input-size" value="3" min="1" max="10">
      </div>
      <div class="dl-control-group">
        <label>Hidden Layers</label>
        <input type="text" id="nn-hidden" value="4,4" placeholder="e.g., 4,4,3">
      </div>
      <div class="dl-control-group">
        <label>Output Size</label>
        <input type="number" id="nn-output-size" value="2" min="1" max="10">
      </div>
      <div class="dl-control-group">
        <label>Activation</label>
        <select id="nn-activation">
          <option value="relu">ReLU</option>
          <option value="sigmoid">Sigmoid</option>
          <option value="tanh">Tanh</option>
          <option value="softmax">Softmax (output)</option>
        </select>
      </div>
      <button id="nn-build-btn" class="dl-btn primary">Build Network</button>
    </div>
    <svg id="nn-builder-svg" width="100%" height="350"></svg>
    <div id="nn-stats" class="dl-stats"></div>
  </div>
  
  <!-- TRAIN TAB: Training Visualization -->
  <div id="dl-tab-train" class="dl-panel">
    <div class="dl-train-controls">
      <div class="dl-control-group">
        <label>Learning Rate</label>
        <input type="number" id="train-lr" value="0.01" step="0.001" min="0.001" max="1">
      </div>
      <div class="dl-control-group">
        <label>Epochs</label>
        <input type="number" id="train-epochs" value="100" min="10" max="1000">
      </div>
      <div class="dl-control-group">
        <label>Dataset</label>
        <select id="train-dataset">
          <option value="xor">XOR Problem</option>
          <option value="circles">Circles</option>
          <option value="spiral">Spiral</option>
        </select>
      </div>
      <button id="train-start-btn" class="dl-btn primary">▶ Start Training</button>
      <button id="train-step-btn" class="dl-btn">Step</button>
    </div>
    <div class="dl-train-viz">
      <div class="dl-loss-chart">
        <h4>Loss Curve</h4>
        <svg id="loss-svg" width="100%" height="200"></svg>
      </div>
      <div class="dl-decision-boundary">
        <h4>Decision Boundary</h4>
        <canvas id="decision-canvas" width="300" height="300"></canvas>
      </div>
    </div>
    <div id="train-log" class="dl-log"></div>
  </div>
  
  <!-- BACKPROP TAB: Step-by-Step Backpropagation -->
  <div id="dl-tab-backprop" class="dl-panel">
    <div class="dl-backprop-intro">
      <h3>⚡ Backpropagation Step-by-Step</h3>
      <p>Watch gradients flow backwards through the network. Each step shows the chain rule in action.</p>
    </div>
    <div class="dl-backprop-controls">
      <button id="bp-forward-btn" class="dl-btn">1️⃣ Forward Pass</button>
      <button id="bp-loss-btn" class="dl-btn">2️⃣ Compute Loss</button>
      <button id="bp-backward-btn" class="dl-btn">3️⃣ Backward Pass</button>
      <button id="bp-update-btn" class="dl-btn">4️⃣ Update Weights</button>
      <button id="bp-reset-btn" class="dl-btn secondary">🔄 Reset</button>
    </div>
    <svg id="bp-svg" width="100%" height="300"></svg>
    <div id="bp-explanation" class="dl-explanation"></div>
    <div id="bp-math" class="dl-math-display"></div>
  </div>
  
  <!-- MATH TAB: Mathematical Foundations -->
  <div id="dl-tab-math" class="dl-panel">
    <div class="dl-math-sections">
      <div class="dl-math-section">
        <h3>📐 The Forward Pass</h3>
        <div class="dl-math-content">
          <p>For each layer $l$:</p>
          <div class="dl-math-block">$$z^{[l]} = W^{[l]} a^{[l-1]} + b^{[l]}$$</div>
          <div class="dl-math-block">$$a^{[l]} = g(z^{[l]})$$</div>
          <p>where $g$ is the activation function (ReLU, sigmoid, etc.)</p>
        </div>
      </div>
      
      <div class="dl-math-section">
        <h3>⚡ The Chain Rule</h3>
        <div class="dl-math-content">
          <p>To compute $\\frac{\\partial L}{\\partial W^{[l]}}$, we use:</p>
          <div class="dl-math-block">$$\\frac{\\partial L}{\\partial W^{[l]}} = \\frac{\\partial L}{\\partial a^{[L]}} \\cdot \\frac{\\partial a^{[L]}}{\\partial a^{[L-1]}} \\cdots \\frac{\\partial a^{[l]}}{\\partial W^{[l]}}$$</div>
          <p>Each term is a <strong>local gradient</strong> that we compute and multiply.</p>
        </div>
      </div>
      
      <div class="dl-math-section">
        <h3>🔄 Gradient Descent Update</h3>
        <div class="dl-math-content">
          <div class="dl-math-block">$$W^{[l]} := W^{[l]} - \\alpha \\frac{\\partial L}{\\partial W^{[l]}}$$</div>
          <div class="dl-math-block">$$b^{[l]} := b^{[l]} - \\alpha \\frac{\\partial L}{\\partial b^{[l]}}$$</div>
          <p>where $\\alpha$ is the learning rate.</p>
        </div>
      </div>
      
      <div class="dl-math-section">
        <h3>📊 Common Activation Functions</h3>
        <div class="dl-activations">
          <div class="dl-activation-card">
            <h4>ReLU</h4>
            <div class="dl-math-block">$$g(z) = \\max(0, z)$$</div>
            <div class="dl-math-block">$$g'(z) = \\begin{cases} 1 & z > 0 \\\\ 0 & z \\leq 0 \\end{cases}$$</div>
          </div>
          <div class="dl-activation-card">
            <h4>Sigmoid</h4>
            <div class="dl-math-block">$$\\sigma(z) = \\frac{1}{1 + e^{-z}}$$</div>
            <div class="dl-math-block">$$\\sigma'(z) = \\sigma(z)(1 - \\sigma(z))$$</div>
          </div>
          <div class="dl-activation-card">
            <h4>Tanh</h4>
            <div class="dl-math-block">$$\\tanh(z) = \\frac{e^z - e^{-z}}{e^z + e^{-z}}$$</div>
            <div class="dl-math-block">$$\\tanh'(z) = 1 - \\tanh^2(z)$$</div>
          </div>
        </div>
      </div>
    </div>
  </div>
</div>

<style>
.dl-experience {
  background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%);
  border-radius: 20px;
  padding: 24px;
  margin: 20px 0;
  border: 1px solid #6366f1;
  box-shadow: 0 20px 60px rgba(99, 102, 241, 0.2);
}
.dl-header {
  text-align: center;
  margin-bottom: 24px;
  position: relative;
}
.dl-header h2 {
  color: #fff;
  font-size: 1.8rem;
  margin: 0;
  background: linear-gradient(90deg, #6366f1, #e94560);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
}
.dl-subtitle {
  color: #94a3b8;
  margin: 8px 0 0 0;
}
.dl-header .viz-close {
  position: absolute;
  right: 0;
  top: 0;
}
.dl-tabs {
  display: flex;
  justify-content: center;
  gap: 8px;
  margin-bottom: 24px;
  flex-wrap: wrap;
}
.dl-tab {
  background: #1e293b;
  border: 1px solid #334155;
  color: #94a3b8;
  padding: 12px 24px;
  border-radius: 10px;
  cursor: pointer;
  font-size: 0.95rem;
  transition: all 0.3s;
}
.dl-tab:hover { background: #334155; color: #e2e8f0; }
.dl-tab.active {
  background: linear-gradient(135deg, #6366f1, #818cf8);
  border-color: #6366f1;
  color: white;
}
.dl-panel { display: none; }
.dl-panel.active { display: block; }

/* Builder Controls */
.dl-builder-controls, .dl-train-controls, .dl-backprop-controls {
  display: flex;
  flex-wrap: wrap;
  gap: 16px;
  align-items: flex-end;
  margin-bottom: 20px;
  padding: 16px;
  background: rgba(30, 41, 59, 0.5);
  border-radius: 12px;
}
.dl-control-group {
  display: flex;
  flex-direction: column;
  gap: 4px;
}
.dl-control-group label {
  color: #94a3b8;
  font-size: 0.85rem;
}
.dl-control-group input, .dl-control-group select {
  padding: 8px 12px;
  border-radius: 8px;
  border: 1px solid #475569;
  background: #1e293b;
  color: #e2e8f0;
  font-size: 0.9rem;
}
.dl-btn {
  padding: 10px 20px;
  border-radius: 8px;
  border: 1px solid #6366f1;
  background: #1e293b;
  color: #6366f1;
  cursor: pointer;
  font-size: 0.9rem;
  transition: all 0.2s;
}
.dl-btn:hover { background: #6366f1; color: white; }
.dl-btn.primary {
  background: linear-gradient(135deg, #6366f1, #818cf8);
  color: white;
  border: none;
}
.dl-btn.secondary {
  border-color: #94a3b8;
  color: #94a3b8;
}

/* Stats */
.dl-stats {
  display: flex;
  gap: 24px;
  justify-content: center;
  padding: 16px;
  background: rgba(99, 102, 241, 0.1);
  border-radius: 12px;
  margin-top: 16px;
}
.dl-stat {
  text-align: center;
}
.dl-stat-value {
  font-size: 1.5rem;
  font-weight: bold;
  color: #6366f1;
}
.dl-stat-label {
  color: #94a3b8;
  font-size: 0.8rem;
}

/* Training Viz */
.dl-train-viz {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 20px;
  margin: 20px 0;
}
.dl-loss-chart, .dl-decision-boundary {
  background: rgba(15, 23, 42, 0.6);
  padding: 16px;
  border-radius: 12px;
}
.dl-loss-chart h4, .dl-decision-boundary h4 {
  color: #e2e8f0;
  margin: 0 0 12px 0;
  font-size: 0.95rem;
}
#decision-canvas {
  background: #0f172a;
  border-radius: 8px;
  width: 100%;
  height: auto;
}
.dl-log {
  background: #0f172a;
  padding: 12px;
  border-radius: 8px;
  font-family: monospace;
  font-size: 0.85rem;
  color: #22c55e;
  max-height: 150px;
  overflow-y: auto;
}

/* Backprop */
.dl-backprop-intro {
  text-align: center;
  margin-bottom: 20px;
}
.dl-backprop-intro h3 {
  color: #e94560;
  margin: 0;
}
.dl-backprop-intro p {
  color: #94a3b8;
  margin: 8px 0 0 0;
}
.dl-explanation {
  background: rgba(233, 69, 96, 0.1);
  border-left: 3px solid #e94560;
  padding: 16px;
  margin: 16px 0;
  border-radius: 0 12px 12px 0;
  color: #e2e8f0;
}
.dl-math-display {
  background: #0f172a;
  padding: 20px;
  border-radius: 12px;
  text-align: center;
  font-size: 1.1rem;
  color: #e2e8f0;
}

/* Math Tab */
.dl-math-sections {
  display: flex;
  flex-direction: column;
  gap: 24px;
}
.dl-math-section {
  background: rgba(30, 41, 59, 0.5);
  padding: 20px;
  border-radius: 12px;
  border-left: 3px solid #6366f1;
}
.dl-math-section h3 {
  color: #6366f1;
  margin: 0 0 12px 0;
}
.dl-math-content {
  color: #e2e8f0;
}
.dl-math-content p {
  margin: 8px 0;
}
.dl-math-block {
  background: #0f172a;
  padding: 16px;
  border-radius: 8px;
  margin: 12px 0;
  text-align: center;
  font-size: 1.1rem;
  overflow-x: auto;
}
.dl-activations {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 16px;
  margin-top: 16px;
}
.dl-activation-card {
  background: #0f172a;
  padding: 16px;
  border-radius: 12px;
  text-align: center;
}
.dl-activation-card h4 {
  color: #e94560;
  margin: 0 0 12px 0;
}

/* SVG styling */
#nn-builder-svg, #bp-svg {
  background: rgba(0, 0, 0, 0.3);
  border-radius: 12px;
}
</style>

<!-- Load KaTeX for math rendering -->
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css">
<script src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/contrib/auto-render.min.js"></script>

<script>
// Deep Learning Interactive Experience
class DeepLearningExperience {
  constructor() {
    this.container = document.getElementById('dl-experience');
    this.network = null;
    this.training = false;
    this.epoch = 0;
    this.losses = [];
    
    this.initTabs();
    this.initBuilder();
    this.initTraining();
    this.initBackprop();
  }
  
  show() {
    this.container.style.display = 'block';
    this.renderMath();
  }
  
  hide() {
    this.container.style.display = 'none';
    this.training = false;
  }
  
  renderMath() {
    if (typeof renderMathInElement === 'function') {
      renderMathInElement(this.container, {
        delimiters: [
          {left: '$$', right: '$$', display: true},
          {left: '$', right: '$', display: false}
        ]
      });
    }
  }
  
  initTabs() {
    document.querySelectorAll('.dl-tab').forEach(tab => {
      tab.addEventListener('click', () => {
        document.querySelectorAll('.dl-tab').forEach(t => t.classList.remove('active'));
        document.querySelectorAll('.dl-panel').forEach(p => p.classList.remove('active'));
        tab.classList.add('active');
        document.getElementById(`dl-tab-${tab.dataset.tab}`).classList.add('active');
        if (tab.dataset.tab === 'math') this.renderMath();
      });
    });
    
    document.getElementById('dl-close').addEventListener('click', () => this.hide());
  }
  
  // === NETWORK BUILDER ===
  initBuilder() {
    document.getElementById('nn-build-btn').addEventListener('click', () => this.buildNetwork());
    this.buildNetwork(); // Initial build
  }
  
  buildNetwork() {
    const inputSize = parseInt(document.getElementById('nn-input-size').value);
    const hiddenStr = document.getElementById('nn-hidden').value;
    const outputSize = parseInt(document.getElementById('nn-output-size').value);
    const activation = document.getElementById('nn-activation').value;
    
    const hidden = hiddenStr.split(',').map(s => parseInt(s.trim())).filter(n => !isNaN(n));
    const layers = [inputSize, ...hidden, outputSize];
    
    this.network = {
      layers: layers,
      activation: activation,
      weights: this.initializeWeights(layers),
      totalParams: this.countParams(layers)
    };
    
    this.renderNetworkBuilder();
    this.updateStats();
  }
  
  initializeWeights(layers) {
    const weights = [];
    for (let i = 0; i < layers.length - 1; i++) {
      weights.push({
        W: this.randomMatrix(layers[i+1], layers[i]),
        b: new Array(layers[i+1]).fill(0)
      });
    }
    return weights;
  }
  
  randomMatrix(rows, cols) {
    // Xavier initialization
    const scale = Math.sqrt(2.0 / (rows + cols));
    const matrix = [];
    for (let i = 0; i < rows; i++) {
      matrix.push([]);
      for (let j = 0; j < cols; j++) {
        matrix[i].push((Math.random() * 2 - 1) * scale);
      }
    }
    return matrix;
  }
  
  countParams(layers) {
    let total = 0;
    for (let i = 0; i < layers.length - 1; i++) {
      total += layers[i] * layers[i+1] + layers[i+1]; // weights + biases
    }
    return total;
  }
  
  renderNetworkBuilder() {
    const svg = d3.select('#nn-builder-svg');
    svg.selectAll('*').remove();
    
    const width = svg.node().getBoundingClientRect().width;
    const height = 350;
    const layers = this.network.layers;
    const maxNeurons = Math.max(...layers);
    const layerSpacing = (width - 120) / (layers.length - 1);
    
    const g = svg.append('g').attr('transform', 'translate(60, 30)');
    
    // Draw connections
    for (let l = 0; l < layers.length - 1; l++) {
      const layerY1 = (height - 60 - layers[l] * 40) / 2;
      const layerY2 = (height - 60 - layers[l+1] * 40) / 2;
      
      for (let i = 0; i < layers[l]; i++) {
        for (let j = 0; j < layers[l+1]; j++) {
          const weight = this.network.weights[l].W[j][i];
          const opacity = Math.min(Math.abs(weight) * 2, 1);
          const color = weight > 0 ? '#6366f1' : '#e94560';
          
          g.append('line')
            .attr('x1', l * layerSpacing)
            .attr('y1', layerY1 + i * 40)
            .attr('x2', (l + 1) * layerSpacing)
            .attr('y2', layerY2 + j * 40)
            .attr('stroke', color)
            .attr('stroke-width', Math.abs(weight) * 2 + 0.5)
            .attr('opacity', opacity * 0.6);
        }
      }
    }
    
    // Draw neurons
    for (let l = 0; l < layers.length; l++) {
      const layerY = (height - 60 - layers[l] * 40) / 2;
      
      for (let i = 0; i < layers[l]; i++) {
        const neuron = g.append('g')
          .attr('transform', `translate(${l * layerSpacing}, ${layerY + i * 40})`);
        
        neuron.append('circle')
          .attr('r', 16)
          .attr('fill', l === 0 ? '#22c55e' : l === layers.length - 1 ? '#e94560' : '#6366f1')
          .attr('stroke', '#fff')
          .attr('stroke-width', 2);
      }
      
      // Layer label
      g.append('text')
        .attr('x', l * layerSpacing)
        .attr('y', height - 40)
        .attr('text-anchor', 'middle')
        .attr('fill', '#94a3b8')
        .attr('font-size', 12)
        .text(l === 0 ? 'Input' : l === layers.length - 1 ? 'Output' : `Hidden ${l}`);
    }
  }
  
  updateStats() {
    const stats = document.getElementById('nn-stats');
    stats.innerHTML = `
      <div class="dl-stat">
        <div class="dl-stat-value">${this.network.layers.length}</div>
        <div class="dl-stat-label">Total Layers</div>
      </div>
      <div class="dl-stat">
        <div class="dl-stat-value">${this.network.totalParams.toLocaleString()}</div>
        <div class="dl-stat-label">Parameters</div>
      </div>
      <div class="dl-stat">
        <div class="dl-stat-value">${this.network.activation.toUpperCase()}</div>
        <div class="dl-stat-label">Activation</div>
      </div>
    `;
  }
  
  // === TRAINING VISUALIZATION ===
  initTraining() {
    document.getElementById('train-start-btn').addEventListener('click', () => this.toggleTraining());
    document.getElementById('train-step-btn').addEventListener('click', () => this.trainStep());
  }
  
  toggleTraining() {
    if (this.training) {
      this.training = false;
      document.getElementById('train-start-btn').textContent = '▶ Start Training';
    } else {
      this.training = true;
      document.getElementById('train-start-btn').textContent = '⏸ Pause';
      this.losses = [];
      this.epoch = 0;
      this.trainingLoop();
    }
  }
  
  trainingLoop() {
    if (!this.training) return;
    
    const maxEpochs = parseInt(document.getElementById('train-epochs').value);
    if (this.epoch >= maxEpochs) {
      this.training = false;
      document.getElementById('train-start-btn').textContent = '▶ Start Training';
      return;
    }
    
    this.trainStep();
    requestAnimationFrame(() => setTimeout(() => this.trainingLoop(), 50));
  }
  
  trainStep() {
    // Simulate training step
    const lr = parseFloat(document.getElementById('train-lr').value);
    const dataset = document.getElementById('train-dataset').value;
    
    this.epoch++;
    
    // Simulate loss decay with some noise
    const baseLoss = 1.0 / (1 + this.epoch * 0.05) + 0.1;
    const noise = (Math.random() - 0.5) * 0.1;
    const loss = Math.max(0.01, baseLoss + noise);
    this.losses.push(loss);
    
    this.renderLossChart();
    this.renderDecisionBoundary();
    this.updateLog(`Epoch ${this.epoch}: Loss = ${loss.toFixed(4)}`);
  }
  
  renderLossChart() {
    const svg = d3.select('#loss-svg');
    svg.selectAll('*').remove();
    
    const width = svg.node().getBoundingClientRect().width;
    const height = 200;
    const margin = {top: 20, right: 20, bottom: 30, left: 50};
    
    const x = d3.scaleLinear()
      .domain([0, Math.max(this.losses.length, 10)])
      .range([margin.left, width - margin.right]);
    
    const y = d3.scaleLinear()
      .domain([0, Math.max(...this.losses, 1)])
      .range([height - margin.bottom, margin.top]);
    
    const line = d3.line()
      .x((d, i) => x(i))
      .y(d => y(d))
      .curve(d3.curveMonotoneX);
    
    // Axes
    svg.append('g')
      .attr('transform', `translate(0,${height - margin.bottom})`)
      .call(d3.axisBottom(x).ticks(5))
      .attr('color', '#64748b');
    
    svg.append('g')
      .attr('transform', `translate(${margin.left},0)`)
      .call(d3.axisLeft(y).ticks(5))
      .attr('color', '#64748b');
    
    // Line
    svg.append('path')
      .datum(this.losses)
      .attr('fill', 'none')
      .attr('stroke', '#e94560')
      .attr('stroke-width', 2)
      .attr('d', line);
    
    // Current point
    if (this.losses.length > 0) {
      const lastIdx = this.losses.length - 1;
      svg.append('circle')
        .attr('cx', x(lastIdx))
        .attr('cy', y(this.losses[lastIdx]))
        .attr('r', 5)
        .attr('fill', '#e94560');
    }
  }
  
  renderDecisionBoundary() {
    const canvas = document.getElementById('decision-canvas');
    const ctx = canvas.getContext('2d');
    const size = 300;
    
    // Create a simple decision boundary visualization
    const imageData = ctx.createImageData(size, size);
    
    for (let y = 0; y < size; y++) {
      for (let x = 0; x < size; x++) {
        const px = (x - size/2) / (size/4);
        const py = (y - size/2) / (size/4);
        
        // Simulate a decision boundary that evolves with training
        const boundary = Math.sin(px * 2 + this.epoch * 0.1) * Math.cos(py * 2 - this.epoch * 0.05);
        const value = boundary + (Math.random() - 0.5) * 0.2;
        
        const idx = (y * size + x) * 4;
        if (value > 0) {
          // Class A: Blue
          imageData.data[idx] = 99;
          imageData.data[idx + 1] = 102;
          imageData.data[idx + 2] = 241;
        } else {
          // Class B: Red
          imageData.data[idx] = 233;
          imageData.data[idx + 1] = 69;
          imageData.data[idx + 2] = 96;
        }
        imageData.data[idx + 3] = 180;
      }
    }
    
    ctx.putImageData(imageData, 0, 0);
    
    // Draw decision boundary line
    ctx.strokeStyle = '#fff';
    ctx.lineWidth = 2;
    ctx.beginPath();
    for (let x = 0; x < size; x++) {
      const px = (x - size/2) / (size/4);
      const boundaryY = Math.sin(px * 2 + this.epoch * 0.1);
      const y = size/2 - boundaryY * (size/4);
      if (x === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
  }
  
  updateLog(message) {
    const log = document.getElementById('train-log');
    log.innerHTML += message + '<br>';
    log.scrollTop = log.scrollHeight;
  }
  
  // === BACKPROPAGATION VISUALIZATION ===
  initBackprop() {
    document.getElementById('bp-forward-btn').addEventListener('click', () => this.bpForward());
    document.getElementById('bp-loss-btn').addEventListener('click', () => this.bpLoss());
    document.getElementById('bp-backward-btn').addEventListener('click', () => this.bpBackward());
    document.getElementById('bp-update-btn').addEventListener('click', () => this.bpUpdate());
    document.getElementById('bp-reset-btn').addEventListener('click', () => this.bpReset());
    
    this.bpState = { step: 0, activations: [], gradients: [] };
    this.renderBackprop();
  }
  
  bpForward() {
    this.bpState.step = 1;
    this.bpState.activations = [[0.5, 0.3, 0.8], [0.6, 0.4, 0.7, 0.5], [0.8, 0.2]];
    this.renderBackprop();
    this.setExplanation('Forward Pass', 
      'Data flows from input to output. Each neuron computes z = Wx + b, then applies activation g(z).');
    this.setMath('$$a^{[l]} = g(W^{[l]} a^{[l-1]} + b^{[l]})$$');
  }
  
  bpLoss() {
    this.bpState.step = 2;
    this.renderBackprop();
    this.setExplanation('Compute Loss', 
      'Compare prediction to target. The loss gradient ∂L/∂a[L] tells us how wrong we are.');
    this.setMath('$$L = \\frac{1}{2}(y - \\hat{y})^2 \\qquad \\frac{\\partial L}{\\partial a^{[L]}} = \\hat{y} - y$$');
  }
  
  bpBackward() {
    this.bpState.step = 3;
    this.bpState.gradients = [[0.1, -0.2], [0.05, -0.1, 0.08, -0.03], [0.02, -0.04, 0.03]];
    this.renderBackprop();
    this.setExplanation('Backward Pass (Backpropagation)', 
      'Gradients flow backwards through the network. Each layer receives gradient from above and multiplies by local gradient (chain rule).');
    this.setMath('$$\\frac{\\partial L}{\\partial W^{[l]}} = \\frac{\\partial L}{\\partial a^{[l]}} \\cdot \\frac{\\partial a^{[l]}}{\\partial z^{[l]}} \\cdot \\frac{\\partial z^{[l]}}{\\partial W^{[l]}}$$');
  }
  
  bpUpdate() {
    this.bpState.step = 4;
    this.renderBackprop();
    this.setExplanation('Update Weights', 
      'Move each weight in the direction that reduces loss. The learning rate α controls step size.');
    this.setMath('$$W := W - \\alpha \\frac{\\partial L}{\\partial W}$$');
  }
  
  bpReset() {
    this.bpState = { step: 0, activations: [], gradients: [] };
    this.renderBackprop();
    document.getElementById('bp-explanation').innerHTML = '';
    document.getElementById('bp-math').innerHTML = '';
  }
  
  setExplanation(title, text) {
    document.getElementById('bp-explanation').innerHTML = `<strong>${title}:</strong> ${text}`;
  }
  
  setMath(latex) {
    const el = document.getElementById('bp-math');
    el.innerHTML = latex;
    if (typeof renderMathInElement === 'function') {
      renderMathInElement(el, {
        delimiters: [{left: '$$', right: '$$', display: true}]
      });
    }
  }
  
  renderBackprop() {
    const svg = d3.select('#bp-svg');
    svg.selectAll('*').remove();
    
    const width = svg.node().getBoundingClientRect().width;
    const height = 300;
    const layers = [3, 4, 2];
    const layerSpacing = (width - 160) / (layers.length - 1);
    
    const g = svg.append('g').attr('transform', 'translate(80, 40)');
    
    // Draw based on current step
    const { step, activations, gradients } = this.bpState;
    
    // Draw connections
    for (let l = 0; l < layers.length - 1; l++) {
      const layerY1 = (height - 80 - layers[l] * 45) / 2;
      const layerY2 = (height - 80 - layers[l+1] * 45) / 2;
      
      for (let i = 0; i < layers[l]; i++) {
        for (let j = 0; j < layers[l+1]; j++) {
          let color = '#475569';
          let strokeWidth = 1;
          
          if (step === 1 && l < step) {
            color = '#6366f1';
            strokeWidth = 2;
          } else if (step === 3 && l >= layers.length - 2 - (step - 3)) {
            color = '#e94560';
            strokeWidth = 2;
          }
          
          g.append('line')
            .attr('x1', l * layerSpacing)
            .attr('y1', layerY1 + i * 45)
            .attr('x2', (l + 1) * layerSpacing)
            .attr('y2', layerY2 + j * 45)
            .attr('stroke', color)
            .attr('stroke-width', strokeWidth)
            .attr('opacity', 0.6);
        }
      }
    }
    
    // Draw neurons
    for (let l = 0; l < layers.length; l++) {
      const layerY = (height - 80 - layers[l] * 45) / 2;
      
      for (let i = 0; i < layers[l]; i++) {
        let fillColor = '#0f3460';
        
        if (step === 1 && activations[l]) {
          fillColor = '#6366f1';
        } else if (step === 2 && l === layers.length - 1) {
          fillColor = '#f59e0b';
        } else if (step === 3 && gradients[l]) {
          fillColor = '#e94560';
        } else if (step === 4) {
          fillColor = '#22c55e';
        }
        
        const neuron = g.append('g')
          .attr('transform', `translate(${l * layerSpacing}, ${layerY + i * 45})`);
        
        neuron.append('circle')
          .attr('r', 18)
          .attr('fill', fillColor)
          .attr('stroke', '#fff')
          .attr('stroke-width', 2);
        
        // Show activation value
        if (step >= 1 && activations[l] && activations[l][i] !== undefined) {
          neuron.append('text')
            .attr('dy', 4)
            .attr('text-anchor', 'middle')
            .attr('fill', '#fff')
            .attr('font-size', 10)
            .text(activations[l][i].toFixed(1));
        }
        
        // Show gradient value
        if (step === 3 && gradients[l] && gradients[l][i] !== undefined) {
          neuron.append('text')
            .attr('dy', -25)
            .attr('text-anchor', 'middle')
            .attr('fill', '#e94560')
            .attr('font-size', 9)
            .text(`∇${gradients[l][i].toFixed(2)}`);
        }
      }
      
      // Layer label
      g.append('text')
        .attr('x', l * layerSpacing)
        .attr('y', height - 50)
        .attr('text-anchor', 'middle')
        .attr('fill', '#94a3b8')
        .attr('font-size', 12)
        .text(l === 0 ? 'Input' : l === layers.length - 1 ? 'Output' : 'Hidden');
    }
    
    // Draw arrow indicating direction
    if (step === 1) {
      g.append('text')
        .attr('x', width / 2 - 40)
        .attr('y', 20)
        .attr('fill', '#6366f1')
        .attr('font-size', 14)
        .text('→ Forward');
    } else if (step === 3) {
      g.append('text')
        .attr('x', width / 2 - 40)
        .attr('y', 20)
        .attr('fill', '#e94560')
        .attr('font-size', 14)
        .text('← Backward');
    }
  }
}

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
  window.dlExperience = new DeepLearningExperience();
});
</script>
'''


def register_deep_learning_routes(app):
    """Register deep learning experience routes."""
    from flask import jsonify
    
    @app.route("/api/dl-experience/activate")
    def api_dl_activate():
        return jsonify({"ok": True, "message": "Deep Learning Experience activated"})


def get_algorithm_animation_html() -> str:
    """Return animated algorithm step-through visualizations."""
    return '''
<!-- Algorithm Animation Player -->
<div id="algo-player" class="algo-player" style="display:none;">
  <div class="algo-header">
    <h3 id="algo-title">Algorithm Animation</h3>
    <div class="algo-speed">
      <label>Speed:</label>
      <input type="range" id="algo-speed-slider" min="100" max="2000" value="500">
    </div>
    <button id="algo-close" class="viz-btn viz-close">✕</button>
  </div>
  
  <div class="algo-main">
    <div class="algo-code-panel">
      <h4>Code</h4>
      <pre id="algo-code"><code></code></pre>
    </div>
    <div class="algo-viz-panel">
      <h4>Visualization</h4>
      <svg id="algo-viz-svg" width="100%" height="300"></svg>
    </div>
  </div>
  
  <div class="algo-controls">
    <button id="algo-restart" class="algo-btn">⏮ Restart</button>
    <button id="algo-prev" class="algo-btn">◀ Prev</button>
    <button id="algo-play" class="algo-btn primary">▶ Play</button>
    <button id="algo-next" class="algo-btn">Next ▶</button>
    <span id="algo-step-counter" class="algo-counter">Step 0/0</span>
  </div>
  
  <div id="algo-explanation" class="algo-explanation"></div>
  <div id="algo-state" class="algo-state"></div>
</div>

<style>
.algo-player {
  background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
  border-radius: 16px;
  padding: 20px;
  margin: 20px 0;
  border: 1px solid #22c55e;
  box-shadow: 0 10px 40px rgba(34, 197, 94, 0.15);
}
.algo-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 16px;
}
.algo-header h3 {
  color: #22c55e;
  margin: 0;
}
.algo-speed {
  display: flex;
  align-items: center;
  gap: 8px;
  color: #94a3b8;
}
.algo-speed input {
  width: 100px;
}
.algo-main {
  display: grid;
  grid-template-columns: 1fr 1.5fr;
  gap: 16px;
  margin-bottom: 16px;
}
.algo-code-panel, .algo-viz-panel {
  background: rgba(0, 0, 0, 0.3);
  border-radius: 12px;
  padding: 16px;
}
.algo-code-panel h4, .algo-viz-panel h4 {
  color: #94a3b8;
  margin: 0 0 12px 0;
  font-size: 0.9rem;
}
#algo-code {
  background: #0f172a;
  padding: 16px;
  border-radius: 8px;
  overflow-x: auto;
  font-size: 0.85rem;
  line-height: 1.6;
}
#algo-code code {
  color: #e2e8f0;
}
#algo-code .highlight {
  background: rgba(34, 197, 94, 0.3);
  display: block;
  margin: 0 -16px;
  padding: 0 16px;
}
.algo-controls {
  display: flex;
  justify-content: center;
  align-items: center;
  gap: 12px;
  margin-bottom: 16px;
}
.algo-btn {
  padding: 10px 20px;
  border-radius: 8px;
  border: 1px solid #22c55e;
  background: #1e293b;
  color: #22c55e;
  cursor: pointer;
  transition: all 0.2s;
}
.algo-btn:hover { background: #22c55e; color: white; }
.algo-btn.primary {
  background: #22c55e;
  color: white;
}
.algo-counter {
  color: #94a3b8;
  font-size: 0.9rem;
}
.algo-explanation {
  background: rgba(34, 197, 94, 0.1);
  border-left: 3px solid #22c55e;
  padding: 16px;
  border-radius: 0 12px 12px 0;
  color: #e2e8f0;
  margin-bottom: 12px;
}
.algo-state {
  background: #0f172a;
  padding: 16px;
  border-radius: 8px;
  font-family: monospace;
  color: #6366f1;
}
</style>

<script>
class AlgorithmPlayer {
  constructor() {
    this.container = document.getElementById('algo-player');
    this.steps = [];
    this.currentStep = 0;
    this.playing = false;
    this.playInterval = null;
    this.algorithm = null;
    
    this.initControls();
  }
  
  initControls() {
    document.getElementById('algo-play').addEventListener('click', () => this.togglePlay());
    document.getElementById('algo-prev').addEventListener('click', () => this.prevStep());
    document.getElementById('algo-next').addEventListener('click', () => this.nextStep());
    document.getElementById('algo-restart').addEventListener('click', () => this.restart());
    document.getElementById('algo-close').addEventListener('click', () => this.hide());
  }
  
  show(title) {
    this.container.style.display = 'block';
    document.getElementById('algo-title').textContent = title;
  }
  
  hide() {
    this.container.style.display = 'none';
    this.stopPlay();
  }
  
  loadAlgorithm(algo) {
    this.algorithm = algo;
    this.steps = algo.steps;
    this.currentStep = 0;
    
    // Set code
    document.getElementById('algo-code').innerHTML = '<code>' + algo.code + '</code>';
    
    this.renderStep();
  }
  
  renderStep() {
    const step = this.steps[this.currentStep];
    if (!step) return;
    
    // Update counter
    document.getElementById('algo-step-counter').textContent = 
      `Step ${this.currentStep + 1}/${this.steps.length}`;
    
    // Update explanation
    document.getElementById('algo-explanation').innerHTML = 
      `<strong>${step.title}:</strong> ${step.description}`;
    
    // Update state
    document.getElementById('algo-state').textContent = step.state || '';
    
    // Highlight code line
    this.highlightCodeLine(step.line);
    
    // Render visualization
    this.renderVisualization(step);
  }
  
  highlightCodeLine(line) {
    const codeEl = document.getElementById('algo-code');
    const lines = codeEl.textContent.split('\\n');
    const highlighted = lines.map((l, i) => 
      i === line - 1 ? `<span class="highlight">${l}</span>` : l
    ).join('\\n');
    codeEl.innerHTML = '<code>' + highlighted + '</code>';
  }
  
  renderVisualization(step) {
    const svg = d3.select('#algo-viz-svg');
    svg.selectAll('*').remove();
    
    // Render based on visualization type
    if (step.viz) {
      step.viz(svg, step.data);
    }
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
  
  restart() {
    this.currentStep = 0;
    this.renderStep();
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
    document.getElementById('algo-play').textContent = '⏸ Pause';
    const speed = parseInt(document.getElementById('algo-speed-slider').value);
    this.playInterval = setInterval(() => this.nextStep(), speed);
  }
  
  stopPlay() {
    this.playing = false;
    document.getElementById('algo-play').textContent = '▶ Play';
    if (this.playInterval) {
      clearInterval(this.playInterval);
      this.playInterval = null;
    }
  }
}

window.algoPlayer = new AlgorithmPlayer();
</script>
'''
