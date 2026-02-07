"""
Curriculum: Deep Learning & Statistical ML — Goodfellow/Bengio/Courville, Bishop, ESL (Hastie et al.), Burkov.
"""
from typing import Dict, Any, List

LEVELS = ["basics", "intermediate", "advanced", "expert"]

BOOKS = [
    {"id": "goodfellow", "name": "Deep Learning (Goodfellow et al.)", "short": "Deep Learning", "color": "#2563eb"},
    {"id": "bishop", "name": "Pattern Recognition & ML (Bishop)", "short": "Bishop", "color": "#059669"},
    {"id": "esl", "name": "Elements of Statistical Learning", "short": "ESL", "color": "#7c3aed"},
    {"id": "burkov", "name": "Hundred-Page ML (Burkov)", "short": "Burkov", "color": "#dc2626"},
]

CURRICULUM: List[Dict[str, Any]] = [
    # Neural Network Fundamentals
    {"id": "dl_feedforward", "book_id": "goodfellow", "level": "basics", "title": "Feedforward Neural Networks",
     "learn": "Multi-layer perceptrons (MLPs): input layer → hidden layers → output layer. Each layer applies W*x + b followed by activation function (ReLU, sigmoid, tanh). Universal approximation theorem: can approximate any continuous function.",
     "try_code": "import torch.nn as nn\nmodel = nn.Sequential(\n    nn.Linear(784, 256),\n    nn.ReLU(),\n    nn.Linear(256, 10)\n)",
     "try_demo": "dl_feedforward"},
    {"id": "dl_activation", "book_id": "goodfellow", "level": "basics", "title": "Activation Functions",
     "learn": "Non-linear activations enable networks to learn complex patterns. ReLU: f(x)=max(0,x), fast and effective. Sigmoid: σ(x)=1/(1+e^-x), output [0,1]. Tanh: output [-1,1]. Leaky ReLU, ELU, GELU for advanced use.",
     "try_code": "import torch.nn as nn\nactivation = nn.ReLU()  # or Sigmoid(), Tanh(), GELU()",
     "try_demo": "dl_activation_compare"},
    {"id": "dl_backprop", "book_id": "goodfellow", "level": "intermediate", "title": "Backpropagation Algorithm",
     "learn": "Backward pass computes gradients via chain rule. For each layer: ∂L/∂W = ∂L/∂out * ∂out/∂W. Automatic differentiation frameworks (PyTorch, TensorFlow) handle this automatically.",
     "try_code": "import torch\nloss = criterion(output, target)\nloss.backward()  # Computes all gradients automatically",
     "try_demo": "dl_backprop_viz"},
    
    # Convolutional Neural Networks
    {"id": "dl_cnn_basics", "book_id": "goodfellow", "level": "intermediate", "title": "Convolutional Layers (CNNs)",
     "learn": "Convolution: slide kernel over input, compute dot product. Parameters shared across spatial locations (translation equivariance). Output feature maps detect patterns (edges, textures, objects).",
     "try_code": "import torch.nn as nn\nconv = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)",
     "try_demo": "dl_convolution"},
    {"id": "dl_pooling", "book_id": "goodfellow", "level": "intermediate", "title": "Pooling Layers",
     "learn": "Pooling reduces spatial dimensions. MaxPool: take maximum in each region (preserves strong activations). AvgPool: take average (smoother). Provides translation invariance and reduces parameters.",
     "try_code": "import torch.nn as nn\npool = nn.MaxPool2d(kernel_size=2, stride=2)  # Halves spatial dims",
     "try_demo": None},
    {"id": "dl_resnet", "book_id": "goodfellow", "level": "advanced", "title": "Residual Networks (ResNet)",
     "learn": "Skip connections: F(x) + x. Allows training very deep networks (50, 101, 152 layers). Gradients flow directly through skip connections, avoiding vanishing gradient problem.",
     "try_code": "# Residual block\nout = F.relu(self.conv1(x))\nout = self.conv2(out)\nout += x  # Skip connection\nout = F.relu(out)",
     "try_demo": None},
    
    # Recurrent Networks & Sequences
    {"id": "dl_rnn_basics", "book_id": "goodfellow", "level": "advanced", "title": "Recurrent Neural Networks (RNNs)",
     "learn": "Process sequences: h_t = tanh(W_hh * h_(t-1) + W_xh * x_t + b). Hidden state h_t carries information from previous time steps. Used for text, speech, time series.",
     "try_code": "import torch.nn as nn\nrnn = nn.RNN(input_size=10, hidden_size=20, num_layers=2)",
     "try_demo": "dl_rnn_sequence"},
    {"id": "dl_lstm", "book_id": "goodfellow", "level": "advanced", "title": "LSTM (Long Short-Term Memory)",
     "learn": "Solves vanishing gradient in RNNs. Uses gates: forget gate (what to forget), input gate (what to add), output gate (what to output). Cell state c_t carries long-term memory.",
     "try_code": "import torch.nn as nn\nlstm = nn.LSTM(input_size=10, hidden_size=20, num_layers=2)",
     "try_demo": "dl_lstm_gates"},
    {"id": "dl_attention", "book_id": "goodfellow", "level": "advanced", "title": "Attention Mechanism",
     "learn": "Weighted combination of inputs: α_i = softmax(score(query, key_i)); output = Σ α_i * value_i. Allows model to focus on relevant parts. Used in transformers.",
     "try_code": "import torch\nimport torch.nn.functional as F\n# Scaled dot-product attention\nscores = torch.matmul(Q, K.transpose(-2, -1)) / sqrt(d_k)\nattention = F.softmax(scores, dim=-1)\noutput = torch.matmul(attention, V)",
     "try_demo": "dl_attention_viz"},
    {"id": "dl_transformer", "book_id": "goodfellow", "level": "advanced", "title": "Transformer Architecture",
     "learn": "Self-attention based model: encoder-decoder structure. Multi-head attention, positional encoding, feedforward layers, layer norm. Parallel processing (vs sequential RNN). Powers BERT, GPT, etc.",
     "try_code": "import torch.nn as nn\ntransformer = nn.Transformer(d_model=512, nhead=8, num_encoder_layers=6)",
     "try_demo": "dl_transformer_blocks"},
    
    # Regularization & Training
    {"id": "dl_regularization", "book_id": "goodfellow", "level": "intermediate", "title": "Regularization (Dropout, L2)",
     "learn": "Reduce overfitting: L2 weight decay, dropout (randomly zero activations), early stopping. Goodfellow Ch 7.",
     "try_code": "from three_books_methods import DeepLearningMethods\nm=DeepLearningMethods()\n# regularization in training",
     "try_demo": None},
    {"id": "dl_dropout", "book_id": "goodfellow", "level": "intermediate", "title": "Dropout Regularization",
     "learn": "Randomly zero activations during training (probability p, often 0.5). Forces network to learn redundant representations. At test time, scale by (1-p) or use all weights.",
     "try_code": "import torch.nn as nn\ndropout = nn.Dropout(p=0.5)  # 50% dropout",
     "try_demo": None},
    {"id": "dl_batchnorm", "book_id": "goodfellow", "level": "intermediate", "title": "Batch Normalization",
     "learn": "Normalize activations: (x - μ_batch) / σ_batch. Stabilizes training, allows higher learning rates, acts as regularization. Applied after conv/linear, before activation.",
     "try_code": "import torch.nn as nn\nbn = nn.BatchNorm2d(num_features=64)",
     "try_demo": None},
    {"id": "dl_data_augmentation", "book_id": "goodfellow", "level": "intermediate", "title": "Data Augmentation",
     "learn": "Artificially expand training data: random crops, flips, rotations, color jitter. Improves generalization without collecting more data. Critical for vision tasks.",
     "try_code": "from torchvision import transforms\naugment = transforms.Compose([\n    transforms.RandomHorizontalFlip(),\n    transforms.RandomCrop(32, padding=4),\n    transforms.ColorJitter(brightness=0.2)\n])",
     "try_demo": "dl_augmentation_viz"},
    
    # Optimization
    {"id": "dl_optimization", "book_id": "goodfellow", "level": "intermediate", "title": "Optimization (Adam, RMSprop)",
     "learn": "Adaptive learning rates: Adam, RMSprop. Momentum and Nesterov. Goodfellow Ch 8.",
     "try_code": "from three_books_methods import DeepLearningMethods\n# optimizer choice in fit()",
     "try_demo": None},
    {"id": "dl_adam", "book_id": "goodfellow", "level": "intermediate", "title": "Adam Optimizer",
     "learn": "Adaptive learning rates: combines momentum (first moment) and RMSprop (second moment). m_t = β1*m_(t-1) + (1-β1)*g_t; v_t = β2*v_(t-1) + (1-β2)*g_t². Update: θ -= lr * m_t / (√v_t + ε).",
     "try_code": "import torch.optim as optim\noptimizer = optim.Adam(model.parameters(), lr=0.001)",
     "try_demo": "dl_optimizer_compare"},
    {"id": "dl_learning_rate", "book_id": "goodfellow", "level": "intermediate", "title": "Learning Rate Scheduling",
     "learn": "Adjust learning rate during training. Step decay: reduce by factor every N epochs. Cosine annealing: smooth decrease. Warmup: start low, ramp up. Improves convergence and final accuracy.",
     "try_code": "from torch.optim.lr_scheduler import CosineAnnealingLR\nscheduler = CosineAnnealingLR(optimizer, T_max=100)",
     "try_demo": None},
    
    # Transfer Learning
    {"id": "dl_transfer_learning", "book_id": "goodfellow", "level": "intermediate", "title": "Transfer Learning",
     "learn": "Use pre-trained model (ImageNet, BERT) as starting point. Freeze early layers (general features), fine-tune later layers (task-specific). Much faster than training from scratch.",
     "try_code": "import torchvision.models as models\nmodel = models.resnet50(pretrained=True)\n# Freeze early layers\nfor param in model.parameters():\n    param.requires_grad = False",
     "try_demo": None},
    
    # Probabilistic ML (Bishop)
    {"id": "bishop_gaussian", "book_id": "bishop", "level": "intermediate", "title": "Gaussian Processes (Bishop)",
     "learn": "Non-parametric: prior over functions, kernel covariance. Posterior predictive distribution.",
     "try_code": "from three_books_methods import BishopMethods\nb=BishopMethods()\n# Gaussian process regression",
     "try_demo": None},
    {"id": "bishop_em", "book_id": "bishop", "level": "advanced", "title": "EM Algorithm (Bishop)",
     "learn": "Expectation-Maximization for latent variable models. E-step: q(z); M-step: maximize bound.",
     "try_code": "from ml_toolbox.textbook_concepts.probabilistic_ml import EMAlgorithm",
     "try_demo": None},
    
    # ESL Methods
    {"id": "esl_svm", "book_id": "esl", "level": "intermediate", "title": "Support Vector Machines (ESL)",
     "learn": "Max-margin classifier. Kernel trick for nonlinearity. C and gamma. ESL Ch 12.",
     "try_code": "from three_books_methods import ESLMethods\ne=ESLMethods()\n# e.support_vector_machine(X,y,kernel='rbf')",
     "try_demo": "esl_svm"},
    {"id": "esl_boosting", "book_id": "esl", "level": "intermediate", "title": "Gradient Boosting (ESL)",
     "learn": "Additive model: fit residuals sequentially. XGBoost, LightGBM use this idea.",
     "try_code": "from three_books_methods import ESLMethods\ne=ESLMethods()\n# gradient boosting classifier/regressor",
     "try_demo": None},
    
    # Practical ML (Burkov)
    {"id": "burkov_workflow", "book_id": "burkov", "level": "basics", "title": "ML Project Workflow (Burkov)",
     "learn": "Define problem → get data → train/val/test → iterate. Keep it simple first.",
     "try_code": "# Define metric, baseline, then improve",
     "try_demo": None},
    {"id": "burkov_ensemble", "book_id": "burkov", "level": "basics", "title": "Ensembles (Burkov)",
     "learn": "Bagging (e.g. Random Forest), boosting (e.g. AdaBoost), stacking. Combine weak learners.",
     "try_code": "from ml_toolbox.textbook_concepts.practical_ml import EnsembleMethods",
     "try_demo": None},
]


def get_curriculum(): return list(CURRICULUM)
def get_books(): return list(BOOKS)
def get_levels(): return list(LEVELS)
def get_by_book(book_id: str): return [c for c in CURRICULUM if c["book_id"] == book_id]
def get_by_level(level: str): return [c for c in CURRICULUM if c["level"] == level]
def get_item(item_id: str):
    for c in CURRICULUM:
        if c["id"] == item_id: return c
    return None
