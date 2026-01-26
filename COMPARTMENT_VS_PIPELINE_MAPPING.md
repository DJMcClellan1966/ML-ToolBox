# Compartment vs. Pipeline Design Mapping

## Question

**Do the 4 toolbox compartments correspond to the FTI (Feature/Training/Inference) pipeline design?**

## Short Answer

**Partially, but not directly.** The compartments are organized by **functionality type**, while the pipelines are organized by **workflow stages**. They serve different purposes but complement each other.

---

## The 4 Compartments

```
MLToolbox
├── 1. Data Compartment
│   └── Preprocessing, validation, transformation
├── 2. Infrastructure Compartment
│   └── Kernels, AI components, LLM
├── 3. Algorithms Compartment
│   └── Models, evaluation, tuning, ensembles
└── 4. MLOps Compartment
    └── Deployment, monitoring, A/B testing, experiment tracking
```

## The FTI Pipelines

```
UnifiedMLPipeline
├── Feature Pipeline
│   └── Data Ingestion → Preprocessing → Feature Engineering → Feature Selection → Feature Store
├── Training Pipeline
│   └── Model Training → Hyperparameter Tuning → Model Evaluation → Model Validation → Model Registry
└── Inference Pipeline
    └── Model Serving → Batch Inference → Real-time Inference → A/B Testing → Monitoring
```

---

## Mapping Analysis

### Feature Pipeline ↔ Compartments

| Feature Pipeline Stage | Primary Compartment | Supporting Compartments |
|------------------------|---------------------|-------------------------|
| **Data Ingestion** | Data Compartment | Infrastructure (kernels) |
| **Preprocessing** | Data Compartment | Infrastructure (kernels) |
| **Feature Engineering** | Data Compartment | Infrastructure (FeatureEngineeringKernel) |
| **Feature Selection** | Data Compartment | Algorithms (feature selection methods) |
| **Feature Store** | Data Compartment | MLOps (storage, versioning) |

**Mapping:** Feature Pipeline ≈ **Data Compartment** (primary) + Infrastructure (support)

---

### Training Pipeline ↔ Compartments

| Training Pipeline Stage | Primary Compartment | Supporting Compartments |
|------------------------|---------------------|-------------------------|
| **Model Training** | Algorithms Compartment | Infrastructure (AlgorithmKernel) |
| **Hyperparameter Tuning** | Algorithms Compartment | Infrastructure (TuningKernel) |
| **Model Evaluation** | Algorithms Compartment | Infrastructure (EvaluationKernel) |
| **Model Validation** | Algorithms Compartment | Data (cross-validation) |
| **Model Registry** | MLOps Compartment | Algorithms (model storage) |

**Mapping:** Training Pipeline ≈ **Algorithms Compartment** (primary) + MLOps (registry)

---

### Inference Pipeline ↔ Compartments

| Inference Pipeline Stage | Primary Compartment | Supporting Compartments |
|-------------------------|---------------------|-------------------------|
| **Model Serving** | MLOps Compartment | Infrastructure (ServingKernel) |
| **Batch Inference** | MLOps Compartment | Algorithms (model execution) |
| **Real-time Inference** | MLOps Compartment | Infrastructure (optimization) |
| **A/B Testing** | MLOps Compartment | Algorithms (model comparison) |
| **Monitoring** | MLOps Compartment | Infrastructure (metrics) |

**Mapping:** Inference Pipeline ≈ **MLOps Compartment** (primary) + Algorithms (models)

---

## Visual Mapping

```
┌─────────────────────────────────────────────────────────────┐
│                    FEATURE PIPELINE                         │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Data Compartment (PRIMARY)                          │   │
│  │  • Data Ingestion                                    │   │
│  │  • Preprocessing                                     │   │
│  │  • Feature Engineering                               │   │
│  │  • Feature Selection                                  │   │
│  └──────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Infrastructure Compartment (SUPPORT)               │   │
│  │  • FeatureEngineeringKernel                          │   │
│  │  • Computational Kernels                             │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                    TRAINING PIPELINE                        │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Algorithms Compartment (PRIMARY)                     │   │
│  │  • Model Training                                     │   │
│  │  • Hyperparameter Tuning                              │   │
│  │  • Model Evaluation                                   │   │
│  │  • Model Validation                                    │   │
│  └──────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  MLOps Compartment (SUPPORT)                         │   │
│  │  • Model Registry                                    │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                    INFERENCE PIPELINE                       │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  MLOps Compartment (PRIMARY)                        │   │
│  │  • Model Serving                                     │   │
│  │  • Batch Inference                                    │   │
│  │  • Real-time Inference                                │   │
│  │  • A/B Testing                                        │   │
│  │  • Monitoring                                         │   │
│  └──────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Algorithms Compartment (SUPPORT)                    │   │
│  │  • Model Execution                                    │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

---

## Key Differences

### 1. **Organization Principle**

| Aspect | Compartments | Pipelines |
|--------|-------------|-----------|
| **Organization** | By functionality type | By workflow stage |
| **Purpose** | Code organization | Workflow orchestration |
| **Focus** | What components exist | How data flows |
| **Usage** | Mix and match tools | Sequential execution |

### 2. **Infrastructure Compartment**

The **Infrastructure Compartment** is special:
- **Not directly mapped** to any single pipeline
- **Supports all pipelines** with kernels and optimization
- Provides:
  - Computational Kernels (performance)
  - Optimization Kernels (8 unified kernels)
  - AI Components (LLM, agents)
  - Medulla Optimizer (resource management)

**Mapping:** Infrastructure ≈ **Cross-cutting concerns** for all pipelines

---

## Complete Mapping Table

| Compartment | Primary Pipeline | Secondary Pipelines | Role |
|------------|------------------|-------------------|------|
| **Data** | Feature Pipeline | Training (validation) | Data processing |
| **Infrastructure** | All pipelines | - | Performance & optimization |
| **Algorithms** | Training Pipeline | Inference (execution) | Model operations |
| **MLOps** | Inference Pipeline | Training (registry) | Production operations |

---

## How They Work Together

### Example: Training a Model

**Using Compartments (Traditional):**
```python
toolbox = MLToolbox()

# Data Compartment
X_processed = toolbox.data.preprocess(X)

# Algorithms Compartment
model = toolbox.algorithms.train(X_processed, y)

# MLOps Compartment
toolbox.mlops.register_model(model)
```

**Using Pipelines (New):**
```python
pipeline = UnifiedMLPipeline(toolbox)

# Feature Pipeline (uses Data + Infrastructure compartments)
# Training Pipeline (uses Algorithms + MLOps compartments)
result = pipeline.execute(X, y, mode='train')
```

**Behind the scenes, the pipeline uses:**
- Feature Pipeline → Data Compartment + Infrastructure Kernels
- Training Pipeline → Algorithms Compartment + MLOps Registry
- All orchestrated automatically

---

## Answer Summary

### Do they correspond?

**Yes, but with important nuances:**

1. **Feature Pipeline** ≈ **Data Compartment** (primary)
2. **Training Pipeline** ≈ **Algorithms Compartment** (primary)
3. **Inference Pipeline** ≈ **MLOps Compartment** (primary)
4. **Infrastructure Compartment** ≈ **Supports all pipelines**

### Key Insight

- **Compartments** = **What** (components available)
- **Pipelines** = **How** (workflow orchestration)

The pipelines **use** the compartments but organize them differently:
- Compartments organize by **functionality**
- Pipelines organize by **workflow**

### Best Practice

- **Use compartments** for: Direct component access, custom workflows, mixing tools
- **Use pipelines** for: Standard workflows, production systems, reproducibility

---

## Conclusion

The 4 compartments **partially correspond** to the FTI pipeline design, but they serve different purposes:

- **Compartments** = Library organization (what's available)
- **Pipelines** = Workflow orchestration (how to use it)

The pipelines **leverage** the compartments but provide a higher-level, workflow-oriented interface. This is why we implemented both - they complement each other!

**The unified pipeline system bridges the gap** between the compartment-based organization and industry-standard pipeline workflows.
