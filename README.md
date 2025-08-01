# PyTorch-Models
Deep Learning Research Projects PyTorch | Computer Vision | Genomics | Model Optimization  Developed 7 end-to-end PyTorch models achieving 85-99.5% accuracy across computer vision, Polynomial Regression, and Neural Networks

---
## Polynomial Regression for Genomic Expression Modeling

**Objective**: Develop a custom polynomial regression model in PyTorch to predict gene expression levels from nucleotide combinations (synthetic data), accurately recovering known biological interactions (e.g., A², G×C, A×T×G) with R²=0.99

**Key Highlights**:
* Simulated gene expression as
  > y = 0.3*A² + 0.1*T - 0.4*G*C + 0.2*A*T*G

* Model learned coefficients:
  * 0.3000 (A²),
  * -0.4001 (G×C),
  * 0.2000 (A×T×G)

  (matches ground truth)

* GenomePolynomialMode() Dynamically encodes nucleotide interactions (e.g., AA, GC, ATG) based on user needs
* NucleotideEncoder() Maps biological terms (e.g., "ATG") to tensor operations
* Near-perfect fit (R²=0.99) on test data (May be overfitted)
* Visualized predictions vs. ground truth with matplotlib

#### Usage
```python
encoder = NucleotideEncoder(max_degree=3)
input_combos = ['aa', 't', 'gc', 'atg']
specified_terms = encoder.encode(input_combos)
genomeModel = GenomePolynomialMode(degrees=3, specified_terms = specified_terms)

genomeModel.eval()
with torch.no_grad():
  test_prediction = genomeModel(x_test)
```
---
## Deep Learning Neural Network Classification Projects

> Five end-to-end PyTorch projects demonstrating non-linear data classification, genomic analysis (with synthetic data), and model interpretability

### Skills
* Custom architectures, training loops, TorchMetrics
* Simulated non-linear datasets (spirals, moons, blobs, tensors)
* LR tuning, node sizing, epoch selection
* Decision boundary plots, SHAP values (genomics)

### 1. Genomic Binary Classification (Linear)

**Objective**: Predicted disease risk from synthetic SNP counts using a linear NN

**Key Achievements**:
* 86% F1-score with class-weighted BCE loss, handling imbalance
* Demonstrated feature importance analysis
* Handling Class Imbalance, Embedding Layers, Privacy-Aware ML
* Future Use: Clinical diagnostics (e.g., polygenic risk scores)

#### Usage
```python
genomeModel0 = LinearBinaryClassification()
genomeModel0.eval()
with torch.no_grad():
  test_pred = genomeModel0(x_test)
```

### 2. Non-Linear Binary Classification
**Objective**: Binary classification of synthetic data with custom NN architectures

**Key Achievements**:
* Compared explicit vs. sequential layer definitions in PyTorch
* Achieved 95%+ accuracy with flexible model design
* Custom nn.Module Design, Debugging, Model Interpretability
* Future Use: Fraud detection, industrial quality control

#### Usage
```python
model1 = NonLinearBinaryClassification()
model1.eval()
with torch.inference_mode():
  y_logits = model1(x_test).squeeze()
  y_predictions = torch.round(torch.sigmoid(y_logits))

  test_loss = lossFunction(test_logits, y_test)
  test_acc = accuracy_fn(y_test, test_predictions)

print(f"Test Loss: {test_loss:.5f}, Test Accuracy: {test_acc:.4f}%")
```

### 3. Multi-Class Subtype Classification
**Objective**: Simulated cancer subtype prediction from gene expression profiles (synthetic blobs)

**Key Achievements**:
* 99.5% accuracy with a sequential NN, showcasing scalability to high-dimensional genomic data
* Emphasized CrossEntropyLoss for multi-class problems
* Future Use: Real-world RNA-seq analysis (e.g., TCGA datasets)

#### Usage
```python
model1 = MultiClassification(input_dimensions=2, output_dimensions=4, nodes=128)
model1.eval()
with torch.inference_mode():
  y_logits = model1(x_test)
  y_prob = torch.softmax(y_logits, dim=1)
  y_predictions = y_prob.argmax(dim=1)
```

### 4. Non-Linear Binary Classification (Moons)
**Objective**: Separated moon-shaped clusters with noise to simulate real-world binary classification

**Key Achievements**:
* 98% test accuracy using a 4-layer NN with ReLU activations
* Highlighted SGD vs. Adam trade-offs for noisy data
* Non-linear Modeling, BCEWithLogitsLoss, Data Augmentation
* Future Use: Anomaly detection, EEG signal classification

#### Usage
```python
model2 = MoonBinaryClassifier(input_dimensions=2, output_dimensions=1, nodes=128)
model2.eval()
plot_decision_boundary(model2, x_test, y_test)
```

### 5. Multi-Class Spiral Classification
**Objective**: Classified 3 intertwined spiral clusters (non-linear data) using a neural network

**Key Achievements**:
* Achieved 96.67% accuracy with a 3-layer ReLU network (128 nodes/layer)
* Demonstrated ability to model complex decision boundaries (visualized with plot_decision_boundary)
* Synthetic Data Generation, Hyperparameter Tuning
* Future Use: Customer segmentation, medical imaging (e.g., tumor type classification).

#### Usage
```python
model3 = SpiralClassificationModel(2,3,128)
model3.eval()
with torch.inference_mode():
    test_logits = model3(x_test)
```
---
## FashionMNIST Classification with TinyVGG
**Objective**: Designed adn trained a lightweight convolutional neural network (TinyVGG) to classify FashionMNIST images, achieving 91.11% accuracy while demonstrating core PyTorchh workflows, model optimisation, and evaluation

**Key Achievements**:
* Model architecture design (nn.Module), data loading (DataLoader), and training loops
* CNN Architecture: Implemented a multi-layer CNN with ReLU activations, max pooling, and linear classification - TinyVGG Architecture
* Research Best Practices: Hyperparameter tuning (batch size, learning rate), loss/accuracy tracking, and confusion matrix analysis
* Reproducibility: Used fixed random seeds, modular functions for training/evaluation, and visualization tools (Matplotlib, mlxtend)

#### Usage
```python
model_0_results = eval_model(model=model0, data_loader=test_dataloader,
    loss_fn=lossFunction, accuracy_fn=accuracy_fn
)
```
---



