# Lab 1 - Deep Learning: Classification et Régression avec PyTorch

**Université Abdelmalek Essaadi**  
**Faculté des Sciences et Techniques de Tanger**  
**Département Génie Informatique**  
**Professeur:** Pr. ELAACHAK LOTFI  
**Étudiante:** ABABRI Chaimae

---

## Aperçu

Ce repository contient l'implémentation du Lab 1 du cours Deep Learning dans le cadre du Master MBD. L'objectif principal est de se familiariser avec la bibliothèque PyTorch pour effectuer des tâches de **Classification** et **Régression** en établissant des architectures DNN/MLP.

Le lab est divisé en deux parties:

1. **Part 1:** Régression sur le dataset NYSE de Kaggle
2. **Part 2:** Classification Multi-classe sur le dataset Machine Predictive Maintenance de Kaggle

Tous les travaux ont été implémentés dans un notebook Kaggle pour un usage efficace du GPU et une gestion optimale des données.

---

## Structure du Repository
```
lab1-deep-learning/
├── notebookAtelier1.ipynb
└── README.md
```

---

## Part 1: Régression

### 1. Analyse Exploratoire des Données (EDA)

- Chargement et inspection du dataset
- Visualisations: distributions, tendances, corrélations, outliers

**Interprétation:**
- Corrélations très élevées entre features de prix (>0.99)
- Volumes asymétriques
- Aucune valeur manquante

### 2. Architecture du Modèle DNN
```python
RegressionMLP:
- Input Layer: 4 features
- Hidden Layer 1: 32 neurones (ReLU)
- Hidden Layer 2: 32 neurones (ReLU)
- Output Layer: 1 neurone
```

### 3. Optimisation des Hyper-paramètres

**Meilleurs paramètres (GridSearchCV):**
```python
{
    'epochs': 20,
    'hidden_size1': 32,
    'hidden_size2': 32,
    'lr': 0.01
}
```

### 4. Résultats

| Métrique | Modèle de Base | Avec Régularisation |
|----------|----------------|---------------------|
| R² Train | 0.9982 | -0.0012 |
| R² Test | 0.9983 | -0.0002 |
| Loss Test | 0.000005 | 0.002991 |

**Interprétation:**
- Modèle de base: Excellentes performances (R² = 0.998)
- Régularisation: Dégradation des performances (sur-régularisation)
- La régularisation n'est pas nécessaire pour ce problème

---

## Part 2: Classification Multi-classe

### 1. Prétraitement

- Nettoyage des données
- Encodage One-Hot de 'Type'
- Normalisation MinMaxScaler
- Suppression colonnes inutiles (UDI, Product ID, Target)

### 2. Analyse Exploratoire (EDA)

- Distribution des classes: Déséquilibre important
- Corrélation négative entre torque et vitesse de rotation
- Visualisations: histogrammes, heatmap, countplot

### 3. Équilibrage - SMOTE

**Distribution après SMOTE:**
- Toutes les classes: 9,652 exemples chacune
- Total: 57,912 exemples (équilibré)

### 4. Architecture du Modèle DNN
```python
ClassificationMLP:
- Input Layer: 8 features
- Hidden Layer 1: 64 neurones (ReLU)
- Hidden Layer 2: 32 neurones (ReLU)
- Output Layer: 6 neurones (softmax)
```

### 5. Optimisation des Hyper-paramètres

**Meilleurs paramètres (GridSearchCV):**
```python
{
    'epochs': 50,
    'hidden_size1': 64,
    'hidden_size2': 32,
    'lr': 0.01
}
```

### 6. Résultats

**Modèle de Base - Test Set:**

| Classe | Precision | Recall | F1-Score |
|--------|-----------|--------|----------|
| Heat Dissipation Failure | 0.99 | 1.00 | 0.99 |
| No Failure | 0.91 | 0.92 | 0.91 |
| Overstrain Failure | 1.00 | 0.98 | 0.99 |
| Power Failure | 0.99 | 1.00 | 1.00 |
| Random Failures | 0.98 | 0.91 | 0.94 |
| Tool Wear Failure | 0.97 | 1.00 | 0.98 |
| **Accuracy** | | | **0.97** |

**Modèle Régularisé - Test Set:**
- Accuracy: 0.17 (dégradation importante)
- Sur-régularisation avec les mêmes hyperparamètres

**Interprétation:**
- Modèle de base: Excellentes performances (97% accuracy)
- SMOTE: Efficace pour équilibrer les classes
- Régularisation: Nécessite un nouveau GridSearch avec hyperparamètres adaptés

---

## Outils Utilisés

- **Environnement:** Kaggle Notebook avec GPU T4 x2
- **Deep Learning:** PyTorch
- **Machine Learning:** scikit-learn (GridSearchCV, métriques)
- **Data Processing:** pandas, numpy
- **Visualisation:** matplotlib, seaborn
- **Balancing:** imbalanced-learn (SMOTE)

---

## Synthèse et Apprentissages

Durant ce lab, j'ai acquis une expérience pratique avec PyTorch pour construire et entraîner des architectures DNN/MLP.

**Compétences développées:**
- Analyse exploratoire des données (EDA)
- Prétraitement et normalisation
- Équilibrage de classes avec SMOTE
- Optimisation d'hyper-paramètres avec GridSearchCV
- Techniques de régularisation (Dropout, L2)
- Interprétation des métriques (R², accuracy, F1-score)

**Défis rencontrés:**
- Gestion de datasets volumineux (échantillonnage nécessaire)
- Configuration GPU pour accélération
- Sur-régularisation: Nécessité d'ajuster les hyperparamètres spécifiquement

**Conclusion:** 
Ce lab a souligné l'importance d'une préparation minutieuse des données et d'un tuning précis des modèles. La régularisation n'est pas toujours bénéfique et nécessite une optimisation spécifique.

---
 
