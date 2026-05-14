# Quantum-Temporal GCN for Recommender Systems

Ce dossier contient une version modulaire, structurée et nettoyée du code initialement présent dans le notebook `GCN_BASICS.ipynb`.

## But du Code

Le but principal de ce dépôt est l'implémentation et la comparaison de systèmes de recommandation basés sur l'apprentissage profond sur graphes (Graph Convolutional Networks - GCN) classiques et leurs extensions quantiques (Quantum Machine Learning). 

Le projet met en place deux concepts clés :
1. **L'apprentissage statique et temporel (Time-Decay)** : Utilisation d'une matrice d'adjacence avec extinction temporelle (*decay temporelle*) pour modéliser le fait que les interactions utilisateur-item récentes sont plus importantes que les interactions anciennes.
2. **L'hybridation Quantique-Classique** : Conception d'un modèle qui utilise la puissance des GCN classiques pour capter la topologie spatiale des interactions et utiliser un scoreur quantique (utilisant les portes d'intrication et de superposition) pour corriger et modéliser des interactions non-linéaires extraordinairement complexes.

---

## Les Modèles et leurs Références

### 1. Modèles Classiques (GCN)
Les algorithmes d'apprentissage sur graphes exploitent la propagation de l'information entre les utilisateurs et les films (Items).

- **`GCNRecommender` (via PyTorch Geometric)** : L'implémentation standard.
  - *Sujet* : Utilise l'opérateur classique GCNConv.
  - *Référence papier* : [Semi-Supervised Classification with Graph Convolutional Networks (Kipf & Welling, ICLR 2017)](https://arxiv.org/abs/1609.02907).
- **`GCNRecommenderFromScratch`** : L'implémentation classique sans module de haut niveau.
  - *Sujet* : Écrit de zéro (`CustomGCNLayer`) via l'opération de multiplication matricielle éparse classique (`torch.sparse.mm`). Utilisé comme base (backbone) afin d'être complètement transparent lors de sa fusion avec les réseaux de neurones quantiques (QNN).

*Note sur la fonction de perte (Loss) :*
Les modèles bénéficient de l'optimisation BPR de recommandation.
- *Référence papier* : [BPR: Bayesian Personalized Ranking from Implicit Feedback (Rendle et al., UAI 2009)](https://arxiv.org/abs/1205.2618).

### 2. Modèles Quantiques et Hybrides (QML)
Les réseaux neuronaux quantiques (QNN) ou réseaux variationnels quantiques (VQC) introduisent une expressivité différente.

- **`QuantumEnhancedScorer` / `QuantumGCNLayer`** : Module quantique utilisant Qiskit.
  - *Sujet* : Utilise des circuits quantiques paramétrés (PQC) implémentant des rotations (`RY`, `RZ`) pour l'encodage et des portes de contrôle (`CX`) pour l'intrication des features latentes. Intégré via `TorchConnector`.
  - *Référence générale QML* : [Classification with Quantum Neural Networks on Near Term Processors (Farhi & Neven, 2018)](https://arxiv.org/abs/1802.06002).
- **`QuantumHybridRecommender`** : La contribution principale.
  - *Sujet* : Le modèle combine un GCN classique pour apprendre l'architecture graphique macroscopique et projette la représentation générée vers un espace quantique à qubits réduits. Ce système prédit l'affinité sous la forme de : 
  `Score = Produit_Scalaire(User, Item) + Β * Correction_Quantique(User, Item)`.
  - *Référence QGCN* : S'inspire des travaux de [Quantum Graph Neural Networks (Verdon et al., 2019)](https://arxiv.org/abs/1909.12264).

---

## Structure du Système

- `src/data_loader.py` : Importe les données (MovieLens), encode les *ID*, effectue la séparation stricte chronologiquement (Train/Val/Test) et génère les graphes pondérés (Statiques et Temporels avec *decay* pondéré).
- `src/models.py` : Catalogue abstrait répertoriant nos classes PyTorch pour la définition précise des algorithmes cités ci-dessus (PyG, From Scratch et Quantum Qiskit).
- `src/evaluation.py` : Logique d'évaluation hors ligne de Recommender Systems (calculs de l'erreur MSE/RMSE, métrique d'ordre AUC, métriques top-K comme Recall@K et NDCG@K).
- `src/train.py` : Contient l'algorithme d'entraînement BPR pour les architectures classiques, ainsi que `train_quantum_hybrid` optimisé avec un clip de gradient, `DataLoader` par *batch*, et une gestion très prudente de la perte MSE/BPR pour les modèles quantiques.
- `main.py` : Fichier principal *plug-and-play*. Il connecte le flux de bout en bout de l'organisation des données à l'évaluation comparative des modèles.

## Instructions d'Exécution

Assurez-vous que l'environnement virtuel contient l'ensemble des bibliothèques (`torch`, `torch_geometric`, `qiskit`, `qiskit-machine-learning`, `numpy`, `pandas`, `scipy`, `scikit-learn`).

```bash
# Lancement de l'expérience standard complète (GCN de zero + QGCN Hybride)
python main.py
```
