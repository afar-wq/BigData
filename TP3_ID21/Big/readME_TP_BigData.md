# 🐾 Classification d'Images de la Faune Sauvage avec PySpark

Ce projet implémente un pipeline complet et distribué de Machine Learning pour la classification d'images d'animaux sauvages à l'aide d'**Apache Spark (PySpark)** et de **Spark MLlib**, conçu et optimisé pour s'exécuter efficacement sur l'environnement Kaggle.

---

## 📋 Table des Matières
1. [Présentation du Projet](#-présentation-du-projet)
2. [Architecture du Pipeline](#-architecture-du-pipeline)
3. [Prérequis et Dépendances](#-prérequis-et-dépendances)
4. [Structure des Données](#-structure-des-données)
5. [Optimisations Clés](#-optimisations-clés)
6. [Utilisation](#-utilisation)

---

## 🎯 Présentation du Projet

L'objectif de ce notebook est de démontrer comment traiter, transformer et classifier un jeu de données d'images en utilisant une approche Big Data distribuée plutôt que des frameworks classiques (comme PyTorch ou TensorFlow). Le projet couvre l'ensemble du cycle de vie de la donnée : de la restructuration des fichiers bruts jusqu'au déploiement d'un modèle prêt pour l'inférence en production.

---

## 🏗️ Architecture du Pipeline

Le notebook est structuré en **8 étapes séquentielles** :

* **Étape Préliminaire : Préparation et Conversion du Dataset**
  * Conversion de la structure d'origine (format de détection YOLO avec fichiers textes `.txt`) vers un format de classification standard (`ImageFolder`).
  * Utilisation d'un traitement multithreadé (`ThreadPoolExecutor`) pour paralléliser les entrées/sorties disque et accélérer la copie des fichiers.

* **Étape 1 : DATA (Initialisation Spark & Chargement des données)**
  * Configuration d'une `SparkSession` adaptée aux contraintes mémoire de Kaggle (`3g` pour le Driver et l'Executor).
  * Lecture distribuée des images brutes au format binaire (`binaryFile`).
  * Extraction automatique des étiquettes (`label_name`) à partir de l'arborescence des dossiers via expressions régulières.

* **Étape 2 : EDA (Exploratory Data Analysis)**
  * Analyse exploratoire distribuée (`groupBy`, `count`, `avg`, `min`, `max`) pour inspecter la volumétrie, la taille des fichiers et l'équilibre des classes d'animaux.

* **Étape 3 : PRÉTRAITEMENT (Nettoyage & Extraction de caractéristiques)**
  * Application d'une UDF (User Defined Function) Python combinant `PIL` et `NumPy` pour ouvrir, redimensionner les images à $32 \times 32$ pixels et les convertir en vecteurs denses normalisés ($3\,072$ dimensions).
  * Filtrage automatique des éventuels fichiers corrompus.

* **Étape 4 : TRANSFORMATION (Indexation, Normalisation & Cache)**
  * **`StringIndexer`** : Encodage des noms textuels des classes en indices numériques ($0, 1, 2, \dots$).
  * **`StandardScaler`** : Centrage et réduction des vecteurs de pixels pour homogénéiser l'échelle des caractéristiques.
  * **Mise en cache (`.cache()`)** : Conservation des DataFrames transformés en mémoire vive (RAM) pour éviter les recalculs coûteux.

* **Étape 5 : ML (Entraînement du modèle)**
  * Entraînement d'un modèle de classification distribué de type Forêt Aléatoire (`RandomForestClassifier` de Spark MLlib) sur les vecteurs de caractéristiques.

* **Étape 6 : EVAL (Évaluation des performances)**
  * Génération des prédictions sur le jeu de test indépendant.
  * Évaluation des performances globales via `MulticlassClassificationEvaluator` (**Accuracy** et **Score F1**).

* **Étape 7 : DÉPLOYER (Sauvegarde & Inférence de production)**
  * Exportation et sauvegarde complète du `PipelineModel` (prétraitements + modèle entraîné) sur le disque.
  * Rechargement du modèle et simulation d'une inférence en production avec traduction inverse (`IndexToString`) des indices prédits vers les noms d'animaux lisibles.

---

## 🛠️ Prérequis et Dépendances

Le notebook est conçu pour tourner sur **Kaggle Notebooks** (CPU standard), mais peut être exécuté sur tout environnement disposant de :
* **Python** $\ge 3.10$
* **Java (OpenJDK)** $8$ ou $11$ (requis pour la JVM de Spark)
* **PySpark** (`pyspark.sql`, `pyspark.ml`)
* **Pillow (`PIL`)** & **NumPy** (traitement matriciel des images)
* **PyYAML** & **tqdm** (lecture de configuration et barres de progression)

---

## 📁 Structure des Données

Le script attend en entrée un dataset au format YOLO contenant un fichier `data.yaml` et les dossiers `train`, `valid`, `test`. Il génère automatiquement l'arborescence suivante dans `/kaggle/working/` :

```text
wildlife_dataset_classification/
├── train/
│   ├── Classe_A/
│   │   ├── img_001.jpg
│   │   └── ...
│   └── Classe_B/
├── valid/
└── test/
```

---

## ⚡ Optimisations Clés

Pour garantir une exécution fluide sans erreur de mémoire (`OutOfMemoryError: Java heap space`), plusieurs choix d'architecture ont été appliqués :
1. **Résolution d'image à $32 \times 32$ pixels** : Réduit la dimensionnalité de $12\,288$ (en $64 \times 64$) à $3\,072$ caractéristiques par image, allégeant considérablement la charge sur la JVM.
2. **Suppression de la PCA** : Le calcul de la matrice de covariance d'une PCA sur plusieurs milliers de dimensions crée un goulot d'étranglement majeur sur Spark en mode local. Le `RandomForestClassifier` gérant nativement la haute dimensionnalité, la PCA a été retirée au profit d'un pipeline léger (`StringIndexer` + `StandardScaler`).
3. **Parallélisation I/O** : L'utilisation de `ThreadPoolExecutor` réduit drastiquement le temps de préparation des dossiers sur le système de fichiers virtuel de Kaggle.

---

## 🚀 Utilisation

1. Importez le dataset *Wildlife Dataset* dans votre environnement Kaggle.
2. Exécutez les cellules de manière **séquentielle** de l'étape préliminaire jusqu'à l'étape 7.
3. *(Optionnel)* En cas de modification des paramètres de mémoire Spark, pensez à effectuer un **Restart Session** du noyau avant de relancer l'exécution.