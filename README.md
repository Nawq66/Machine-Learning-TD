# Projet Machine Learning - Prévision de Consommation Électrique

Ce projet analyse et prévoit la consommation électrique en France en utilisant différentes approches de machine learning.

## Structure du Projet

- `partie4.py` : Implémentation des modèles de régression pour la prévision de consommation
- `requirements.txt` : Liste des dépendances Python
- `eco2mix_cleaned.csv` : Données de consommation électrique (à ajouter manuellement)

## Installation

1. Cloner le repository :
```bash
git clone [URL_DU_REPO]
cd [NOM_DU_DOSSIER]
```

2. Créer un environnement virtuel :
```bash
python -m venv venv
venv\Scripts\activate  # Sur Windows
```

3. Installer les dépendances :
```bash
pip install -r requirements.txt
```

## Utilisation

1. Placer le fichier `eco2mix_cleaned.csv` dans le dossier du projet
2. Exécuter le script :
```bash
python partie4.py
```

## Fonctionnalités

- Préparation et nettoyage des données
- Feature engineering avancé
- Implémentation de plusieurs modèles de machine learning
- Visualisation des résultats
- Analyse des performances

## Modèles Implémentés

- Régression Linéaire
- Ridge Regression
- Lasso Regression
- Arbre de Décision
- Random Forest
- XGBoost 