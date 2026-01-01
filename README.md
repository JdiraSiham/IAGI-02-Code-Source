Ce dossier regroupe l’ensemble des fichiers développés dans le cadre de ce projet, dont l’objectif principal est la **modélisation**, la **résolution** et l’**optimisation** d’un problème de **programmation linéaire** à l’aide du langage **Python**.

---

## 1. Contenu du dossier

### main.py
Ce fichier contient le cœur du projet.  
Il permet :
- de résoudre le problème de programmation linéaire ;
- d’effectuer l’analyse post-optimisation ;
- d’effectuer l’analyse du goulot d’étranglement.

### test.py
Ce fichier est destiné aux tests.  
Il permet de :
- vérifier le bon fonctionnement des fonctions de résolution ;
- valider les résultats obtenus par le programme principal.

### interface.py
Ce fichier contient l’interface graphique.  
L’interface permet de :
- modifier les paramètres du problème de programmation linéaire ;
- visualiser l’impact de chaque paramètre sur la solution.

### requirement.txt
Ce fichier contient la liste des bibliothèques Python nécessaires pour exécuter le projet.  
Il permet d’installer facilement toutes les dépendances avec la commande :
```bash
pip install -r requirement.txt
```

---

## 2. Exécution du projet

### 1. Installer les dépendances :
```bash
pip install -r requirement.txt
```
### 2. Lancer la résolution et l’optimisation :
```bash
python main.py
```

### 3. Lancer les tests :
```bash
python test.py
```

### 4. Lancer l’interface graphique :
```bash
streamlit run interface.py
```
