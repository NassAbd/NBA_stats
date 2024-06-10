# Analyse et Prédiction des Performances des Lakers

Analyse des performances historiques des Lakers et prédiction des performances futures.

Source du dataset : [TeamYearByYearStats](https://github.com/swar/nba_api/blob/master/docs/nba_api/stats/endpoints/teamyearbyyearstats.md)


## Objectifs

Explorer les éléments les plus déterminants dans le succès ou l'échec des Los Angeles Lakers au cours de leur histoire.

## Plan d'analyse :

### Définir les Indicateurs de Succès

Par exemple :

- Victoires et défaites par saison.
- Performances en playoffs.
- Statistiques offensives et défensives (points marqués, points encaissés).
- Taux de victoire.
- Classements de conférence et de division.
- Apparitions en finales NBA.

### Analyser des Données

L'analyse des données comprend les étapes suivantes :

- **Performances en Saison Régulière et Playoffs :** Étudier les performances des Lakers en saison régulière et en playoffs pour identifier les tendances et les facteurs de succès.
- **Évolution des Statistiques de Jeu :** Analyser l'évolution des statistiques offensives et défensives (points marqués, rebonds, etc.) au fil du temps.

### Prédiction sur les Performances Futures

Utilisation de plusieurs modèles de machine learning pour prédire les performances futures des Lakers, notamment :

- **Arbre de Décision**
- **Forêt Aléatoire** 
- **K Plus Proches Voisins**
- **Régression Linéaire** 

### Visualisation des Données

Des visualisations ont été créées pour illustrer les résultats de cette analyse :

- Graphiques linéaires montrant l'évolution des victoires, des défaites et du taux de victoire par saison.
- Histogrammes comparant les statistiques offensives et défensives au fil des saisons.

## Captures d'écran

![Graphique Victoires](./images/victoires.png)
![Graphique Défaites](./images/defaites.png)
![Graphique Pourcentage de Victoires](./images/pourcentage_victoires.png)

## License

Ce projet est licencié sous la licence MIT - voir le fichier [LICENSE](LICENSE) pour plus de détails.
