# Projet Abalone - INF8175 - Automne 2023

**Nom de l’équipe _à trouver_**

Nom 1 - Matricule 1  
Nom 2 - Matricule 2

**Sommaire :**
<!-- TOC -->
* [Projet Abalone - INF8175 - Automne 2023](#projet-abalone---inf8175---automne-2023)
  * [I. Méthodologie](#i-méthodologie)
    * [A. Algorithme de recherche](#a-algorithme-de-recherche)
    * [B. Fonction heuristique](#b-fonction-heuristique)
  * [II. Évolution](#ii-évolution)
  * [III. Résultats](#iii-résultats)
  * [IV. Discussion et conclusion](#iv-discussion-et-conclusion)
  * [V. Annexes](#v-annexes)
    * [Inspirations](#inspirations)
    * [Bugs](#bugs)
<!-- TOC -->


## I. Méthodologie

### A. Algorithme de recherche

**Fait :**
- Min Max
- Alpha Beta pruning
- Ajout de la profondeur max

**Idée d’évolutions :**
- Trier les actions. Au lieu de prendre les actions comme elles viennent dans la liste (aléatoire), 
on pourrait créer une fonction d'évaluation (plus rapide que la fonction heuristique) qui mettrait en premier les 
actions les plus susceptible d'être intéressantes, par exemple un coup qui pousse une boule adverse. Cela pourrait 
augmenter grandement les performances de la recherche et permettrait donc de regarder plus profond.

  - 1ère implémentation: On regarde si il y a une différence de score et on étudie en priorité les actions qui poussent 
    des billes adverses dehors et en dernier les actions ou l'on pousse nos propres billes dehors.  
    Résultat : Avec une profondeur de 4 sur les 2 premières actions, on passe de **~50/55sec** ~200/250k noeuds explorés et ~7/8k 
    pruning à **~35/40sec** ~150/200k noeuds explorés et ~5k pruning


- On pourrait créer des tests de performances. J'ai commencé en affichant le nombre d'actions analysées à chaque coup 
joué (je ne suis pas tout à fait sûr de l'implémentation d'ailleurs). Le but serait de vérifier si le pruning permet de 
visiter moins de coups, pareil pour le tri des actions. Ça simplifierait la détection de bugs et aiderait à l'optimisation.


- Essayer d'éviter (ou mitiger) "l'effet de myopie" (slide 40, module 2), c-à-d évaluer une position comme positive alors 
qu'au coup suivant on perd l'avantage.  
Pour cela il ne faudrait arrêter la recherche que sur des états "stable", par exemple lorsque que l'on ne pousser 
aucune bille dehors.


- Pour gagner grandement en performance, il faudrait se pencher sur le problème des chemins redondant menant aux mêmes coups.
Dans le cours le prof mentionne les tables de transpositions (slide 33, module 2)
  - 1ère implémentation: table de transposition normal qui assign un random key à chaque neoud puis le skock dans une table. Ralenti les performances parce que le system doit parcourit la table a chaque fois pour vérifier si le noeud y es présent. Rajoute des secondes de traitement.
  - A faire: zobrist table with maybe a transposition table



- Réfléchir à la manière de distribuer le temps pour chaque coups. Peut être regarder du coté de l'iterative deepening search.
Je crois que le prof en a parlé, ça consiste à d'abord regarder à une profondeur de 1 puis de recommencer à une profondeur de 2, etc etc.  
L'intérêt est que si l'on fixe une limite de temps à l'algo, il peut s'arrêter à tout instant sans en garantissant une 
exploration complète. Pour améliorer ce processus, on peut regarder en priorité les meilleurs coups trouvés par la recherche 
précédente.
- 1ère implémentation: Allocation de 18s pour chaque coup


- Search extension. Le principe est de passer plus de temps (regarder plus profond) si un coup nous paraît prometteur.  
À voir les critères que l'on peut choisir pour trouver un coup prometteur, par exemple un coup on pousse une bille 
adverse sur le bord.


### B. Fonction heuristique

**Fait :**
- **On fournit son score à l'agent**.
- **On fournit son score moins celui de l'adversaire**. Il peut prendre en compte la situation de l'adversaire qu'il ignorait jusq'à présent.
- **On fournit le résultat de la partie si elle s'arrêtait la.** On prend donc en compte la position des billes qui 
doivent être proche du centre en cas d'égalité de score. C'est pour l'instant assez efficace car l'agent n'est pas 
agressif. L'adversaire étant assez passif aussi, la plupart des parties se finissent avec un score nul.

**Idée d’évolutions :**
- On peut faire en sorte que les fonctions heuristiques donnent un score en + et - l'infini (représentant respective la 
victoire et la défaite).  
Ça simplifiera les fonctions heuristiques, car elles pourront renvoyer n'importe quelle valeur (réelle).


- Il faut inciter l'agent à placer ses billes au centre et aussi (et surtout) à pousser celles de l'adversaire sur les bords.
Pour cela, il faudrait donner un score plus élevé à nos billes proche du centre et un score négatif pour nos billes proche du bord.
Il faut aussi faire l'inverse pour les billes de l'adversaire.  
Les valeurs peuvent plus ou moins balancé afin d'inciter plus à l'attaque ou à la défence.

  - Attention à bien pondérer par le nombre de pièces ou avec le score parce que sinon le modèle peut suicider des billes
    pour améliorer son score.
  - 1ère implémentation: on a créé une table de correspondance qui pour chaque pièce donne un score en fonction de sa 
    distance au centre et de son propriétaire.


- On pourrait aussi donner des bonus s’il met ses billes en ligne de 2 ou 3. On pourrait aussi donner un malus si 
jamais il a des billes isolées.  
Si on fait l'inverse avec l'adversaire ça poussera notre agent à isoler les billes adverses.


- Le plus dur sera surement de trouver les bonnes valeur de bonus et malus pour que notre agent soit vraiment efficace.  
Par exemple si on recompense trop les billes isolé par rapport aux billes qu'il ferait tomber il pourrait essayer de 
simplement isoler les billes adverse sans les pousser au bord. Il faudra donc faire attention à bien équilibrer les 
valeurs des recompenses.


- Il faudra aussi faire en sorte que le calcul de la fonction heuristique soit le plus rapide possible. Globalement, 
chaque optimisation aura un gros impact sur les performances de l'agent, car on pourra aller plus profond.

- Parler de pourquoi on a utiisé une seule fonction de recherche au lieu de 2 (min et max)

## II. Évolution

**Fait :**

**Idée d’évolutions :**

## III. Résultats

## IV. Discussion et conclusion

## V. Annexes

### Inspirations

Ces 2 vidéos sont super intéressante et même si ça parle d'échecs, beaucoup de problèmes sont très similaires.
- [Coding Adventure: Chess](https://www.youtube.com/watch?v=U4ogK0MIzqk&t=3s&ab_channel=SebastianLague)
- [Coding Adventure: Making a Better Chess Bot](https://www.youtube.com/watch?v=_vqlIPDR2TU&ab_channel=SebastianLague)

### Bugs

- Parfois, le programme prend beaucoup de RAM (jusqu'a 31Go), trouver pourquoi et corriger.