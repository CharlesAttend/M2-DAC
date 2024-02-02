## Exercice 1 : FACE : cours 1 page 36
### Question 1
* $\mathcal{E}$ l'ensemble des données $e$ tel que la prediction par $f$ n'est pas égale à celle de $x$ et dont la densité (fréquence) de la même classe est supérieur à $\delta$ = une donnée de l'autre classe mais pas loin de plein de point de la même classe
* $e^*$ l’explication 
    * qui minimise sa distance avec $x$ == proche de $x$
    * et le nombre de dimension qui on changé (norme 0) par un facteur $\lambda$ == parcimonie
* == explication contre-factuelle de $x$ 
* Par exemple un point 

### Question 2
Ici on fait l'hypothèse qu'on a accès aux données et à leurs prédictions. Ainsi qu'au modèle pour tester la classe de $e$

### Question 3
Diapo cours 1 p36

## Exercice 2

### Question 1
Comprendre le graphique du cours 3 pages 26. Si la frontière de décision est non linéaire, en particulier plus ou moins loin du point à expliquer. Le paramètre vas permettre de plus ou moins prendre en compte le comportement non linéaire plus ou moins loins. *C'est complexe à expliquer*

### Question 2
Pour moi on peut utiliser n'importe quel kernel ou mesure de similarité. Dans LIME c'est un kernel gaussien qui est utilisé, je dirai donc que c'est ma préféré (j'ai aucun argument)

## Exercice 3
### Question 1
Catégorie:
* Couleur
* Crit'Air
* Diesel

One Hot : $y_i \in \mathbb{R}^7$ i-ème exemple, 
* 3 première colonne : bleue rouge vert
* 4 et 5 ème colonne : Crit'Air 2 et Crit'Air 3 ou +
* 6 ème colonne : Diesel
* 7ème colonne: Age

On peut réduire (???) en disant qu'une voiture appartiens obligatoirement à une unique catégorie 
* 3 première colonne : bleue rouge (vert ?)
* 4 et 5 ème colonne : Crit'Air 2
* 6 ème colonne : Diesel
* 7ème colonne: Age

### Question 2
L'approche est dans les grande lignes proche de LIME, 

* On créé une base d’entraînement dans le voisinage de x dont la classification par f couvre les deux classes
    * Utilisation d'un algo génétique 
    * Croisement d'individu (tirage de deux attributs pour le croisement) & Mutation d'individu (remplacement d'un attribut par une valeur random en suivant sa distribution dans l'ensemble de test)
* On apprend un modèle de substitution $g$, ici un arbre de décision.
* L'interprétation de $g$ nous donne à la fois une explication pour $x$ (en utilisant les branches de $g(x)$) et une explication contrefactuelle pour les branche qui ne servent pas pour $g(x)$. 

### Question 3
* Branche associé à la classe $g(x_1)$ : Bleu -> 3+ -> non diesel -> 1
* Exemple contrefactuel non associé à la classe oposé dans l'arbre (-1) : 
    * Rouge
    * Vert, age $\leq 8$
    * Bleu, 3+, Diesel
* Explication : il faut trouver la règle $r_{best}$ qui minimise le nombre de test invalisé par $x_0$ == le contre exemple le plus proche de $x_0$
    * Bleu, 3+, Diesel

## Exercice 4
### Question 1
* a = aéroport, d = durée, p = prix
* $u_1$ veut $a = Orly, d < 120, p \leq 200$
* -> $v_2$ vol rapide

### Question 2
* Formalisation:
    * Orly = Orly
    * Environ 2h = $d \in [0;125]$
    * ~200€ = $[0,210]$ 
    * Si plus rapide ou moins chère == mieux
* => $v_1, v_2$

### Question 3
* $e(130) = 1200, e(160) = 1050, e(170) = 1200$
* La consommation se trouvera entre $1050$ et $1200$ kilo de CO2 par personne

### Question 4
En utilisant uniquement de la logique classique, il faut fallut modéliser par une fonction mathématique le lien entre le nombre personnes et la quantité de jus de tomate. En logique flou exprimant cela avec de la logique flou, le résultat est serte moins précis mais suffisant pour être interprétable. Dans notre exemple, le vol aura besoin "d'un peu plus de 8 litre de soupe". La logique flou apporte des explications plus proche de la manière dont on (l'humain) raisonne, mais au coût d'une perte de précision.