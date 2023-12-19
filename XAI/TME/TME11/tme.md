# Abduction de théories clausales

Voici un exemple de ce qui est demandé.

## Exercice 1 Qui vient à la soirée ?
Une soirée a lieu et l’on cherche à savoir qui vient ou non. On sait que quelqu’un vient s’il est libre et motivé pour venir. Quelqu’un est libre si et seulement si il n’a pas d’autre rendez-vous et qu’il ne travaille pas. On considère qu’une personne qui s’ennuie et qui n’a pas d’autre rendez-vous est motivée. De mˆeme, une personne est motivée pour venir si au moins deux de ses amis viennent. Par contre, une personne n’est pas motivée pour venir si une personne qu’elle déteste vient.

1. Représentation. Traduire ces phrases en une théorie clausale T . On utilise les prédicats :
- vient(X) : la personne X vient à la soirée.
- libre(X) : la personne X est libre.
- travail(X) : la personne X a du travail.
- autreRdv(X) : la personne X un autre rendez-vous.
- motive(X) : la personne X est motivée pour venir.
- ennui(X) : X s’ennuie.
- ami(X,Y) : X considère Y comme son ami (non forcément symétrique).
- deteste(X,Y) : X déteste Y.
- eq(X,Y) : X et Y désignent la même personne.

```
cnf(t,axiom,[vient(X),-libre(X),-motive(X)]).
cnf(t,axiom,[libre(X),autreRdv(X),travail(X)]).
cnf(t,axiom,[-libre(X),-autreRdv(X)]).
cnf(t,axiom,[-libre(X),-travail(X)]).
cnf(t,axiom,[motive(X),-ennui(X),autreRdv(X)]).
cnf(t,axiom,[motive(X),-vient(Y),-vient(Z),-ami(X,Y),-ami(X,Z),eq(Y,Z)]).
cnf(t,axiom,[-motive(X),-vient(Y),-deteste(X,Y)]).
```

2. Déduction. On considère les faits suivants :
Alban et Claire sont amis l’un avec l’autre. Alban, Claire et Didier considèrent Brian comme un ami. Didier et Elsa considère Alban comme un ami, et Elsa considère aussi Claire comme une amie. Brian est motivé pour venir et libre. Claire et Didier sont aussi libres. Alban ne travaille pas et n’a pas d’autres rendez-vous. par contre Elsa travaille. Enfin, Claire s’ennuie.

(a) Traduire ces phrases en un ensemble de faits (clauses unitaires) F1. On utilise les constantes a,b,c,d,e pour désigner respectivement Alban, Brian, Claire, Didier et Elsa.

(c) On considère en plus la phrase Claire déteste Didier. et on note F2 = F1 ∪ {deteste(c,d).}. Que vaut Carc(T ∪ F2,Pv) ? Que peut-on en conclure ?

```
cnf(f,top_clause,[eq(X,X)]).
cnf(f,top_clause,[-eq(a,b)]).
cnf(f,top_clause,[-eq(a,c)]).
cnf(f,top_clause,[-eq(a,d)]).
cnf(f,top_clause,[-eq(a,e)]).
cnf(f,top_clause,[-eq(b,a)]).
cnf(f,top_clause,[-eq(b,c)]).
cnf(f,top_clause,[-eq(b,d)]).
cnf(f,top_clause,[-eq(b,e)]).
cnf(f,top_clause,[-eq(c,a)]).
cnf(f,top_clause,[-eq(c,b)]).
cnf(f,top_clause,[-eq(c,d)]).
cnf(f,top_clause,[-eq(c,e)]).
cnf(f,top_clause,[-eq(d,a)]).
cnf(f,top_clause,[-eq(d,b)]).
cnf(f,top_clause,[-eq(d,c)]).
cnf(f,top_clause,[-eq(d,e)]).
cnf(f,top_clause,[-eq(e,a)]).
cnf(f,top_clause,[-eq(e,b)]).
cnf(f,top_clause,[-eq(e,c)]).
cnf(f,top_clause,[-eq(e,d)]).
%cnf(f,top_clause,[deteste(c,d)]).
```

Maintenant, il faut implémenter de la même manière l'exercice suivant.

## Exercice 2 Diagnostic
On considère un problème simple de diqgnostic médical en logique des prédicats. On utilise les prédicats suivants :
- a(X,Y ) signifie qu’une personne X a la maladie Y .
- fievre(X) signifie qu’une personne X a de la fièvre.
- toux(X) signifie qu’une personne X a de la toux.
- antibio(X) signifie qu’une personne X prend des antibiotiques.
- mieux(X) signifie qu’une personne X va mieux le lendemain.
- diff(X,Y ) signifie que les maladies X et Y sont différentes.

On utiliser de plus les constantes grippe, bronchite, angine, simplerhume pour les maladies, ainsi que la constante p pour représenter le patient considéré.

1. Traduire chacune des phrases suivantes en une règle ou un fait.
(a) Quand une personne a une grippe, elle a de la fièvre.
(b) Quand une personne a une angine, elle a de la fièvre.
(c) Quand une personne a une bronchite, elle a de la fièvre.
(d) Quand une personne a une simple rhume, elle va mieux le lendemain.
(e) Quand une personne a une angine et prend des antibiotiques, elle va mieux le lendemain.
(f) Quand une personne a une bronchite et prend des antibiotiques, elle va mieux le lendemain.
(g) Quand une personne a une bronchite, elle a de la toux.
(h) La grippe, l’angine et la bronchite ne sont pas de simples rhumes.
(i) Quand une personne a un simple rhume, c’est qu’elle n’a pas d’autres maladies. (note : utiliser diff(Y ,simplerhume)).
(j) Le patient p ne tousse pas.
(k) Le patient p prend des antibiotiques.

2. Traduire ces règles et faits en une théorie clausale Σ en nommant chaque clause.
3. Utiliser la méthode de résolution inversée pour donner toutes les hypothèses expliquant l’observation O1 : fievre(p). Détailler les étapes en  récisant les conséquences que vous calculez (et lesquelles sont éliminés et pourquoi).
4. Faire de mˆeme pour l’observation O2 : mieux(p).
5. Donner la ou les hypothèses permettant d’expliquer O1 ∧ O2 en justifiant formellement votre raisonnement. On peut utiliser les résultats précédent en notant que si D1 est une conséquence de Σ ∪ {C1} et que D2 est une conséquence de Σ ∪ {C2}, alors D1 ∨ D1 est une conséquence de Σ ∪ {C1 ∨ C2}.