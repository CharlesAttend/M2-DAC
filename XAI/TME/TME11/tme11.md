Charles Vin (21216136) & Aymeric Delefosse (21213744) <span style="float:right">DAC 2023-2024</span><br>


# TME 11 - Abduction


## Exercice 1 : _Qui vient à la soirée ?_

### 1. Représentation 

```java
cnf(t,axiom,[vient(X),-libre(X),-motive(X)]).
cnf(t,axiom,[libre(X),autreRdv(X),travail(X)]).
cnf(t,axiom,[-libre(X),-autreRdv(X)]).
cnf(t,axiom,[-libre(X),-travail(X)]).
cnf(t,axiom,[motive(X),-ennui(X),autreRdv(X)]).
cnf(t,axiom,[motive(X),-vient(Y),-vient(Z),-ami(X,Y),-ami(X,Z),eq(Y,Z)]).
cnf(t,axiom,[-motive(X),-vient(Y),-deteste(X,Y)]).
```

### 2. Déduction 

(a)
```java
cnf(f,top_clause,[ami(a,c)]).
cnf(f,top_clause,[ami(c,a)]).
cnf(f,top_clause,[ami(a,b)]).
cnf(f,top_clause,[ami(c,b)]).
cnf(f,top_clause,[ami(d,b)]).
cnf(f,top_clause,[ami(d,a)]).
cnf(f,top_clause,[ami(e,a)]).
cnf(f,top_clause,[ami(e,c)]).
cnf(f,top_clause,[motive(b)]).
cnf(f,top_clause,[libre(b)]).
cnf(f,top_clause,[libre(c)]).
cnf(f,top_clause,[libre(d)]).
cnf(f,top_clause,[-travail(a)]).
cnf(f,top_clause,[-autreRdv(a)]).
cnf(f,top_clause,[travail(e)]).
cnf(f,top_clause,[ennui(c)]).
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
```

(b)

Champ de production :

```java
pv([vient(_),-vient(_)]<=1).
```

Résultats : 

```
4 FOUND CONSEQUENCES
[+vient(c)]
[+vient(b)]
[+vient(a)]
[+vient(d)]
```

Clauses utilisées (`-used`) :

```
USED CLAUSES:
[+ennui(c)]
[-ennui(c), +motive(c), +autreRdv(c)]
[-motive(c), +vient(c), -libre(c)]
[+libre(c)]
[-autreRdv(c), -libre(c)]
[+libre(c)]

[+vient(b)]

USED CLAUSES:
[+libre(b)]
[-libre(b), +vient(b), -motive(b)]
[+motive(b)]

[+vient(a)]

USED CLAUSES:
[-eq(c,b)]
[+eq(c,b), +motive(a), -vient(c), -vient(b), -ami(a,c), -ami(a,b)]
[-motive(a), +vient(a), -libre(a)]
[+libre(a), +autreRdv(a), +travail(a)]
[-autreRdv(a)]
[-travail(a)]
[+vient(c), -motive(c), -libre(c)]
[+motive(c), -ennui(c), +autreRdv(c)]
[+ennui(c)]
[-autreRdv(c), -libre(c)]
[+libre(c)]
[+libre(c)]
[+vient(b), -motive(b), -libre(b)]
[+motive(b)]
[+libre(b)]
[+ami(a,c)]
[+ami(a,b)]

[+vient(d)]

USED CLAUSES:
[-eq(c,b)]
[+eq(c,b), +motive(a), -vient(c), -vient(b), -ami(a,c), -ami(a,b)]
[-motive(a), +vient(a), -libre(a)]
[-vient(a), +motive(d), -vient(b), -ami(d,a), -ami(d,b), +eq(a,b)]
[-motive(d), +vient(d), -libre(d)]
[+libre(d)]
[+vient(b), -motive(b), -libre(b)]
[+motive(b)]
[+libre(b)]
[+ami(d,a)]
[+ami(d,b)]
[-eq(a,b)]
[+libre(a), +autreRdv(a), +travail(a)]
[-autreRdv(a)]
[-travail(a)]
[+vient(c), -motive(c), -libre(c)]
[+motive(c), -ennui(c), +autreRdv(c)]
[+ennui(c)]
[-autreRdv(c), -libre(c)]
[+libre(c)]
[+libre(c)]
[+vient(b), -motive(b), -libre(b)]
[+motive(b)]
[+libre(b)]
[+ami(a,c)]
[+ami(a,b)]
```

Arbre de SOL-résolution : (`-proof`)

```
PROOF:
(0/10 nodes)
root @1
 + +ennui(c) 1st (ext) @2
   + -ennui(c) (ext target)
   + +motive(c) 1st (ext) @3
   | + -motive(c) (ext target)
   | + +vient(c) 1st (skip) @4
   | + -libre(c) 1st (unit axiom) @5
   + +autreRdv(c) 1st (ext) @6
     + -autreRdv(c) (ext target)
     + -libre(c) 1st (unit axiom) <subgoal> @7

[+vient(b)]

PROOF:
(0/5 nodes)
root @1
 + +libre(b) 1st (ext) @2
   + -libre(b) (ext target)
   + +vient(b) 1st (skip) @3
   + -motive(b) 1st (unit axiom) <subgoal> @4

[+vient(a)]

PROOF:
(0/26 nodes)
root @1
 + -eq(c,b) 1st (ext) @2
   + +eq(c,b) (ext target)
   + +motive(a) 1st (ext) @3
   | + -motive(a) (ext target)
   | + +vient(a) 1st (skip) @4
   | + -libre(a) 1st (ext) @5
   |   + +libre(a) (ext target)
   |   + +autreRdv(a) 1st (ext) @6
   |   | + -autreRdv(a) (ext target)
   |   + +travail(a) 1st (unit axiom) @7
   + -vient(c) 1st (ext) @9
   | + +vient(c) (ext target)
   | + -motive(c) 1st (ext) @10
   | | + +motive(c) (ext target)
   | | + -ennui(c) 1st (unit axiom) @11
   | | + +autreRdv(c) 1st (ext) @12
   | |   + -autreRdv(c) (ext target)
   | |   + -libre(c) 1st (unit axiom) @13
   | + -libre(c) 1st (unit axiom) @14
   + -vient(b) 1st (ext) @16
   | + +vient(b) (ext target)
   | + -motive(b) 1st (unit axiom) @17
   | + -libre(b) 1st (unit axiom) @18
   + -ami(a,c) 1st (unit axiom) @19
   + -ami(a,b) 1st (unit axiom) <subgoal> @20

[+vient(d)]

PROOF:
(0/40 nodes)
root @1
 + -eq(c,b) 1st (ext) @2
   + +eq(c,b) (ext target)
   + +motive(a) 1st (ext) @3
   | + -motive(a) (ext target)
   | + +vient(a) 1st (ext) @4
   | | + -vient(a) (ext target)
   | | + +motive(d) 1st (ext) @5
   | | | + -motive(d) (ext target)
   | | | + +vient(d) 1st (skip) @6
   | | | + -libre(d) 1st (ext) @7
   | | |   + +libre(d) (ext target)
   | | + -vient(b) 1st (ext) @8
   | | | + +vient(b) (ext target)
   | | | + -motive(b) 1st (ext) @9
   | | | | + +motive(b) (ext target)
   | | | + -libre(b) 1st (unit axiom) @10
   | | + -ami(d,a) 1st (ext) @11
   | | | + +ami(d,a) (ext target)
   | | + -ami(d,b) 1st (unit axiom) @12
   | | + +eq(a,b) 1st (unit axiom) @13
   | + -libre(a) 1st (ext) @15
   |   + +libre(a) (ext target)
   |   + +autreRdv(a) 1st (unit axiom) @16
   |   + +travail(a) 1st (unit axiom) @17
   + -vient(c) 1st (ext) @19
   | + +vient(c) (ext target)
   | + -motive(c) 1st (ext) @20
   | | + +motive(c) (ext target)
   | | + -ennui(c) 1st (unit axiom) @21
   | | + +autreRdv(c) 1st (ext) @22
   | |   + -autreRdv(c) (ext target)
   | |   + -libre(c) 1st (unit axiom) @23
   | + -libre(c) 1st (unit axiom) @24
   + -vient(b) 1st (ext) @26
   | + +vient(b) (ext target)
   | + -motive(b) 1st (unit axiom) @27
   | + -libre(b) 1st (unit axiom) @28
   + -ami(a,c) 1st (unit axiom) @29
   + -ami(a,b) 1st (unit axiom) <subgoal> @30
```

(c)

```
cnf(f,top_clause,[deteste(c,d)]).
INSATISFIABLE
```

$Carc(T \cup F_2 \cup F_3) = \{ \emptyset \}$$

La nouvelle top clause rend la théorie fausse : Claire déteste Didier $\Rightarrow$ Claire ne vient pas si Didier vient.





## Exercice 2 - _Diagnostic_

### 1.

(a) $\forall \,$ $X$, a($X$, grippe) $\Rightarrow$ fievre($X$) <br>
(b) $\forall \,$ $X$, a($X$, angine) $\Rightarrow$ fievre($X$)<br>
(c) $\forall \,$ $X$, a($X$, bronchite) $\Rightarrow$ fievre($X$)<br>
(d) $\forall \,$ $X$, a($X$, simplerhume) $\Rightarrow$ mieux($X$)<br>
(e) $\forall \,$ $X$, (a($X$, angine) $\land$ antibio($X$)) $\Rightarrow$ mieux($X$)<br>
(f) $\forall \,$ $X$, (a($X$, bronchite) $\land$ antibio($X$)) $\Rightarrow$ mieux($X$)<br>
(g) $\forall \,$ $X$, a($X$, bronchite) $\Rightarrow$ toux($X$)<br>
(h) $\forall \,$ $X$, a($X$, grippe) $\land$ a($X$, angine) $\land$ a($X$, bronchite) $\Rightarrow$ $\lnot$ a($X$, simplerhume)<br>
(i) $\forall \,$ $X$, a($X$, simplerhume) $\Rightarrow$ $\lnot$ $\exists$ $Y$, (a($X$, $Y$) $\land$ diff($Y$, simplerhume))<br>
(j) $\lnot$ toux(p)<br>
(k) antibio(p)<br>


### 2.

```
cnf(a,axiom,[-a(p,grippe),fievre(p)]).
cnf(b,axiom,[-a(p,angine),fievre(p)]).
cnf(c,axiom,[-a(p,bronchite),fievre(p)]).
cnf(d,axiom,[-a(p,simplerhume),mieux(p)]).
cnf(e,axiom,[-a(p,angine),-antibio(p),mieux(p)]).
cnf(f,axiom,[-a(p,bronchite),-antibio(p),mieux(p)]).
cnf(g,axiom,[-a(p,bronchite),toux(p)]).
cnf(h,axiom,[a(p,grippe),diff(grippe,simplerhume)]).
cnf(h,axiom,[a(p,angine),diff(angine,simplerhume)]).
cnf(h,axiom,[a(p,bronchite),diff(bronchite,simplerhume)]).
cnf(i,axiom,[-a(p,Y),diff(Y,simplerhume),a(p,simplerhume)]).
cnf(j,top_clause,[-toux(p)]).
cnf(k,top_clause,[antibio(p)]).
```

### 3.

