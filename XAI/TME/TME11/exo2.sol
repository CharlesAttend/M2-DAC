% rules
cnf(a,axiom,[-a(X,grippe),fievre(X)]).
cnf(b,axiom,[-a(X,angine),fievre(X)]).
cnf(c,axiom,[-a(X,bronchite),fievre(X)]).
cnf(d,axiom,[-a(X,simplerhume),mieux(X)]).
cnf(e,axiom,[-a(X,angine),-antibio(X),mieux(X)]).
cnf(f,axiom,[-a(X,bronchite),-antibio(X),mieux(X)]).
cnf(g,axiom,[-a(X,bronchite),toux(X)]).
cnf(h,axiom,[-a(X,grippe),-a(X,angine),-a(X,bronchite),-a(X,simplerhume)]).
cnf(i,axiom,[-a(X,Y),diff(Y,simplerhume),a(X,simplerhume)]).
% facts
cnf(j,axiom,[-toux(p)]).
cnf(k,axiom,[antibio(p)]).


