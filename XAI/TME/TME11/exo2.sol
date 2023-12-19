% rules
cnf(t,axiom,[-a(X,grippe),fievre(X)]).
cnf(t,axiom,[-a(X,angine),fievre(X)]).
cnf(t,axiom,[-a(X,bronchite),fievre(X)]).
cnf(t,axiom,[-a(X,simplerhume),mieux(X)]).
cnf(t,axiom,[-a(X,angine),-antibio(X),mieux(X)]).
cnf(t,axiom,[-a(X,bronchite),-antibio(X),mieux(X)]).
cnf(t,axiom,[-a(X,bronchite),toux(X)]).
cnf(t,axiom,[a(X,grippe),diff(grippe,simplerhume)]).
cnf(t,axiom,[a(X,angine),diff(angine,simplerhume)]).
cnf(t,axiom,[a(X,bronchite),diff(bronchite,simplerhume)]).
cnf(t,axiom,[-a(X,Y),diff(Y,simplerhume),a(X,simplerhume)]).
% facts
cnf(f,axiom,[toux(p)]).
cnf(f,axiom,[antibio(p)]).


