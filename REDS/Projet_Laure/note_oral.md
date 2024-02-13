# Neural architecture search
## Overview of NAS
* NAS = automatiser le design des architecture neuronale pour une tache donnée 
* Certaine archis trouvé surpasse celle designé par l'humain
* On cherche dans quoi et comment ? 
    * On cherche dans un espace de recheche prédéfinir qui encode les architectures = espace compréhensible par machine
    * Avec une certaine stratégie de recherche
    * Et en estimant les performances pour éviter l’entraînement de chaque archi
* Zoom sur ces 3 boites

## Search space
* Très grand : 10^20
    * Car combinaison de nombre de couche, opération possible, hyparamètre
* on peut restriction avec connaissance préalable
* Contrepartie :  réduit les chance de trouver une architecture vraiment innovante
* Classiquement un graph d'opération

## Type de Search Space
* Plusieurs type de search space
* Graph
* Cells based : Assemblage de bloc = resnet/vgg
* Naturellement, cedxrtaine technique ne rentre dans aucunes des cases 

## Search strategy
* Stratégie de recherche 
* Deux grandes categories
    * Black box :  algo qu'on connaît, technique classique : RL, algo génétique ect
    * One shot: 
        * HyperNetwork : réseau qui génère les poids pour d'autre modèle
        * Supernetwork : Archi avec toutes les opération possible qu'on vient découper
            * Une opération entre deux noeuds -> beaucoup de combinaison d'archi possible 
        * Parfois ça marche, parfois non 

## Taxonomy One shot
* Naturellement, taxonomy
* Inclus la figure d'un supernetwork parce que c'est assez fou

## Performance estimation strategy
* Plein technique éviter entraînement complet avec un prédicteur
* 
* Learning curve extrapolation
    * L'image parle d'elle même, 
* Zero cost proxy : une forward / backward + croise les doigts corrélation
    * Fonctionne dans certain cas
    * Souvent combiné avec une autre technique d'estimation et/ou de recherche
* Subset selection : 
    * Sub set de donnée, généré artificiellement ou non 
* Meta Learning 
    * Prédicteur de performance, surrogate model
    * Prends archi et dataset encodé pour prédire la performance 

## Benchmark
* Benchmark : un search space où chaque archi est étiqueté avec son accuracy à plusieurs époch
* Fort car évite l’entraînement à chaque fois 
* A une epoch les gens entraîné l'archi final sur 600 epoch, qu'importe si les 200 dernière epoch faisait rien 
* Permet test les algo de recherche de manière standardisé, reproductive et avec des test statistiques
* Plein de benchmark différent et spécialisé par tache ou non 

## Evaluation tableau 1 
* Liste de méthode d'NAS 
* Accuracy de la meilleurs architecture trouvé
* Par exemple ici c'est sur 3 run de recherche d'architecture
* On remarque le nombre d'architecture entraîné plus où moins grand

## Evaluation tableau 2 
* Autre tableau 
* Cette fois on voit aussi le nombre de paramètre et le coût GPU 

# Contribution
