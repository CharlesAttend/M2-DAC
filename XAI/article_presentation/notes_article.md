---
marp: true
header: 
# footer: Hypothesis-driven Decision Support using Evaluative AI
paginate: true
---

# <!-- fit --> Explainable AI is Dead, Long Live Explainable AI!
## <!-- fit --> Hypothesis-driven Decision Support using Evaluative AI

---
# <!-- fit -->  Introduction
<!-- Y'a vraiment une idée de decision support dans le papier je crois  -->
<!-- Tim miller a l'air de peser un peu, il publie beaucoup mais sur des sujets assez varié  -->
---
<!-- header: Introduction -->

# What makes a good decisions
* Simply : Identify, compare, choose options
* Less simply : The 10 'cardinal decision issue' outlined by Yates and Potworowski

---

* Options : Help to identify options, as well as help to narrow down the list of feasible or realistic options
* Possibilities : Help to to identify possible outcomes for each of the identified options
* Judgement : Help to judge which outcomes are most likely
* Value : Help to identify the positive and negative impacts on stakeholders for each of the identified options
* Trade-offs : Help to make trade-offs on the above criteria for each options
* Understandable : Help to understand how and why the tools works as it does, and when it fails

---

## Cognitive processes for decision making : Abductive reasoning
Figure 2 
=> Evaluative AI 

---

## Explainable AI and decision making : Over-reliance
* Over-realiance : Automation bias
* Participants who do seem to believe that they understand explanatory information, leading to over-confidence, even when presented with explanations that contain no useful information. 
* => Not enought cognitive engagement with the system 
* => Cognitive forcing : 
    * Eg. forcing people to give a decision before seeing a recommendation
    * slightly mitigated overreliance, but not enought to lead to a statistically significant differences
    * Least prefered method by participant : people not wanting to exert mental energy
<!-- This indicates that Evaluative AI could prove useful by not fixating people on particular recommendations, but allowing peo- ple to assess whether evidence supports their hypotheses, rather than trying to understand the DA’s reasoning. -->
---

# How current decision support align with humain decision process
# (Explanable) AI as Decision Support
## Giving recommendations with no explanatory information
* Joue seulement sur la confiance en notre décision finale, de manière binaire : si je suis d'accord j'ai plus confianc , si je ne suis pas d'accord je reréfléchis à ma décision 
* Mais ne coche aucune des cases du tableau 2

## Giving recommendations with explanatory information
* Mis-calibrated trust => Give justifications for decisions, providing evidence to support the decisions, and making models simple and easy to understand
* Contrastive explaination help to considers other possibilities and make trades offs but only toward the model initial decision
<!-- What about other XAI method -->
## Giving recommendations with an interpretable model
* Same as before but this time it help to provide understanding of the machine decision

## Cognitive forcing  
* The commonality between these three approaches is that decisions are initially withheld from the decision maker 
* Withholding recommendations ‘forces’ the decision maker to cognitively engage with the decision and therefore, to consider different options and make trade- offs. 
*  partially helps to provide new options or to filter out unlikely options by forcing the decision maker to do so
*  Partially support making trade-offs
*  Does provide understandings of the machine decisions

---
# The evaluative AI framework
* Option: Show the most likely/certain option (perhaps withholding the probabilities) => Not a single recommendation
* Jugement: The machine provide feedback on humain jugement only. 
* Trade-offs (option awarness): 
    * Good decision makers assess an by looking for evidence that support it, but also evidence that **refutes it**.
    * Evaluative AI explain trade-offs between two sets of option
    * Evaluative AI provides evidence both for and against each option, irrelevant of the judged likelihood of that option.
    <!-- * SHAP fait ça  -->

---

# Example : Diagnosis

---

# Summary
* Naturellement leur modèle coche toutes les cases de leur tableau

---

# Long live explainable AI
* Application du XAI au delà de l'aide à la décision où les Recommendation drivent approaches sont bien et adapté
    * Making decision at scale 
* Il faudra toujours un model recommendation based pour n'importe quelle XAI technique
* Evaluative AI $ \subset $ XAI
* Many existing XAI tools are already adapted to Evaluative AI 
    * Constrastive explanation 
    * Feature importance (SHAP)
    * Wieghts of Evidence, case-based reasoning techniques 

---

# Limits
* First, if people tend to dismiss recommendations and any explainability information, why would they pay attention to evidence?
    * Evaluative AI give better control, with a process that they will naturally follow (contrairement à l'approche recommendation qui casse ça)
    * **=> j'aimerai bien une preuve de ça quand même **
* Cognitive load remain a problem 
    * Evaluative AI still reduce the quantity of information the decision maker needs (only revelant information are presented)
    * Still the less prefered solution by decision makers 

# Research agenda
Reprends et précise des éléments d'avant