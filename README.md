# Catégorisez automatiquement des questions

Stack Overflow est un site célèbre de questions-réponses liées au développement informatique. Pour poser une question sur ce site, il faut entrer plusieurs tags de manière à retrouver facilement la question par la suite. Pour les utilisateurs expérimentés, cela ne pose pas de problème, mais pour les nouveaux utilisateurs, il serait judicieux de suggérer quelques tags relatifs à la question posée.

Amateur de Stack Overflow, qui vous a souvent sauvé la mise, vous décidez d'aider la communauté en retour. Pour cela, vous développez un système de suggestion de tag pour le site. Celui-ci prendra la forme d’un algorithme de machine learning qui assigne automatiquement plusieurs tags pertinents à une question.



### La mission de ce projet:

- Réaliser le pétraitement des questions (documents) issues des données de l'API stackexchange-explorer
- Comparer des approches suppervisées (LR, KNN, SVM, RF, Gboust) et non supervisées (LDA, NMA) avec plusieurs tests de méthodes d'extraction de features. Notamment, 
- une approche de type bag-of-words, et 3 approches de Word/Sentence Embedding  Word2Vec, BERT et USE.  
- Développer une API 
