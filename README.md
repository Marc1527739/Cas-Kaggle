# Pràctica Kaggle APC UAB 2022-23
### Nom: Marc Rodríguez Cañero
### DATASET: League of Legends
### URL: [[kaggle](http://....)](https://www.kaggle.com/datasets/bobbyscience/league-of-legends-diamond-ranked-games-10-min)

## Tota la informació del treball es troba explicada de forma més extensa dintre del repositori models en el document Anàlisi_Cas_Kaggle.pdf

## Resum
La base de dades (dataset) està formada per 9879 files i 4 columnes.

Cada fila mostrarà les estadístiques de una partida disputada en aquest joc (fins a un total de 9879 jocs ja que té 9879 files).
El joc està format per dos equips, un blau i un altre vermell, per això, podem veure que alguns atributs tenen la paraula red o blue, per especificar a quin equip pertany el atribut.

Inicialment, tenim 6 atributs del tipus float i 34 del tipus int. Tots els atributs han sigut convertits a tipus float, degut a que tots són atributs númerics i al convertir-los en float els podem tractar a tots els atributs de la mateixa manera.

Posteriorment, s'eliminen els atributs inecessaris que no ens aporten informació. El atribut gameID, que només ens indica el ID de la partida, característica inútil en quan al joc es refereix (i per a les prediccions posteriors).

En aquest punt tenim una nova base de dades original que ha estat adaptada.

Per acabar, la base de dades es normalitza, creant així una nova base de dades normalitzada.

El atribut objectiu serà l'atribut blueWins. Atribut binari que indica quin equip guanya la partida. Tindrà el valor de 1 si el equip blau guanya la partida o un atribut de 0 si el equip vermell guanya. 


### Objectius del dataset
El objectiu principal del dataset es aplicar diferents models (prediccions) per a poder comprovar quin és el pre-processament, la distribució del conjunt d'aprenentatge, el model i els millors paràmetres per a cada model que millors resultat dona per aquesta base de dades. 

Durant el procés, també tenim altres objectius, com analitzar el perquè dels resultats i analitzar resultats no esperats o resultats pitjors, no sempre analitzar els millors resultalts.

### Preprocessat

El processament de les dades ha consistit en tracta de reduir la dimensionalitat de la base de dades segons diferents patrons i correlacions.

Quan es treballa en dades n-dimensionals (més d'un atribut), una opció és reduir la seva n-dimensionalitat i escollir 2,3,4... components principals, obtenint unes dades que (ara sí) poden ser visualitzables en un nou espai. Per això, s'apliquen les técniques, PCA (Anàlisis Principal de les Components) i TNSE (t-nse) sobre la base de dades normalitzada creant així la base de dades PCA i la base de dades TNSE. 

## Experiments
Durant aquesta pràctica inicialment hem aplicat els models bàsics SVM (amb kernel gaussiana), regressió lógistica i regressió lineal per a totes les bases de dades creades que ens ha permès detectar amb quin conjunt d’aprenentatge i quina base de dades obtenim millors resultats (prediccions), en quan a temps i precisió. El millor model seria el model de regressió lineal sobre la base de dades normalitzada amb conjunt d’aprenentatge d’un 60% de test i 40% de train.

Hem continuat amb l'aplicació de models més avançats com Random Forest i K-nearest Neighbors (KNN) i ens han donat resultats similars al models bàsics i no ens aportaven cap informació rellevant. 

Això cambiaria al arribar l'aplicació del métode de validació creuada K-fold, quan hem vist que amb el conjunt d’aprenentatge amb 60% test, 20% entrenament i 20% de validació (K=4) sobre la base de dades normalitzada obteniem millors resultats.

Finalment, hem acabat analitzant utilitzant les tècniques  GridSearchCV i RandomizedSearchCV, que ens han permès realitzar combinacions de models, concepte que no havien contemplat, i analitzar quins són els millors paràmetres per als millors models, també afegint un nou concepte, ja que durant tots els anàlisis, pràcticament hem utilitzat els paràmetres per defecte en els models. 

### Model
Degut a que, durant la pràctica hem utilitzat el mateix model amb diferents valors en quan a la base de dades i conjunt d'aprenentatge i hem analitzat diverses métriques com la precisió, el temps i el mse, en aquesta taula per a que quedi més clar, només inclourem els millors resultats de cada model utilitzat, amb els seus paràmetre, la base de dades i el conjunt d'aprenentatge utilitzat i les métriques precisió i temps que han donat com a resultats.

La taula extensa amb tots els experiments es troba en el document mencionat al inici d'aquest fitxer.

Taula resum amb els millors resultats per a cada model:

| Model | Hiperparametres | Base de dades | Conjunt d'aprenentatge | Precisió (%) | Temps (segons) |
| -- | -- | -- | -- |--| -- |
| Regressió logística | Per defecte | Normalitzada | 60% train 40% test | 72,89 | 0.0229 | 
| Regressió lineal | Per defecte | Normalitzada | 60% train 40% test | 27,42 | 0.0089 |
| SVM | kernel: rbf | Original | 60% train 40% test | 72,79 | 2.1056 |
| KNN | Per defecte | Normalitzada | 60% train 40% test | 69,14 | 0.6634 |
| Random Forest | Per defecte | Normalitzada | 60% train 40% test | 71,62 | 0.9612 |
| K-fold | n_splits (K) = 5 | Normalitzada | 60% test 20% train 20% val | 74,94 | 0.3059 |
| GridSearchCV | Combinació: K-fold = 5 i Regressió logística per defecte | Normalitzada | 60% test 20% train 20% val | 73,09 | Varies proves (no es calcula el temps)|
| RandomizedSearchCV | Combinació: K-fold = 5 i Regressió logística per defecte | Normalitzada | 60% test 20% train 20% val | 73,09 | Varies proves (no es calcula el temps)|

## Demo
Tot i que tot el desenvolupament s'ha realizat al notebook Cas_Kaggle, trobem tres fitxers amb els quals podem fer alguna prova.

Tenim tres scripts python que podem cambiar per fer proves. El script generate_features.py es un script que neteja la base de dades (la càrrega) i la prepara per posteriors probes, el script train_model.py realitza les millors prediccions per a cada model utilitzant les bases de dades creades al script generate_features.py i el script score_model.py es un script que ens proporciona un exemple de com fer noves predccions utilitzant models ja entrenats.

Variant qualsevol dels tres scripts podriem aconseguir realitzar tot tipus de proba.

Inicialment, poder fer les probes predeterminades ejecutant qualsevol dels tres script, ja sigui per la terminal de comandes, o utilitzant plataformes que ens permetin ejecutar script en python. 

## Conclusions
El millor model que s'ha aconseguit ha estat el model d'aplicar validació creuada, la técnica K-fold amb el conjunt d’aprenentatge amb 60% test, 20% train i 20% val (K=4), una precisió del  74,94% i un temps de 0.3059. 

En comparació amb l'estat de l'art i els altres treballs que hem analitzat, considerem que hem tocat casi tots els conceptes vistos i estudiats en el àmbit aprenentatge computacional (classificació). Tot i així, en un acte d'autocrítica, potser ens hauria faltat tocar el tema de xarxes neuronals, però no l'hem tocat degut a que no tenim imatges per a poder realitzar les prediccions d'aquestes, és a dir, per el problema plantejat no ho hem considerat adient.

## Idees per treballar en un futur
Crec que seria interesant indagar més aplicar els métodes One-vs-One o One-vs-Rest i combinar-los amb els utilitzats ja que aquests dos métodes són adients per els problemes de classificació binària, com és aquest cas. Crec que podriem obtenir millores en els resultats.

## Llicencia
El projecte s’ha desenvolupat sota llicència ZZZz.
