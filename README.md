# ListLabs_zadatak3

Dijelovi koda.

1. Izabir startnog modela. Izabrao sam K-nearest neighbors te testirao accuracy sa zadanim postavkama.
   - Dobio sam accuracy 0.99823, sto je visok rezultat
   - Provjerio sam da li skalirani podaci daju jos bolje rjesenje i ispalo je 0.99969, te sam u nastavku koristio skalirane modatke
   - Zatim sam napravio podesavanje modela mijenjajuci dva hiperparametra unutar for petlje i pomocu ParameterGrid() (slika 1.)
   - na tom setu hiperparametara najbolji rezultat dala je kombinacija n_neighbors=3, te weights='uniform'
   - Model sam refitao prema najboljim parametrima te su ispisane su confusion matrix i classification report
     ![Figure1_knn_tuning results](https://github.com/user-attachments/assets/7c84efb0-f6f5-4cec-a921-8cbb43abb810)

     
2. U drugom dijelu sam testirao i druge modele, ukupno njih 6 sa zadanim postavkama, te ovaj put koristeci GridSearchCV cross validation
   - Ispisane su matrice zabune te su testirne razlicite performanse modela tokom cross validation procesa (slika2, slika3, slika4)
   - KNN model se pokazao kao najbolji iako svi modeli imaju vrlo visoke rezultate na ovom skupu podataka
   ![Figure2_confusion_matrices_for_default_models](https://github.com/user-attachments/assets/385b1bca-f058-49df-bf03-a1055cdb894d)
![Figure3_cross_validate_metrics](https://github.com/user-attachments/assets/a5485d7b-31ef-4b8e-a900-1392580746a2)

   ![Figure4_mean_of_cross_val_metrics](https://github.com/user-attachments/assets/4b48449e-fe69-4845-baad-318df7e4aa8c)

 3. U trecem dijelu izvrsio sam podesavanje hiperparametara za svih 6 ML modela koristeci GridSearchCV i po dva hiperparametara za svaki model
      - Ispisane su sve matrice zabune za svaki model s najboljim parametrima (slika 5), te classification reporti, i accuracy score (slika6)
      - U ovom koraku bi se inace moglo utrositi vise vremena trazeci najbolje parametre raznim metodama, ali najbolji modeli vec su davali jako visoke rezultate. 

![Figure5_confusion_matrices_for_tuned_models](https://github.com/user-attachments/assets/61396357-606f-44b5-af6e-534e3a7dd8bb)

![Figure6_final_test_scores](https://github.com/user-attachments/assets/24a23079-a96a-4503-bd13-4a656baa3ed0)

4. U cetvrtom dijelu sam testirao vaznost featura pomocu modela koji imaju opciju da izmjere njihovu vaznost (Decision Tree, Random Forest)
   - Pokazalo se da su cetvrti i sesti feature (indexi 3 i 5) najvazniji za oba modela
   - Tada sam ponovno testirao modele sa zadanim postavkama ali ovaj put uzimajuci u obzir samo dvije najvaznije kolone podataka
   - Iako postoje razne metode za redukciju dimenzionalnost u ovom setu podataka pokazalo se da i dva najbolja featura daju jako dobre rezultate
   - Prikazane se matrice zabune(slika 7), klasifikacijska izvjesca, te accuracy score
   - Dodatno prikazao sam (slika 8) klasifikacijske granice za sve modele na temelju samo dva najvaznija featura
  
   ![Figure7_confusion_matrices_for_reduced_models](https://github.com/user-attachments/assets/c3fcd9ec-7751-49f1-ab5f-1bb003d8fb54)

   ![Figure8_decision_maps_2features](https://github.com/user-attachments/assets/97782026-d264-4e00-9b73-5bd1e57fc7cc)

5. Na kraju sam prema rezultatim odabrao K-nearest neighbors kao najbolji model te ga spremio pomocu joblib-a u pkl format
