# CNN practice

<br><br>

1. Dog and Cat image Classification<br>

    Data counts

    - cat      : 31197
    - dog      : 31303
<br>

  - dog_cat_classification.py
  - dog_cat_utils.py

<br><br>

2. Natural image Classification <br>

    Data counts

    - airplane : 727
    - car      : 968
    - cat      : 885
    - dog      : 702
    - flower   : 843
    - fruit    : 1000
    - motorbike: 788
    - person   : 986

  - natural_classification.py
  - natural_utils.py

-natural classification results<br>
model 1 : loss:        - accuracy: 0.94   ( train : val : test = 9 : 0 : 1)<br>
model 3 : loss: 0.2214 - accuracy: 0.9235 ( train : val : test = 8 : 1 : 1)<br>
model 4 : loss: 0.19   - accuracy: 0.9380 ( train : val : test = 8 : 1 : 1)<br>
model 5 : loss: 0.1576 - accuracy: 0.9452 ( train : val : test = 8 : 1 : 1)

<br><br>

3. Dev-Matching <br>
- DenseNet121 : 3-Folds Cross Validation x 33 [99 epochs]
- Last validation accuracy :  98.58657121658325
<img src="https://github.com/nhm0819/CNN_prac/blob/master/plots/dev_matching_acc.png?raw=true" width="40%" height="40%">  
