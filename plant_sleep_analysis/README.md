**Sleep tracking with plants**

This is sample code to take Wav files recorded with the oxocard, which are then split into segments of "awake", as recorded by a sleep tracking app, and "asleep", corresponding to the first 30 minutes of sleep as recorded by the sleep tracking app


3918 spectrograms, 1589 awake, and 2328 asleep
Test Accuracy: 60.59%
Training accuracy 99.39 after 50 epochs



             precision    recall  f1-score   support

       Awake       0.56      0.64      0.60       318
      Asleep       0.73      0.65      0.69       466

    accuracy                           0.65       784
   macro avg       0.64      0.65      0.64       784
weighted avg       0.66      0.65      0.65       784

![accuracy](https://github.com/user-attachments/assets/aec3d807-e0ac-4e3d-83a4-56dd308b510e)
![confusion_matrix](https://github.com/user-attachments/assets/1b4bed81-62b8-4841-aeaf-7fc821f00aab)
