# LSTM test

## Caso di test

Per testare la rete LSTM usiamo il caso dove è presente un ingresso
e un uscita.
Il valore atteso dell'uscita $ y_1^(t) $ è il 1 se il valore dell'ingresso è uguale al valore del primo step $ x_1^{(t-1)} $ altrimenti è -1.

## Modello

Il modello di rete è un rete con un layer di 1 nodo LSTM con attivazione Tanh.
