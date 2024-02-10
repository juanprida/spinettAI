# Language Models: Pretraining and Supervise finetuning

In this repo, using a lyrics dataset from Luis Alberto Spinetta, we build a transformer decoder only from scratch in Pytorch.
We also explore PEFT techniques, finetuning a LLM using Lora (Low rank adapters) on a google colab.

Sing to me Luis Alberto:
```
Dia, noche
Nunca escribas
Alguna vez yo podrás
Buenos Aires reposo
Brillaste, ll los realidad
Something beautiful
```



The repo contains rhe following structure:
```
├── README.md
├── data
│   ├── prepare.py
├── pretrain
│   ├── model.py
│   ├── tokenize_data.py
│   ├── train.py
├── finetune
```

## data
In this folder we have all the scripts needed to generate the training dataset. 
En esta carpeta se encuentran los scripts para generar los datos utilizados para entrenar el modelo.
The lyrics can be obtained from the [Genius](https://docs.genius.com/) 

## pretrain
Starting from random weights, we train our model on next token prediction.

We start tokenising our lyrics with```tokenize_data.py```.
Lastly,  ```model.py``` contains the class for the transformer itself and ```train.py``` performs the forward and backward passes updating pir weights.
