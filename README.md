# SpinettAI
![Luis Alberto Spinetta](spinetta_picture.jpeg)

```
Dia, noche
Nunca escribas
Alguna vez yo podrás
Buenos Aires reposo
Brillaste, ll los realidad
Something beautiful
```

Este es el resultado de entrenar a un minusculo transformer (decoder) desde cero, solamente con letras de Spinetta.

El repositorio contiene la siguiente estructura:

```
.
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
Simplemente nos conectamos a la API de ```lyricsgenius``` y descargamos las letras de Spinetta. 
Luego de limpiar los datos, los guardamos en un archivo ```.json```.

## Pretrain
Generalmente, al trabajar con modelos de lenguaje pre-entrenados, se suele utilizar un modelo de lenguaje de gran tamaño, como por ejemplo GPT-2.
