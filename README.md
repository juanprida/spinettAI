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

Este es el resultado de entrenar a un pequeño transformer desde cero, solamente con letras de Spinetta. La intención de este repositorio es mostrar cómo se puede entrenar un modelo de lenguaje desde cero, y luego utilizarlo para generar texto.

El repositorio contiene la siguiente estructura:

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
En esta carpeta se encuentran los scripts para generar los datos utilizados para entrenar el modelo.
Es necesario contar con un token para poder utilizar la API de [Genius](https://docs.genius.com/) para descargar letas.

Simplemente nos conectamos a la API de ```lyricsgenius``` y descargamos las letras de Spinetta. 
Luego de limpiar los datos, el resultado se guardara en un archivo ```.json```.

Todo esto se puede hacer con el script ```prepare.py```.

## pretrain
Generalmente, al trabajar con modelos de lenguaje, la primera parte del proceso es pre-entrenar el modelo con un corpus de texto grande en donde el objetivo es aprender a predecir la siguiente palabra.
En un contexto industrial, se suele utilizar un corpus de texto muy grande y diverso, de donde luego se extraen las características aprendidas para utilizarlas en otras tareas.

En la carpeta se encuentra el script ```tokenize_data.py``` que se encarga de tomar el archivo ```.json``` generado en la carpeta ```data``` y transformarlo en un set de input y output tokens.
Esto se hace utilizando la librería ```tokenizers```, que nos permite crear un tokenizador a partir de un vocabulario y luego utilizarlo para codificar y decodificar texto.
Para mas informacion sobre tokenizers, ver [aquí](https://huggingface.co/docs/tokenizers/python/latest/quicktour.html).

Luego, en  ```model.py``` se encuentra la definición del modelo, en este caso un transformer del estilo decoder only, muy similar a GPT-2 o GPT-3, con la diferencia de que este modelo es mucho mas pequeño en terminos de cantidad de parámetros (embedding size, cantidad de capas, etc).

Finalmente, en ```train.py``` se encuentra el código para entrenar el modelo.
