

# MSGO

Este repositorio proporciona los datos y métodos del artículo: \
**Generador de estructuras moleculares basado en pseudodatos para revelar compuestos químicos desconocidos**

Aceptado para publicación en **Nature Machine Intelligence**

Autores: Nanyang Yu&dagger;, Zheng Ma&dagger;, Qi Shao&dagger;, Laihui Li, Xuebing Wang, Bingcai Pan, Hongxia Yu y Si Wei&ddagger;

&dagger;: Contribución igual \
&ddagger;: Autor correspondiente

### Configuración

### Entorno
Python: 3.7 \
Torch: 1.7.1

### Datos

[2025.11.28 actualización] Debido al fallo del enlace de datos smiles, hemos colocado directamente los archivos relevantes en la carpeta de datos. Por favor, consúltelos en la carpeta de datos.

Para el entrenamiento, utilizamos más de 30k pares de smiles-espectro pseudo generados por cfmid (~~puede descargar el archivo de listas de smiles originales [aquí](https://www.aliyundrive.com/s/BXKd1ThQy19)~~). Para la evaluación, utilizamos más de 300 espectros reales para verificar nuestro método (~~descargar [aquí](https://www.aliyundrive.com/s/JTVRbipqXLh)~~). Para la evaluación en muestras reales, utilizamos un conjunto de datos LC–QTOF para muestras de aguas residuales para verificar nuestro modelo (descargar [aquí](https://pan.baidu.com/s/1KuKzkRMGr1DhkfJobkXKpQ), código: gmas).


### Pesos del modelo
Proporcionamos el modelo MSGO ([pfas](https://pan.baidu.com/s/1J_qllzAsv-dxqD2D28_sLw), código: 0bfg; [lipid](https://pan.baidu.com/s/1lbL7hdHgblsQAgkBomlBMg), código: 37it) entrenado con pares de smiles-espectro pseudo utilizando todos los métodos mencionados en el artículo. También puede entrenar su propio modelo con otros métodos.

### Entrenamiento

Puede replicar nuestro experimento, incluyendo todas las técnicas:
```
python tools/train.py --id all_trick --use_precursor 1 --use_mask 1 --use_formula 1
```
Puede ver más opciones en opt.py

### Evaluación

Descargue los pesos del modelo en ckpts/pfas o ckpts/lipid, ejecute
```
python tools/eval.py --log_path [ckpts/pfas or ckpts/lipid]
```

### Predecir datos reales
Proporcionamos datos de ejemplo en data/example.

Para pfas, ejecute :
```
python tools/eval_standard.py --log_path ckpts/pfas --real_csv ./data/example/pfas.csv --out_csv ./pfas_results.csv --beam_size 500 --polar neg
```

Para lipid, ejecute:
```
python tools/eval_standard.py --log_path ckpts/lipid --real_csv ./data/example/lipid.csv --out_csv ./lipid_results.csv --beam_size 300 --polar pos
```
Luego, podrá obtener un archivo csv de resultados que incluye las 10 mejores predicciones.

---
## Por hacer
- [x] Publicar pesos del modelo
- [x] Publicar datos pseudo y reales
- [x] Publicar proceso de entrenamiento

---
## Implementación de modelos de referencia 

Todo el código está en las carpetas baseline_models

Para baseline_models/ms2mol
```
cd ms_bart 
python train.py
```

Para massgenie y spec2mol

Entrenamiento
Puede replicar nuestro experimento con la configuración predeterminada, ejecute

```python tools/train.py```

Evaluación
Puede ejecutar

```python tools/utils_eval.py```

Predecir datos reales
Proporcionamos un archivo example.py como referencia. Puede reemplazar [data path] con sus propios datos para la predicción.
