# Plan de Clase 1 — NLP, Tokenización y Vectorización
**UTN-FRSF · Ingeniería en Sistemas de Información · Inteligencia Artificial**  
**Dr. Jorge Roa · Dra. María de los Milagros Gutiérrez**

---

**Pregunta central:** ¿Cómo representa texto una máquina?

**Formato:** Notebook Jupyter con metadata de slideshow (nbconvert + Reveal.js)  
**Comando para slides:** `jupyter nbconvert --to slides clase1_slides.ipynb --post serve`

---

## Bloque 1 — Introducción al ecosistema

### Contenido teórico

**IA, ML, DL — las cajas chinas**
- Diagrama de contenedores anidados: IA ⊃ ML ⊃ DL
- Tabla de ejemplos del curso con su categoría (NLP→ML, Embeddings→DL, LLMs→DL, RAG→ML+DL, Agentes→IA)

**Open source vs closed source**
- Tabla comparativa: acceso, privacidad, costo, uso en el curso
- Conclusión: el curso usa modelos open weight via Ollama (qwen3:8b / qwen3:4b)

**Discriminativo vs Generativo**
- Discriminativo: clasifica a partir de texto existente (BERT, RoBERTa)
- Generativo: genera texto nuevo a partir de una instrucción (GPT, Llama, Claude)
- Énfasis: el curso trabaja principalmente con modelos generativos

**La explosión 2013 → 2025**
- Tabla cronológica: Word2Vec (2013) · Transformer (2017) · BERT (2018) · GPT-3 (2020) · ChatGPT (2022) · Agentes (2024/25)

**El pipeline de un LLM — caja negra**
```
Texto → Tokenización → Embeddings → Modelo → Output
(hoy)     (hoy)          (hoy)      (Clase 2) (Clase 2)
```
- Cada clase abre una o más cajas del pipeline

**TP Integrador**
- Agente RAG funcional sobre un dominio a elección
- Entregables: código + notebook documentado + demo en vivo + informe de decisiones técnicas
- Componentes: ingestión de documentos, búsqueda semántica, agente con herramientas, stack open source

---

## Bloque 2 — Representaciones clásicas: BoW y TF-IDF

### Contenido teórico

**El problema de fondo**
- Sinonimia: "auto" y "coche" son palabras distintas pero equivalentes
- Polisemia: "banco" puede ser institución financiera o mueble
- Contexto: "fui al banco" — ¿cuál? Depende de todo lo que lo rodea

**Bag of Words**
- Idea: representar cada documento como un vector de conteos de palabras
- Corpus de ejemplo: "El gato come pescado" / "El perro come carne" / "El gato duerme"
- Matriz resultante con columnas por palabra única
- Problemas: sinonimia ignorada, orden de palabras perdido
- Conector: BM25 mejora BoW para búsqueda → hybrid search en Clase 3

**TF-IDF**
- Idea: frecuencia en el documento × rareza en la colección
- Fórmula: `TF-IDF(término, doc) = TF × log(N / df)`
- Tabla comparativa: "el/la/de" (muy bajo) · "come" (bajo) · "gato" (medio) · "pescado" (alto)

### Ejercicios
1. Ejecutar BoW → observar la matriz (4')
2. Ejecutar TF-IDF → comparar scores con BoW (3')
3. **Ejercicio libre:** escribir 3 oraciones propias y construir la matriz BoW (5')

### Código
```python
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import pandas as pd

oraciones = ["El gato come pescado", "El perro come carne", "El gato duerme"]
vec = CountVectorizer()
df = pd.DataFrame(vec.fit_transform(oraciones).toarray(),
                  columns=vec.get_feature_names_out())
```

---

## Bloque 3 — Tokenización

### Contenido teórico

**¿Qué es un token?**
- Por palabras: `["tokenización"]` — simple pero falla con palabras nuevas
- Por caracteres: `["t","o","k","e","n",...]` — nunca falla pero secuencias muy largas
- Por subpalabras ✓: `["token","ización"]` — balance ideal, requiere entrenamiento previo

**BPE — Byte Pair Encoding** (GPT-4o, Llama, Mistral)
1. Arrancás con caracteres: `"base"` → `b · a · s · e`
2. Contás pares adyacentes: `"b a"` → 8 veces, `"a s"` → 5 veces...
3. Fusionás el más frecuente: `"b a"` → `"ba"` en todo el corpus
4. Repetís hasta alcanzar el vocabulario objetivo (GPT-4o: ~200K tokens)

Implementación manual en ~20 líneas — cada fusión queda expuesta.

**WordPiece** (BERT, DistilBERT)
- Maximiza probabilidad del corpus
- Marca continuación con `##`: `token` + `##ización`

**SentencePiece** (Llama 2, T5, mT5)
- No pre-tokeniza por espacios — ideal para idiomas sin separadores
- Marca inicio de palabra con `▁`: `▁token` + `ización`

**El costo del idioma**
- GPT-4o (o200k_base), sin API key, 100% local
- Español: ~14 tokens / Inglés: ~8 tokens para la misma oración
- Impacto en costos de API: escribir en español puede costar hasta el doble

### Ejercicios
1. NLTK y spaCy sobre la misma oración — comparar diferencias (5')
2. BPE manual — observar cada fusión iteración por iteración (5')
3. **Ejercicio libre:** tokenizar frase propia en español e inglés, contar diferencia. ¿Qué pasa con "microservicio"? ¿Y con emojis? (5')

### Código
```python
import tiktoken
enc = tiktoken.get_encoding("o200k_base")  # GPT-4o, sin API key

frase_es = "La inteligencia artificial está transformando la ingeniería."
frase_en = "Artificial intelligence is transforming engineering."
print(f"Español: {len(enc.encode(frase_es))} tokens")
print(f"Inglés:  {len(enc.encode(frase_en))} tokens")
```

---

## Bloque 4 — Embeddings y similitud semántica

### Contenido teórico

**Embeddings — representar significado**
- Palabras con contextos similares → vectores cercanos
- "auto" y "coche" aparecen cerca de "manejar", "rueda" → sus vectores quedan cerca
- La distancia codifica significado: "auto" lejos de "banana"
- Limitación de Word2Vec: un vector por palabra sin importar el contexto
  - "banco" (dinero) = "banco" (mueble) → mismo punto en el espacio

**Sentence Transformers**
- Modelo: `paraphrase-multilingual-MiniLM-L12-v2` (~90MB, funciona bien en español)
- Input: oración completa → output: vector de 384 dimensiones
- El vector cambia según el contexto completo

**Similitud coseno**
- No importa la longitud — importa el ángulo. Score entre -1 y 1
- "Me gusta el fútbol" ↔ "Disfruto el fútbol" → 0.93
- "Me gusta el fútbol" ↔ "Juego al tenis" → 0.58
- "Me gusta el fútbol" ↔ "La física cuántica es compleja" → 0.11

**Visualización 2D**
- Panel izquierdo: clusters semánticos (vehículos, frutas, royalty, finanzas)
- Panel derecho: analogía vectorial `rey − hombre + mujer ≈ reina`

### Ejercicios
1. Cargar modelo y generar embeddings del corpus de 20 oraciones (7')
2. Calcular similitud coseno de 3 pares, verificar scores (5')
3. **Ejercicio libre:** ¿qué score tienen "auto" y "coche"? ¿"banco" en dos contextos distintos? ¿Dos oraciones en idiomas diferentes? (5')

### Código
```python
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

modelo = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
embeddings = modelo.encode(CORPUS)
np.save('corpus_embeddings_clase1.npy', embeddings)  # → Clase 3
```

---

## Bloque 5 — Visualización con UMAP

### Contenido teórico

**¿Por qué visualizar?**
- Los embeddings tienen 384 dimensiones — imposible visualizar directamente
- UMAP reduce a 2D preservando estructura local: puntos cercanos en 384D → cercanos en 2D
- No es perfecto — UMAP es estocástico y el corpus es chico

### Ejercicios
1. Reducir embeddings de 384D a 2D con UMAP (3')
2. Scatter plot con 4 colores (NLP, LLMs, RAG, Agentes) — observar clusters (7')
3. **Ejercicio libre:** agregar 2-3 oraciones propias — ¿forman cluster propio o se mezclan? (10')

### Código
```python
import umap
reducer = umap.UMAP(n_components=2, n_neighbors=5, min_dist=0.3, random_state=42)
embeddings_2d = reducer.fit_transform(embeddings)
```

---

## Archivos generados

| Archivo | Descripción | Usado en |
|---|---|---|
| `corpus_embeddings_clase1.npy` | Embeddings 384D del corpus de 20 oraciones | Clase 3 — vector store |
| `corpus_clase1.csv` | Oraciones + categorías del corpus | Clase 3 — metadata |
| `embeddings_viz_clase1.png` | Visualización 2D: clusters + analogía vectorial | — |
| `clusters_umap_clase1.png` | Scatter UMAP con 4 categorías | — |

---

## Stack de la clase

```
nltk · spacy · tiktoken · sentence-transformers · umap-learn · scikit-learn · matplotlib
```

---

## Referencias

### Libros
- Jurafsky, D. & Martin, J. H. (2024). *Speech and Language Processing* (3rd ed. draft). Stanford.  
  https://web.stanford.edu/~jurafsky/slp3/
- Goodfellow, I., Bengio, Y. & Courville, A. (2016). *Deep Learning*. MIT Press.  
  https://www.deeplearningbook.org/
- Tunstall, L., von Werra, L. & Wolf, T. (2022). *NLP with Transformers*. O'Reilly.  
  https://www.oreilly.com/library/view/natural-language-processing/9781098136789/

### Papers
- Mikolov, T., et al. (2013). Efficient Estimation of Word Representations in Vector Space. *ICLR 2013*.  
  https://arxiv.org/abs/1301.3781
- Sennrich, R., Haddow, B. & Birch, A. (2016). Neural Machine Translation of Rare Words with Subword Units. *ACL 2016*.  
  https://arxiv.org/abs/1508.07909
- Devlin, J., et al. (2018). BERT: Pre-training of Deep Bidirectional Transformers. *NAACL 2019*.  
  https://arxiv.org/abs/1810.04805
- Reimers, N. & Gurevych, I. (2019). Sentence-BERT. *EMNLP 2019*.  
  https://arxiv.org/abs/1908.10084
- McInnes, L., Healy, J. & Melville, J. (2018). UMAP. *arXiv*.  
  https://arxiv.org/abs/1802.03426
