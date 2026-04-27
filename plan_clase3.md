# Plan de Clase 3 — RAG: Retrieval Augmented Generation
**UTN-FRSF · Ingeniería en Sistemas de Información · Inteligencia Artificial**  
**Dr. Jorge Roa · Dra. María de los Milagros Gutiérrez**

---

**Pregunta central:** ¿Cómo le doy conocimiento propio a un LLM sin reentrenarlo?

**Formato:** Notebook Jupyter con metadata de slideshow (nbconvert + Reveal.js)  
**Comando para slides:** `jupyter nbconvert --to slides clase3_slides.ipynb --post serve`

---

## Timing

| Sección | Duración | Acumulado |
|---|---|---|
| CSS + título + recap + setup | 10 min | 0:10 |
| **B1 — ¿Por qué RAG?** | 20 min | 0:30 |
| **B2 — Pipeline RAG naive** | 55 min | 1:25 |
| *Break* | 10 min | 1:35 |
| **B3 — Hybrid Search** | 30 min | 2:05 |
| **B4 — RAG avanzado** | 25 min | 2:30 |
| **B5 — Evaluación de RAG** | 15 min | 2:45 |
| Cierre: resumen + bibliografía + bridge Clase 4 | 10 min | 2:55 |
| Buffer | 5 min | 3:00 |

---

## Recap — conexión con Clases 1-2

**1 slide, formato tabla (misma estructura que Clase 2)**

```
  Clase 1                            Clase 2
  ───────                            ───────
  Texto → Tokens → Embeddings        Embeddings → Transformer → Texto generado
       BoW  TF-IDF  Vectores 384D        Attention   Prompting   llamar_llm()
                │                              │
                └──────── HOY ─────────────────┘
                     RAG une retrieval con generación
```

| Lo que ya sabemos | Lo que resolvemos hoy |
|---|---|
| Embeddings capturan significado semántico | Los usamos para **recuperar** documentos relevantes |
| TF-IDF / BM25 buscan por coincidencia de palabras | Los combinamos con búsqueda semántica |
| `llamar_llm()` genera texto con un prompt | Le **inyectamos contexto** antes de generar |
| Context window tiene límites | RAG selecciona **solo lo relevante** |

---

## Setup

```bash
pip install sentence-transformers chromadb rank_bm25 groq
```

**Lo que usamos hoy:**
- `sentence-transformers` — embeddings (continuidad de Clase 1)
- `chromadb` — vector store in-memory
- `rank_bm25` — búsqueda léxica
- `groq` — API LLM (free tier, rápido)
- `llamar_llm()` — wrapper de Clase 2 (ahora default Groq)

### Código

```python
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from sentence_transformers import SentenceTransformer
modelo_emb = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

import os

USE_OLLAMA = False  # True para Ollama local, False para Groq (recomendado)

def llamar_llm(messages, model=None, temperature=0.7):
    """Wrapper unificado para Ollama o Groq (de Clase 2)."""
    if USE_OLLAMA:
        import ollama
        model = model or 'qwen3:8b'
        resp = ollama.chat(model=model, messages=messages,
                           options={'temperature': temperature})
        return resp['message']['content']
    else:
        from groq import Groq
        client = Groq(api_key=os.environ.get('GROQ_API_KEY'))
        model = model or 'llama-3.1-8b-instant'
        resp = client.chat.completions.create(
            model=model, messages=messages, temperature=temperature
        )
        return resp.choices[0].message.content

# Corpus de Clase 1 (regenerado — los estudiantes pueden no tener el .npy)
CORPUS = [
    "Los modelos de lenguaje procesan texto para entender su significado.",
    "El procesamiento del lenguaje natural permite a las máquinas leer y escribir.",
    "Tokenizar un texto significa dividirlo en unidades mínimas de significado.",
    "Los embeddings representan palabras como vectores en un espacio matemático.",
    "La similitud semántica entre palabras puede medirse con la distancia coseno.",
    "Los modelos de lenguaje de gran escala se entrenan con enormes volúmenes de texto.",
    "GPT y Llama son ejemplos de modelos generativos basados en la arquitectura Transformer.",
    "El fine-tuning adapta un modelo preentrenado a una tarea específica.",
    "Un modelo base predice el siguiente token; un modelo instruct sigue instrucciones.",
    "El tamaño del contexto limita cuánto texto puede procesar un modelo a la vez.",
    "RAG combina recuperación de información con generación de texto.",
    "Un vector store guarda embeddings para hacer búsquedas semánticas eficientes.",
    "La búsqueda híbrida combina similitud semántica con búsqueda por palabras clave.",
    "El chunking divide documentos largos en fragmentos aptos para ser indexados.",
    "Recuperar contexto relevante reduce las alucinaciones del modelo.",
    "Un agente inteligente percibe su entorno y toma acciones para alcanzar objetivos.",
    "Los agentes modernos usan LLMs para razonar y decidir qué herramienta usar.",
    "El patrón ReAct alterna entre razonamiento y acción en un ciclo iterativo.",
    "Los sistemas multiagente dividen tareas complejas entre agentes especializados.",
    "Claude Code y Manus son ejemplos de agentes profundos con planificación y memoria.",
]
CATEGORIAS = ['NLP'] * 5 + ['LLMs'] * 5 + ['RAG'] * 5 + ['Agentes'] * 5

corpus_embeddings = modelo_emb.encode(CORPUS, show_progress_bar=True)
```

---

## Bloque 1 — ¿Por qué RAG? (20 min)

### Contenido teórico

**El problema del LLM puro — tres limitaciones que no se resuelven con mejor prompting**

| Limitación | Ejemplo | Consecuencia |
|---|---|---|
| **Knowledge cutoff** | "¿Qué pasó en las elecciones de ayer?" | El modelo no tiene datos recientes |
| **Alucinaciones** | "Citame el artículo 47 del reglamento de la UTN" | Inventa contenido con total confianza |
| **Context window** | 50.000 documentos internos de una empresa | No caben ni en 1M de tokens |

```
  Opción 1: Re-entrenar el modelo     → costoso, lento, no práctico
  Opción 2: Meter todo en el prompt   → no escala, "Lost in the Middle"
  Opción 3: RAG                       → recuperar solo lo relevante ✓
```

> **RAG = no cambiás el modelo, cambiás lo que le das para leer.**

**RAG — Retrieval Augmented Generation**

Idea central: antes de que el LLM responda, buscar información relevante y ponerla en el prompt.

Diagrama SVG del pipeline:
```
  Base de conocimiento → [indexar] → Vector Store
  Query → [buscar] → Vector Store → [top-k] → Prompt (chunks + query) → LLM → Respuesta
```

Los 4 pasos:
1. **INDEXAR:** documentos → chunks → embeddings → vector store
2. **CONSULTAR:** query → embedding → buscar top-k chunks similares
3. **AUGMENTAR:** armar prompt = system + chunks recuperados + pregunta
4. **GENERAR:** el LLM responde usando los chunks como contexto

### Demo guiada — LLM sin contexto vs con contexto

```python
pregunta = "¿Qué es el chunking en el contexto de RAG?"

# Sin contexto — el LLM solo con su conocimiento
resp_sin = llamar_llm([
    {"role": "system", "content": "Respondé en español, máximo 3 oraciones."},
    {"role": "user", "content": pregunta}
], temperature=0.3)

# Con contexto — le damos info del corpus
contexto = "\n".join([
    "El chunking divide documentos largos en fragmentos aptos para ser indexados.",
    "RAG combina recuperación de información con generación de texto.",
    "Recuperar contexto relevante reduce las alucinaciones del modelo."
])
resp_con = llamar_llm([
    {"role": "system",
     "content": "Respondé SOLO con base en el contexto proporcionado. Si no hay info suficiente, decilo. Máximo 3 oraciones."},
    {"role": "user",
     "content": f"Contexto:\n{contexto}\n\nPregunta: {pregunta}"}
], temperature=0.3)
```

> La diferencia: con contexto, el LLM se basa en TUS documentos. Eso es RAG. Ahora vamos a automatizar el retrieval.

Sin ejercicio libre — la demo es guiada. 10 min teoría + 10 min demo.

---

## Bloque 2 — Pipeline RAG naive (55 min)

### Corpus realista en español

5 párrafos sobre temas de IA relevantes para ingeniería de sistemas, cada uno de 4-6 oraciones (~80-120 palabras). Son lo suficientemente largos para necesitar chunking pero lo suficientemente cortos para procesarse en vivo.

Temas:
1. **Arquitectura de sistemas con IA** — integrar modelos de ML en microservicios, latencia de inferencia, sidecar pattern, streaming de tokens
2. **Testing de software con IA** — generación de tests con LLMs, Copilot/Claude Code, limitaciones de tests generados, visual testing con CNNs
3. **Bases de datos vectoriales** — qué son, ChromaDB/Pinecone/Weaviate/Qdrant, HNSW, métricas de distancia (coseno para texto, euclidiana para imágenes)
4. **Seguridad en aplicaciones de IA** — prompt injection, data poisoning, documentos falsos en RAG, mitigaciones
5. **MLOps y deploy de modelos** — CI/CD para modelos, model drift, MLflow/W&B/DVC, cuantización INT8/INT4

### Contenido teórico

**Chunking — partir documentos en fragmentos**

¿Por qué? Un documento de 10 páginas no cabe como un solo embedding.

| Estrategia | Cómo funciona | Cuándo usarla |
|---|---|---|
| **Tamaño fijo** | Cada N caracteres, con overlap | Default simple, documentos homogéneos |
| **Por oración** | Partir en `. ` `? ` `! ` | Cuando cada oración es autocontenida |
| **Por párrafo** | Partir en `\n\n` | Documentos bien estructurados |
| **Recursivo** | Probar `\n\n` → `\n` → `. ` → espacio | LangChain default, el más robusto |

Diagrama ASCII mostrando chunking fijo con overlap vs chunking por párrafo.

**Overlap:** repetir N caracteres entre chunks para no cortar ideas a la mitad.
Regla práctica: overlap = 10-20% del tamaño del chunk.

### Código — Chunking

```python
DOCUMENTOS = [
    {"id": "doc_arquitectura", "titulo": "Arquitectura de sistemas con IA",
     "contenido": "Integrar modelos de inteligencia artificial en una arquitectura de software..."},
    {"id": "doc_testing", "titulo": "Testing de software con IA",
     "contenido": "La inteligencia artificial está transformando el testing de software..."},
    {"id": "doc_vectordb", "titulo": "Bases de datos vectoriales",
     "contenido": "Las bases de datos vectoriales almacenan y buscan datos representados..."},
    {"id": "doc_seguridad", "titulo": "Seguridad en aplicaciones de IA",
     "contenido": "Las aplicaciones que integran LLMs introducen nuevos vectores de ataque..."},
    {"id": "doc_mlops", "titulo": "MLOps y deploy de modelos",
     "contenido": "MLOps aplica prácticas de DevOps al ciclo de vida de modelos de ML..."},
]

def chunk_fixed(texto, chunk_size=200, overlap=50):
    """Chunking por tamaño fijo con overlap."""
    chunks = []
    start = 0
    while start < len(texto):
        end = start + chunk_size
        chunks.append(texto[start:end])
        start += chunk_size - overlap
    return chunks

def chunk_por_oracion(texto):
    """Chunking por oración."""
    import re
    oraciones = re.split(r'(?<=[.!?])\s+', texto)
    return [o.strip() for o in oraciones if o.strip()]

# Demostrar ambas estrategias sobre el primer documento
# Para el resto de la clase usamos chunk_por_oracion
```

**ChromaDB — vector store en memoria**

```
  Base relacional (SQL):              Vector store:
  ─────────────────────               ─────────────
  SELECT * FROM docs                  "buscar los 3 chunks
  WHERE categoria = 'IA'              más parecidos a esta query"
  → coincidencia EXACTA               → similitud APROXIMADA

  Busca por: igualdad, rango          Busca por: distancia en espacio N-dim
  Índice: B-tree                      Índice: HNSW
```

ChromaDB:
- Open source, Python nativo
- Funciona **in-memory** — sin instalar nada
- Perfecto para prototipos y clases
- En producción: Pinecone, Weaviate, Qdrant, pgvector

| Config | Valor que usamos | Por qué |
|---|---|---|
| Distancia | `cosine` | Misma métrica que Clase 1 |
| Embedding function | `sentence-transformers` manual | Control total, ya lo conocemos |
| Metadata | `titulo`, `doc_id`, `chunk_index` | Para saber de dónde viene cada chunk |

### Código — Indexar en ChromaDB

```python
import chromadb

client = chromadb.Client()  # in-memory
collection = client.create_collection(
    name="clase3_docs",
    metadata={"hnsw:space": "cosine"}
)

# Preparar chunks de TODOS los documentos
all_chunks, all_ids, all_metadatas = [], [], []

for doc in DOCUMENTOS:
    chunks = chunk_por_oracion(doc["contenido"])
    for i, chunk in enumerate(chunks):
        all_chunks.append(chunk)
        all_ids.append(f'{doc["id"]}_chunk_{i}')
        all_metadatas.append({
            "titulo": doc["titulo"],
            "doc_id": doc["id"],
            "chunk_index": i
        })

all_embeddings = modelo_emb.encode(all_chunks).tolist()

collection.add(
    documents=all_chunks,
    embeddings=all_embeddings,
    metadatas=all_metadatas,
    ids=all_ids
)
```

### Código — Retrieval

```python
def buscar_chunks(query, n_results=3):
    """Busca los top-k chunks más similares a la query."""
    query_embedding = modelo_emb.encode([query]).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=n_results,
        include=["documents", "metadatas", "distances"]
    )
    return results
```

**Augmentation — el prompt que une todo**

El paso clave: construir un prompt que combine chunks recuperados con la pregunta.

```
  ┌─────────────────────────────────────────────────┐
  │  SYSTEM                                         │
  │  "Sos un asistente que responde basándose       │
  │   SOLO en el contexto proporcionado.            │
  │   Si no hay info suficiente, decilo."           │
  ├─────────────────────────────────────────────────┤
  │  USER                                           │
  │  Contexto:                                      │
  │  [1] (Arquitectura) "Los modelos de ML se..."   │
  │  [2] (MLOps) "MLOps aplica prácticas de..."     │
  │  [3] (MLOps) "Los contenedores Docker con..."   │
  │                                                 │
  │  Pregunta: ¿Cómo se despliegan modelos en prod? │
  ├─────────────────────────────────────────────────┤
  │  ASSISTANT                                      │
  │  "Según los documentos, los modelos se..."      │
  └─────────────────────────────────────────────────┘
```

Reglas del prompt RAG:
1. El system prompt **restringe** al LLM a usar solo el contexto dado
2. Los chunks se numeran para que el LLM pueda citar fuentes
3. La pregunta va **al final** (recency bias: el modelo presta más atención al final)

### Código — RAG end-to-end

```python
SYSTEM_RAG = """Sos un asistente técnico que responde preguntas basándose ÚNICAMENTE 
en el contexto proporcionado. 

Reglas:
- Usá solo la información del contexto para responder.
- Si el contexto no tiene información suficiente, decí "No tengo información suficiente en los documentos proporcionados."
- Citá la fuente entre corchetes cuando sea posible, ej: [Arquitectura de sistemas con IA].
- Respondé en español, de forma concisa (máximo 4-5 oraciones)."""

def rag_query(pregunta, n_chunks=3, verbose=True):
    """Pipeline RAG completo: retrieval → augmentation → generation."""
    # 1. Retrieval
    results = buscar_chunks(pregunta, n_results=n_chunks)

    # 2. Augmentation — construir contexto numerado
    contexto_partes = []
    for i in range(len(results['documents'][0])):
        doc = results['documents'][0][i]
        titulo = results['metadatas'][0][i]['titulo']
        contexto_partes.append(f'[{i+1}] ({titulo}): {doc}')
    contexto = "\n\n".join(contexto_partes)

    # 3. Generation
    messages = [
        {"role": "system", "content": SYSTEM_RAG},
        {"role": "user", "content": f"Contexto:\n{contexto}\n\nPregunta: {pregunta}"}
    ]
    respuesta = llamar_llm(messages, temperature=0.3)

    if verbose:
        print(f'Pregunta: {pregunta}')
        print(f'\n📎 Chunks recuperados:')
        for i, parte in enumerate(contexto_partes):
            sim = 1 - results['distances'][0][i]
            print(f'  {parte[:100]}... (sim: {sim:.3f})')
        print(f'\n💬 Respuesta RAG:')
        print(respuesta)

    return respuesta, results

# Probar el pipeline
respuesta, _ = rag_query("¿Qué es el prompt injection y cómo se defiende?")
```

### Ejercicios
1. Ejecutar chunking y observar las tres estrategias (5')
2. Indexar en ChromaDB y verificar count (3')
3. Retrieval: probar query y observar similitudes (5')
4. RAG end-to-end: correr la demo guiada (5')
5. **Ejercicio libre:** probar queries variadas, cambiar n_chunks de 1 a 5, probar query sin respuesta ("¿Cuál es la capital de Francia?") (8')

---

## Bloque 3 — Hybrid Search (30 min)

### Contenido teórico

**BM25 — recap rápido**

En Clase 1 vimos BoW y TF-IDF. **BM25** es la evolución de TF-IDF para búsqueda:

```
  BM25(query, doc) = Σ  IDF(término) × TF_saturada(término, doc) × normalización_largo
                    término∈query
```

Mejoras sobre TF-IDF:
- **Saturación de TF:** la 10ma aparición de una palabra no suma tanto como la 1ra
- **Normalización por largo:** docs cortos no se penalizan injustamente

**¿Cuándo falla la búsqueda semántica?**

| Query | Búsqueda semántica | BM25 |
|---|---|---|
| "ChromaDB" | Busca conceptos similares a DBs vectoriales ✓ | Busca la palabra exacta "ChromaDB" ✓✓ |
| "HNSW" | No entiende la sigla, devuelve ruido ✗ | Encuentra el chunk que dice "HNSW" ✓ |
| "seguridad en IA" | Entiende el concepto amplio ✓✓ | Solo matchea si dice "seguridad" literal ✓ |
| "cómo protegerse de ataques" | Entiende la intención ✓✓ | No matchea "prompt injection" ✗ |

**Conclusión:** BM25 es mejor para términos técnicos y siglas. Semántica es mejor para conceptos. Lo ideal: combinarlos.

### Código — BM25

```python
from rank_bm25 import BM25Okapi
import re

def tokenize_simple(text):
    """Tokenización simple para BM25."""
    return re.findall(r'\w+', text.lower())

corpus_tokenizado = [tokenize_simple(chunk) for chunk in all_chunks]
bm25 = BM25Okapi(corpus_tokenizado)

def buscar_bm25(query, n_results=3):
    """Búsqueda BM25 (léxica) sobre los chunks indexados."""
    query_tokens = tokenize_simple(query)
    scores = bm25.get_scores(query_tokens)
    top_indices = np.argsort(scores)[::-1][:n_results]
    results = []
    for idx in top_indices:
        results.append({
            "chunk": all_chunks[idx],
            "score": scores[idx],
            "metadata": all_metadatas[idx]
        })
    return results

# Comparar: BM25 vs semántica para una sigla técnica
query_tecnica = "HNSW"
# → BM25 encuentra el chunk exacto, semántica devuelve ruido
```

### Código — Hybrid Search

```python
def hybrid_search(query, n_results=3, alpha=0.5):
    """
    Búsqueda híbrida: combina BM25 (léxico) + semántico (vectores).
    alpha=1.0 → solo semántico, alpha=0.0 → solo BM25.
    """
    # BM25 scores
    query_tokens = tokenize_simple(query)
    bm25_scores = bm25.get_scores(query_tokens)
    bm25_norm = bm25_scores / bm25_scores.max() if bm25_scores.max() > 0 else bm25_scores

    # Semántico scores
    query_emb = modelo_emb.encode([query]).tolist()
    sem_results = collection.query(
        query_embeddings=query_emb,
        n_results=len(all_chunks),
        include=["distances"]
    )
    sem_scores = np.zeros(len(all_chunks))
    for i, idx_id in enumerate(sem_results['ids'][0]):
        pos = all_ids.index(idx_id)
        sem_scores[pos] = 1 - sem_results['distances'][0][i]
    sem_norm = sem_scores / sem_scores.max() if sem_scores.max() > 0 else sem_scores

    # Combinar: weighted sum
    hybrid_scores = alpha * sem_norm + (1 - alpha) * bm25_norm

    top_indices = np.argsort(hybrid_scores)[::-1][:n_results]
    results = []
    for idx in top_indices:
        results.append({
            "chunk": all_chunks[idx],
            "hybrid_score": hybrid_scores[idx],
            "sem_score": sem_norm[idx],
            "bm25_score": bm25_norm[idx],
            "metadata": all_metadatas[idx]
        })
    return results
```

### Ejercicios
1. BM25 vs semántico sobre sigla técnica: "HNSW", "gRPC", "INT8", "DVC" (5')
2. Semántico vs BM25 sobre concepto amplio: "cómo proteger una aplicación de IA" (3')
3. **Ejercicio libre:** comparar los tres métodos, variar alpha de 0.0 a 1.0 (8')

---

## Bloque 4 — RAG avanzado (25 min, conceptual + demos)

### Contenido teórico

**Reranking — segunda pasada de precisión**

Problema: el retrieval trae top-k, pero el orden no siempre es óptimo.

```
  Pipeline naive:                    Con reranking:
  ───────────────                    ──────────────
  Query → Vector Store → Top 10     Query → Vector Store → Top 10
                          ↓                                  ↓
                        LLM ✗                        Cross-encoder
                                                     reordena → Top 3
                                                          ↓
                                                        LLM ✓
```

Cross-encoder vs Bi-encoder:

| | Bi-encoder (lo que usamos) | Cross-encoder (reranker) |
|---|---|---|
| Input | Query y doc por separado | Query + doc juntos |
| Velocidad | Rápido (embeddings pre-calculados) | Lento (procesa cada par) |
| Precisión | Buena | Mejor |
| Uso | Retrieval inicial (miles de docs) | Reranking (top 10-20 candidatos) |

```
  Bi-encoder:    embed(query)  →  comparar con  ←  embed(doc)   [rápido, menos preciso]
  Cross-encoder: model(query + doc)  →  score de relevancia     [lento, más preciso]
```

Modelo: `cross-encoder/ms-marco-MiniLM-L-6-v2` (~80MB)

### Código — Demo reranking

```python
from sentence_transformers import CrossEncoder

reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

query = "¿Cómo se monitorea un modelo en producción?"

# Paso 1: retrieval amplio (top 10)
results = buscar_chunks(query, n_results=10)
candidatos = results['documents'][0]

# Paso 2: reranking con cross-encoder
pares = [[query, doc] for doc in candidatos]
scores_rerank = reranker.predict(pares)
ranking = np.argsort(scores_rerank)[::-1]

# Mostrar antes vs después
```

**HyDE — Hypothetical Document Embeddings**

Problema: la query del usuario es corta; los documentos son largos. El embedding de "cómo monitorear un modelo" no se parece al embedding de un párrafo técnico sobre MLOps.

Idea: pedirle al LLM que **invente** un documento hipotético que responda la pregunta, y buscar con ESE embedding.

```
  Pipeline normal:     Query → embed(query) → buscar → chunks
  Pipeline HyDE:       Query → LLM genera doc hipotético
                                    ↓
                              embed(doc_hipotético) → buscar → chunks
```

¿Por qué funciona? El embedding del documento hipotético está más cerca del "espacio" de los documentos reales que el embedding de la query corta.

Costo: una llamada extra al LLM por cada query (latencia + tokens).

### Código — Demo HyDE

```python
def hyde_search(query, n_results=3):
    """Búsqueda con HyDE: genera doc hipotético y busca con su embedding."""
    # Paso 1: generar documento hipotético
    messages = [
        {"role": "system",
         "content": "Escribí un párrafo técnico de 3-4 oraciones que responda la pregunta. "
                    "No inventes datos específicos. Escribí como si fuera un fragmento de "
                    "documentación técnica."},
        {"role": "user", "content": query}
    ]
    doc_hipotetico = llamar_llm(messages, temperature=0.5)

    # Paso 2: buscar con el embedding del doc hipotético
    hyde_embedding = modelo_emb.encode([doc_hipotetico]).tolist()
    results = collection.query(
        query_embeddings=hyde_embedding,
        n_results=n_results,
        include=["documents", "metadatas", "distances"]
    )
    return results, doc_hipotetico

# Comparar normal vs HyDE
query = "¿Cómo se manejan los cambios en los datos después del deploy?"
```

**Parent-child chunks — contexto completo (solo concepto)**

Problema: si indexás chunks chicos, el retrieval es preciso pero pierde contexto. Si indexás chunks grandes, el retrieval es impreciso.

Solución: indexar chunks chicos (hijos) pero recuperar el chunk grande (padre).

```
  Documento original:
  ┌────────────────────────────────────────────────────┐
  │  Párrafo completo sobre seguridad en IA (padre)    │
  │  ┌──────────┐ ┌──────────┐ ┌──────────┐           │
  │  │ Oración 1│ │ Oración 2│ │ Oración 3│  (hijos)  │
  │  │ (indexar) │ │ (indexar) │ │ (indexar) │           │
  │  └──────────┘ └──────────┘ └──────────┘           │
  └────────────────────────────────────────────────────┘

  Query → match con Oración 2 (hijo)
       → devolver Párrafo completo (padre)
       → el LLM tiene más contexto para responder
```

Implementación: cada chunk hijo guarda un `parent_id` en metadata. Al recuperar, se busca por hijo pero se devuelve el padre.

Trade-off: más contexto para el LLM (mejor respuesta) vs más tokens por chunk (más costo, posible ruido).

> Esta técnica se implementa fácilmente con LangChain `ParentDocumentRetriever`. La vemos en detalle en Clase 4 si hace falta para el TP Integrador.

Sin ejercicio libre en B4 — las demos son guiadas.

---

## Bloque 5 — Evaluación de RAG (15 min, conceptual + bonus)

### Contenido teórico

**¿Por qué necesitamos métricas específicas?**

Un RAG puede fallar en **dos puntos distintos**:

```
  Query → Retrieval → Chunks → LLM → Respuesta
              │                  │
              ▼                  ▼
         ¿Trajo los chunks   ¿Usó bien los
          correctos?          chunks para
                              responder?
```

| El retrieval es... | La generación es... | Resultado |
|---|---|---|
| Bueno ✓ | Buena ✓ | Respuesta correcta |
| Bueno ✓ | Mala ✗ | Tiene la info pero la ignora o distorsiona |
| Malo ✗ | Buena ✓ | Respuesta coherente pero sobre chunks incorrectos |
| Malo ✗ | Mala ✗ | Desastre total |

Necesitamos evaluar **retrieval** y **generación** por separado.

**RAGAS — framework de evaluación de RAG**

| Métrica | Qué evalúa | Pregunta clave | Rango |
|---|---|---|---|
| **Faithfulness** | Generación | ¿La respuesta está soportada por los chunks? | 0-1 |
| **Answer Relevance** | Generación | ¿La respuesta responde la pregunta? | 0-1 |
| **Context Precision** | Retrieval | ¿Los chunks relevantes están primeros? | 0-1 |
| **Context Recall** | Retrieval | ¿Se recuperaron todos los chunks necesarios? | 0-1 |

```
  Faithfulness = de todas las afirmaciones de la respuesta,
                 ¿cuántas se pueden verificar en los chunks?

  Ejemplo:
    Respuesta: "Los modelos se despliegan como microservicios con Docker."
    Chunk dice: "Los modelos se despliegan como microservicios."
    Chunk NO dice nada de Docker.

    Faithfulness = 1/2 = 0.5  (una afirmación verificable, una no)
```

```
  Answer Relevance = ¿la respuesta responde lo que se preguntó?

  Query: "¿Qué métrica de distancia se usa para texto?"
  Respuesta A: "Para texto se usa coseno." → relevance alta
  Respuesta B: "ChromaDB es una base vectorial." → relevance baja (no responde)
```

> RAGAS usa un LLM como juez para calcular estas métricas. Esto tiene su propia limitación: el juez también puede equivocarse.

### Código — Bonus: demo evaluación manual tipo RAGAS

```python
# 🔽 Bonus: evaluación manual inspirada en RAGAS

def evaluar_faithfulness_manual(pregunta, respuesta, chunks):
    """Usa el LLM como juez para evaluar faithfulness."""
    chunks_text = "\n".join([f"[{i+1}] {c}" for i, c in enumerate(chunks)])
    messages = [
        {"role": "system",
         "content": """Sos un evaluador de calidad de respuestas RAG.
Tu tarea: verificar si CADA afirmación de la respuesta está soportada por los chunks.

Respondé SOLO con un JSON:
{
  "afirmaciones": ["afirmación 1", "afirmación 2", ...],
  "soportadas": [true/false, true/false, ...],
  "faithfulness": 0.XX
}"""},
        {"role": "user",
         "content": f"Chunks:\n{chunks_text}\n\nRespuesta a evaluar:\n{respuesta}"}
    ]
    return llamar_llm(messages, temperature=0.1)

# Ejecutar una evaluación sobre la respuesta de rag_query()
pregunta = "¿Cómo se despliegan modelos de IA en producción?"
respuesta_rag, results = rag_query(pregunta, verbose=False)
chunks_usados = results['documents'][0]
evaluacion = evaluar_faithfulness_manual(pregunta, respuesta_rag, chunks_usados)
```

> En producción se usa el framework RAGAS completo con datasets de evaluación: `pip install ragas` → `ragas.evaluate(dataset)`.

---

## Archivos generados

| Archivo | Descripción | Usado en |
|---|---|---|
| Colección ChromaDB in-memory | Vector store con chunks de 5 documentos | Solo durante la clase (no persiste) |
| Pipeline `rag_query()` | Función RAG completa reutilizable | Referencia para TP Integrador |
| Pipeline `hybrid_search()` | Función de búsqueda híbrida | Referencia para TP Integrador |

---

## Stack de la clase

```
sentence-transformers · chromadb · rank_bm25 · groq · numpy
```

---

## Referencias

### Libros
- Jurafsky, D. & Martin, J. H. (2024). *Speech and Language Processing* (3rd ed. draft). Cap. 14 — Question Answering.  
  https://web.stanford.edu/~jurafsky/slp3/

### Papers
- Lewis, P., Perez, E., Piktus, A., et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. *NeurIPS 2020*.  
  https://arxiv.org/abs/2005.11401
- Robertson, S. & Zaragoza, H. (2009). The Probabilistic Relevance Framework: BM25 and Beyond. *Foundations and Trends in IR*.  
  https://dl.acm.org/doi/10.1561/1500000019
- Gao, L., Ma, X., Lin, J. & Callan, J. (2023). Precise Zero-Shot Dense Retrieval without Relevance Labels (HyDE). *ACL 2023*.  
  https://arxiv.org/abs/2212.10496
- Es, S., James, J., Espinosa-Anke, L. & Schockaert, S. (2023). RAGAS: Automated Evaluation of Retrieval Augmented Generation. *EMNLP 2023*.  
  https://arxiv.org/abs/2309.15217
- Liu, N. F., et al. (2023). Lost in the Middle: How Language Models Use Long Contexts. *TACL 2023*.  
  https://arxiv.org/abs/2307.03172

### Documentación
- ChromaDB. https://docs.trychroma.com/
- Sentence-Transformers. https://www.sbert.net/
- RAGAS. https://docs.ragas.io/
