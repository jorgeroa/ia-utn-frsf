# Plan de Curso — Inteligencia Artificial
**UTN-FRSF · Ingeniería en Sistemas de Información**  
**Dr. Jorge Roa · Dra. María de los Milagros Gutiérrez**

---

## Resumen general

| Clase | Tema | Pregunta central |
|---|---|---|
| 1 | NLP, Tokenización y Vectorización | ¿Cómo representa texto una máquina? |
| 2 | LLMs — Arquitectura, Ecosistema y Prompting | ¿Cómo genera texto un modelo? |
| 3 | RAG — Retrieval Augmented Generation | ¿Cómo le doy conocimiento propio? |
| 4 | Agentes, Multiagentes y Deep Agents | ¿Cómo actúa de forma autónoma? |
| Bonus | Ontologías, Graph DBs y GraphRAG/LightRAG | Se da si hay tiempo |

**TP Integrador:** agente RAG funcional sobre un dominio a elección — código + notebook + demo en vivo + informe de decisiones técnicas.

**Stack tecnológico del curso:** Ollama (qwen3:8b / qwen3:4b) · Groq API (fallback) · HuggingFace · ChromaDB · LangChain · Python

---

## Clase 1 — NLP, Tokenización y Vectorización

**Pregunta central:** ¿Cómo representa texto una máquina?

### Bloque 1 — Introducción al ecosistema
- IA, ML, DL — las cajas chinas
- Open source vs closed source — justificación del stack del curso
- Discriminativo vs Generativo — énfasis en generativos
- La explosión 2013 → 2025: Word2Vec · Transformer · BERT · GPT-3 · ChatGPT · Agentes
- El pipeline de un LLM — caja negra: tokenización → embeddings → modelo → output
- TP Integrador — presentación y entregables

### Bloque 2 — Representaciones clásicas
- **Bag of Words:** intuición, matriz de conteos, limitaciones (sinonimia, orden)
- **TF-IDF:** frecuencia × rareza, tabla comparativa sobre corpus propio
- Conector: BM25 como mejora de TF-IDF → hybrid search en Clase 3

*Práctica:* `CountVectorizer` + `TfidfVectorizer` sobre corpus propio

### Bloque 3 — Tokenización
- Comparación: por palabras / caracteres / subpalabras
- **BPE** (Byte Pair Encoding) — los 4 pasos + implementación manual en ~20 líneas
- **WordPiece** (BERT) y **SentencePiece** (Llama 2, T5)
- Costo del idioma: español vs inglés con `tiktoken` (o200k_base, sin API key)

*Práctica:* NLTK, spaCy, BPE manual, experimento multilingüe con tiktoken

### Bloque 4 — Embeddings y similitud semántica
- Limitación de Word2Vec: una palabra = un vector sin contexto
- Sentence Transformers: `paraphrase-multilingual-MiniLM-L12-v2` (384D)
- Similitud coseno: score entre -1 y 1
- Visualización matplotlib: clusters semánticos + analogía rey−hombre+mujer≈reina

*Práctica:* generar embeddings del corpus → guardar `corpus_embeddings_clase1.npy` para Clase 3

### Bloque 5 — Visualización con UMAP
- Reducción 384D → 2D preservando estructura local
- Scatter plot con 4 categorías coloreadas (NLP, LLMs, RAG, Agentes)

*Práctica:* UMAP + matplotlib, ejercicio con oraciones propias

### Referencias — Clase 1
- Jurafsky & Martin (2024). *Speech and Language Processing* (3rd ed.). https://web.stanford.edu/~jurafsky/slp3/
- Mikolov et al. (2013). Word2Vec. https://arxiv.org/abs/1301.3781
- Sennrich et al. (2016). BPE / Subword NMT. https://arxiv.org/abs/1508.07909
- Devlin et al. (2018). BERT. https://arxiv.org/abs/1810.04805
- Reimers & Gurevych (2019). Sentence-BERT. https://arxiv.org/abs/1908.10084
- McInnes et al. (2018). UMAP. https://arxiv.org/abs/1802.03426

---

## Clase 2 — LLMs: Arquitectura, Ecosistema y Prompting

**Pregunta central:** ¿Cómo genera texto un modelo?

### Bloque 1 — Tokenizers en modelos reales
- Los mismos algoritmos de Clase 1 aplicados en modelos reales:
  - GPT-4o → `tiktoken` / `o200k_base` (~200K vocab)
  - Llama 3.1 → `tiktoken` / `cl100k_base` (~128K vocab)
  - BERT multilingual → `transformers` / WordPiece (~120K vocab)
  - Qwen2.5 / Qwen3 → BPE tiktoken-based (~150K vocab)
- Comparación de cantidad de tokens por modelo e idioma
- Tabla de costos por 1M tokens (GPT-5.4, Claude, Gemini, Qwen3, Ollama)

*Práctica:* comparación en vivo con tiktoken y transformers sobre la misma frase

### Bloque 2 — Arquitectura Transformer
- Motivación: limitación de RNN/LSTM — olvido en secuencias largas
- **Attention — intuición:** "el banco quebró porque estaba podrido" → cómo el modelo resuelve polisemia
- **Q, K, V** sin matemática pesada — diagrama ASCII: qué busco / qué tengo / qué devuelvo
- **Encoder vs Decoder vs Encoder-Decoder** — tabla comparativa con ejemplos
- **Escala:** qué es un parámetro, evolución GPT-2 → GPT-3 → Llama → Qwen3 → Kimi K2.5
- **MoE (Mixture of Experts):** modelo denso vs MoE, cómo funciona el router

*Bonus tracks (si da el tiempo):*
- Heatmap de attention weights con matplotlib (6 cabezas, capa 0, BertViz)
- Fórmula completa: `Attention(Q,K,V) = softmax(QKᵀ/√dk) · V` desglosada paso a paso

*Práctica:* visualización de attention weights con BertViz o matplotlib

### Bloque 3 — Ciclo de vida del modelo
- **Pre-training:** predecir el siguiente token, el modelo "sabe el idioma"
- **SFT (Supervised Fine-Tuning):** pares pregunta→respuesta curados, el modelo sigue instrucciones
- **RLHF / DPO:** preferencias humanas, el modelo responde bien
- **Base model vs Instruct model:** demo en vivo con Ollama

*Práctica:* demo base vs instruct con qwen3:8b via Ollama

### Bloque 4 — Ecosistema actual (abril 2026)

**Closed source:**

| Modelo | Organización | Contexto | Fortaleza |
|---|---|---|---|
| GPT-5.4 | OpenAI | 200K | General, multimodal |
| Claude Opus 4.6 | Anthropic | 200K | Coding, reasoning |
| Claude Sonnet 4.6 | Anthropic | 200K | Balance costo/calidad |
| Gemini 3.1 Pro | Google | 1M | Multimodal, contexto gigante |
| Grok 4 | xAI | 1M | Datos en tiempo real |

**Open weight:**

| Modelo | Org | Params activos | Local | Fortaleza |
|---|---|---|---|---|
| Llama 4 Scout | Meta | 17B (MoE 109B) | parcial | Contexto 10M tokens |
| Qwen3 / Qwen3.5 | Alibaba | 4B–72B | ✓ Ollama | Multilingüe, liviano |
| Kimi K2.5 | Moonshot AI | 32B (MoE 1T) | ✗ | Agentes, coding, multimodal |
| Mistral Small 4 | Mistral | 22B | ✓ Ollama | Eficiente |
| Gemma 4 | Google | 4B–27B | ✓ Ollama | On-device |
| DeepSeek-R1 | DeepSeek | varios | parcial | Razonamiento |

- **Benchmarks:** qué miden y qué NO miden
- Gráfico matplotlib: calidad vs costo relativo
- **Recomendación del curso:** `qwen3:8b` o `qwen3:4b` via Ollama

### Bloque 5 — Prompting como ingeniería
- **System prompt:** estructura de una llamada, qué define (rol, restricciones, formato)
- **4 técnicas fundamentales:**
  - Zero-shot: sin ejemplos
  - Few-shot: 2-3 ejemplos en el prompt
  - Chain of Thought (CoT): razonar paso a paso
  - Structured output: forzar respuesta en JSON schema
- **Context window:** límites, "Lost in the Middle", por qué aparece RAG

*Práctica:* wrapper `llamar_llm()` (Ollama + Groq), las 4 técnicas con ejemplos de dominio de sistemas, ejercicio de diseño de prompt

### Referencias — Clase 2
- Vaswani et al. (2017). Attention Is All You Need. https://arxiv.org/abs/1706.03762
- Brown et al. (2020). GPT-3 / Few-Shot Learners. https://arxiv.org/abs/2005.14165
- Ouyang et al. (2022). InstructGPT / RLHF. https://arxiv.org/abs/2203.02155
- Rafailov et al. (2023). DPO. https://arxiv.org/abs/2305.18290
- Wei et al. (2022). Chain-of-Thought Prompting. https://arxiv.org/abs/2201.11903
- Shazeer et al. (2017). Mixture of Experts. https://arxiv.org/abs/1701.06538
- Liu et al. (2023). Lost in the Middle. https://arxiv.org/abs/2307.03172
- Moonshot AI (2026). Kimi K2.5. https://github.com/MoonshotAI/Kimi-K2.5

---

## Clase 3 — RAG: Retrieval Augmented Generation

**Pregunta central:** ¿Cómo le doy conocimiento propio a un LLM sin reentrenarlo?

### Bloque 1 — ¿Por qué RAG?
- Limitaciones del LLM puro: knowledge cutoff, alucinaciones, context window
- Los embeddings del corpus de Clase 1 (`corpus_embeddings_clase1.npy`) se cargan directamente
- Conector con Clase 2: context window como motivación de RAG

### Bloque 2 — Pipeline RAG naive
```
Documentos → Chunking → Embeddings → Vector Store → Retrieval → Augmentation → LLM → Respuesta
```
- **Chunking:** estrategias (tamaño fijo, por párrafo, por oración), overlap
- **Indexado:** cargar embeddings en ChromaDB
- **Retrieval:** similitud coseno sobre el vector store
- **Augmentation:** construir el prompt con los chunks recuperados

*Práctica:* RAG end-to-end sobre el corpus de Clase 1 con ChromaDB + Ollama

### Bloque 3 — Hybrid Search
- BM25 (léxico) + búsqueda semántica (vectores) → combinación ponderada
- Cuándo usar cada uno: términos técnicos vs conceptos semánticos

*Práctica:* implementar hybrid search con `rank_bm25` + ChromaDB

### Bloque 4 — RAG avanzado
- **Reranking:** cross-encoder para reordenar los resultados recuperados
- **HyDE (Hypothetical Document Embeddings):** generar un documento hipotético para mejorar el retrieval
- **Parent-child chunks:** indexar chunks pequeños pero recuperar el contexto completo del padre

### Bloque 5 — Evaluación de RAG
- **Faithfulness:** ¿la respuesta está respaldada por los documentos recuperados?
- **Answer relevance:** ¿la respuesta responde la pregunta?
- Herramientas: RAGAS framework

*Práctica:* evaluar el pipeline propio con métricas RAGAS

### Referencias — Clase 3
- Lewis et al. (2020). RAG: Retrieval-Augmented Generation. https://arxiv.org/abs/2005.11401
- Robertson & Zaragoza (2009). BM25. https://dl.acm.org/doi/10.1561/1500000019
- Gao et al. (2023). HyDE. https://arxiv.org/abs/2212.10496
- Es et al. (2023). RAGAS. https://arxiv.org/abs/2309.15217
- Jurafsky & Martin (2024). SLP3, Cap. 14. https://web.stanford.edu/~jurafsky/slp3/

---

## Clase 4 — Agentes, Multiagentes y Deep Agents

**Pregunta central:** ¿Cómo actúa un LLM de forma autónoma?

### Bloque 1 — ¿Qué es un agente?
- Definición: percibir → razonar → actuar → observar → iterar
- Diferencia con un LLM puro: herramientas, memoria, planificación
- Taxonomía: agentes simples, multi-step, multiagentes, deep agents

### Bloque 2 — Patrón ReAct
```
Thought → Action → Observation → Thought → Action → ...
```
- Razonamiento y acción intercalados
- Implementación con LangChain + Ollama

*Práctica:* agente ReAct con herramientas (búsqueda, calculadora, lectura de archivos)

### Bloque 3 — Herramientas y MCP
- **Tool calling / Function calling:** cómo el LLM decide qué herramienta usar y con qué argumentos
- **MCP (Model Context Protocol):** estándar abierto de Anthropic para conectar agentes con herramientas externas
- Herramientas comunes: búsqueda web, ejecución de código, lectura de archivos, APIs

*Práctica:* agente con herramientas custom via LangChain

### Bloque 4 — Sistemas multiagente
- Orquestador + agentes especializados
- Patrones: secuencial, paralelo, jerárquico
- **Kimi K2.5 Agent Swarm** como ejemplo de estado del arte: hasta 100 agentes en paralelo
- Frameworks: LangGraph, AutoGen, CrewAI

### Bloque 5 — Deep Agents y estado del arte
- **Claude Code, Kimi Code, Gemini CLI:** agentes de terminal
- **Planificación a largo plazo:** gestión de contexto y memoria persistente
- Limitaciones actuales: fiabilidad, costos, latencia

*Práctica:* sistema multiagente simple con LangGraph — orquestador + agente RAG + agente de síntesis

### TP Integrador — demo final
- Presentación de los agentes RAG de cada grupo
- Evaluación: funcionamiento, decisiones técnicas, calidad de retrieval

### Referencias — Clase 4
- Yao et al. (2022). ReAct: Synergizing Reasoning and Acting. https://arxiv.org/abs/2210.03629
- Wang et al. (2023). A Survey on Large Language Model based Autonomous Agents. https://arxiv.org/abs/2308.11432
- Anthropic (2024). Model Context Protocol. https://modelcontextprotocol.io
- Moonshot AI (2026). Kimi K2.5 Agent Swarm. https://github.com/MoonshotAI/Kimi-K2.5

---

## Bonus — Ontologías, Graph DBs y GraphRAG

**Condición:** se da si hay tiempo al final de Clase 4 o como clase adicional.

### Contenido
- Limitaciones del RAG vectorial: no captura relaciones entre entidades
- **Ontologías:** conceptos, relaciones, jerarquías — OWL, RDF
- **Graph Databases:** Neo4j, estructura de nodos y aristas
- **GraphRAG (Microsoft):** extracción de grafo de conocimiento + community summaries + retrieval híbrido
- **LightRAG:** alternativa más liviana a GraphRAG
- Cuándo usar GraphRAG vs RAG vectorial

### Referencias — Bonus
- Edge et al. (2024). From Local to Global: A Graph RAG Approach. https://arxiv.org/abs/2404.16130
- Guo et al. (2024). LightRAG. https://arxiv.org/abs/2410.05779
- Gruber (1993). A Translation Approach to Portable Ontology Specifications. https://doi.org/10.1006/knac.1993.1008

---

## Archivos generados durante el curso

| Archivo | Generado en | Usado en |
|---|---|---|
| `corpus_embeddings_clase1.npy` | Clase 1 — B4 | Clase 3 — vector store |
| `corpus_clase1.csv` | Clase 1 — B4 | Clase 3 — metadata |
| `clusters_umap_clase1.png` | Clase 1 — B5 | — |
| `ecosistema_llm_2026.png` | Clase 2 — B4 | — |
| `attention_heatmap.png` | Clase 2 — B2 bonus | — |
| `vector_store/` (ChromaDB) | Clase 3 — B2 | Clase 3 — B3, B4, B5 |
| `agente_rag.py` | Clase 4 — TP | Demo final |
