# Plan de Clase 2 — LLMs: Arquitectura, Ecosistema y Prompting
**UTN-FRSF · Ingeniería en Sistemas de Información · Inteligencia Artificial**  
**Dr. Jorge Roa · Dra. María de los Milagros Gutiérrez**

---

**Pregunta central:** ¿Cómo genera texto un modelo?

**Formato:** Notebook Jupyter con metadata de slideshow (nbconvert + Reveal.js)  
**Comando para slides:** `jupyter nbconvert --to slides clase2_slides.ipynb --post serve`

---

## Recap — conexión con Clase 1

| Lo que ya sabemos | Lo que vemos hoy |
|---|---|
| Tokenizers: BPE, WordPiece, SentencePiece | Los mismos en GPT-4o, Llama, BERT |
| Embeddings: vectores de 384 dimensiones | Cómo el Transformer los procesa |
| BoW / TF-IDF: representación ingenua | Por qué attention lo supera |

Los embeddings de `corpus_embeddings_clase1.npy` se vuelven a usar en Clase 3.

---

## Bloque 1 — Tokenizers en modelos reales

### Contenido teórico

**Los mismos algoritmos, vocabularios distintos**

| Modelo | Algoritmo | Vocab | Librería |
|---|---|---|---|
| GPT-4o | BPE `o200k_base` | ~200K | `tiktoken` — local |
| Llama 3.1 | BPE `cl100k_base` | ~128K | `tiktoken` — local |
| BERT multilingual | WordPiece | ~120K | `transformers` — local |
| Qwen2.5 / Qwen3 | BPE tiktoken-based | ~150K | `tiktoken` / `transformers` |

El tokenizer es independiente del modelo — corre 100% local, sin API key.

**Tokens y costo — tabla de precios (abril 2026)**

| Modelo | Input / 1M tokens | Output / 1M tokens |
|---|---|---|
| GPT-5.4 | ~$15 | ~$60 |
| Claude Opus 4.6 | ~$15 | ~$75 |
| Claude Sonnet 4.6 | ~$3 | ~$15 |
| Gemini 3.1 Pro | ~$3.5 | ~$10 |
| Qwen3 8B (Groq) | ~$0.10 | ~$0.10 |
| Ollama local | $0 | $0 |

Escribir en español puede costar 30-50% más que en inglés (más tokens por oración).

### Ejercicios
1. Comparar cantidad de tokens por modelo e idioma sobre la misma frase (5')
2. Ver cómo parte cada tokenizer las mismas palabras — diferencias entre BPE y WordPiece (5')
3. **Opcional:** Qwen tokenizer via HuggingFace (requiere autenticación)

### Código
```python
import tiktoken
from transformers import AutoTokenizer

enc_gpt4o  = tiktoken.get_encoding('o200k_base')
enc_llama3 = tiktoken.get_encoding('cl100k_base')
tok_bert   = AutoTokenizer.from_pretrained('bert-base-multilingual-cased')
```

---

## Bloque 2 — Arquitectura Transformer

### Contenido teórico

**Motivación: limitación de RNN/LSTM**
- Procesaban el texto de izquierda a derecha, secuencialmente
- Problema: al llegar a la palabra 50, el modelo "olvidaba" la palabra 1
- Transformer lo resuelve: procesa todo en paralelo

**Attention — intuición**

Oración: *"el banco quebró porque estaba podrido"*

```
banco  ←─────────────────────── podrido  (muy relevante: madera → mueble)
banco  ←───────────── quebró            (relevante)
banco  ←──── el                         (poco relevante)
```

"Podrido" tiene alto peso → `banco = mueble`, no institución. Resuelve la polisemia que BoW y Word2Vec no podían manejar.

**Q, K, V — diagrama ASCII**
```
Para cada token el modelo genera tres vectores:

Q (Query)  = "¿qué estoy buscando?"
K (Key)    = "¿qué información tengo?"
V (Value)  = "¿qué voy a devolver si me buscan?"

┌──────────┐     ┌──────────┐     ┌──────────────────────┐
│  Query   │     │   Keys   │     │  Values              │
│ "banco"  │ ──▶ │ todos    │ ──▶ │ combinación ponderada│
│ ¿qué soy?│     │ los tok. │     │ → representación     │
└──────────┘     └──────────┘     │   enriquecida        │
                                   └──────────────────────┘
```

**Encoder vs Decoder vs Encoder-Decoder**

| Arquitectura | Para qué sirve | Ejemplos | Contexto |
|---|---|---|---|
| Solo Encoder | Entender texto (clasificación, embeddings) | BERT, RoBERTa | Bidireccional |
| Solo Decoder | Generar texto (siguiente token) | GPT, Llama, Qwen, Claude | Solo el pasado |
| Encoder-Decoder | Transformar texto (traducción, resumen) | T5, BART, mT5 | Mixto |

El curso usa modelos **Decoder** — generan texto.

**Escala y MoE**
```
GPT-2    (2019): 1.5B parámetros
GPT-3    (2020): 175B parámetros
Llama 3  (2024): 8B / 70B / 405B
Qwen3    (2025): 0.6B → 235B
Kimi K2.5(2026): 1T total / 32B activos (MoE)
```

MoE — Mixture of Experts:
```
Modelo denso: activa TODOS los parámetros por token → costoso
Modelo MoE:   activa solo los "expertos" relevantes → eficiente

Kimi K2.5: 1T parámetros totales, 32B activos por request
→ calidad de modelo grande, costo de modelo pequeño
```

### Bonus tracks (si da el tiempo, slides que bajan con ↓)

**Bonus 1 — Heatmap de attention weights**
- Visualización matplotlib: 6 cabezas de la capa 0 de BERT
- Cada celda (i,j) = cuánto el token i presta atención al token j
- Diferentes cabezas capturan relaciones sintácticas, semánticas, co-referencia

**Bonus 2 — Fórmula completa de Scaled Dot-Product Attention**

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q \cdot K^T}{\sqrt{d_k}}\right) \cdot V$$

Desglose paso a paso:
1. `Q · Kᵀ` — similitud entre query y todas las keys → matriz (seq_len × seq_len)
2. `/ √dk` — escalado para evitar saturación de softmax
3. `softmax(...)` — normalización: los scores suman 1
4. `· V` — combinación ponderada: el output absorbe información del contexto

### Ejercicios
1. Visualización de attention weights con BertViz o matplotlib (10')
2. **Bonus:** interpretar el heatmap — ¿qué captura cada cabeza?

### Código
```python
from bertviz import head_view
from transformers import BertTokenizer, BertModel

model_bert = BertModel.from_pretrained('bert-base-multilingual-cased', output_attentions=True)
# head_view(attention, tokens)  # ejecutar en Jupyter
```

---

## Bloque 3 — Ciclo de vida del modelo

### Contenido teórico

**Los tres estadios**
```
1. PRE-TRAINING
   Datos:    Billones de tokens (internet, libros, código)
   Objetivo: Predecir el siguiente token
   Resultado: El modelo "sabe el idioma"
   Ejemplo:  "La capital de Francia es" → "París, y también es conocida como..."

2. SFT — Supervised Fine-Tuning
   Datos:    Miles de pares (pregunta → respuesta ideal) curados por humanos
   Objetivo: Aprender a seguir instrucciones
   Resultado: El modelo responde preguntas en lugar de completar texto
   Ejemplo:  "¿Cuál es la capital de Francia?" → "París."

3. RLHF / DPO — Alineación con preferencias humanas
   Datos:    Humanos ranquean respuestas alternativas (A mejor que B)
   Objetivo: Aprender qué respuestas prefieren los humanos
   Resultado: El modelo responde de forma más útil, segura y honesta
   RLHF: usa un modelo de recompensa + RL
   DPO:  optimiza directamente sobre los pares preferidos (más simple)
```

**Base model vs Instruct model**
- `Qwen3-8B` → base · `Qwen3-8B-Instruct` → instruct (el que usamos)
- `Llama-3.1-8B` → base · `Llama-3.1-8B-Instruct` → instruct (el que usamos)

### Ejercicios
1. Demo en vivo: base vs instruct con qwen3:8b via Ollama (10')

---

## Bloque 4 — Ecosistema actual (abril 2026)

### Modelos closed source

| Modelo | Organización | Contexto | Fortaleza |
|---|---|---|---|
| GPT-5.4 | OpenAI | 200K | General, reasoning, multimodal |
| Claude Opus 4.6 | Anthropic | 200K | Coding, escritura, reasoning |
| Claude Sonnet 4.6 | Anthropic | 200K | Balance calidad/costo |
| Gemini 3.1 Pro | Google | 1M | Multimodal, contexto gigante |
| Grok 4 | xAI | 1M | Datos en tiempo real |

### Modelos open weight

| Modelo | Org | Params activos | Local | Fortaleza |
|---|---|---|---|---|
| Llama 4 Scout | Meta | 17B (MoE 109B) | parcial | Contexto 10M tokens |
| Qwen3 / Qwen3.5 | Alibaba | 4B–72B | ✓ Ollama | Multilingüe, liviano |
| Kimi K2.5 | Moonshot AI | 32B (MoE 1T) | ✗ | Agentes, coding, multimodal |
| Mistral Small 4 | Mistral | 22B | ✓ Ollama | Eficiente |
| Gemma 4 | Google | 4B–27B | ✓ Ollama | On-device |
| DeepSeek-R1 | DeepSeek | varios | parcial | Razonamiento |

**Kimi K2.5 — mención especial**
- Lanzado enero 2026, Moonshot AI (China), MIT License
- MoE: 1T parámetros totales, 32B activos por request
- Entrenado en 15T tokens visuales + texto (multimodal nativo)
- Agent Swarm: hasta 100 agentes en paralelo (PARL — Parallel Agent RL)
- Usado internamente por Cursor Composer 2

**Benchmarks — qué miden y qué NO miden**
- Sí miden: MMLU (conocimiento), HumanEval/SWE-Bench (código), MATH, HLE
- No miden: utilidad real, consistencia en conversaciones largas, comportamiento en español, costo real, latencia en producción

> Regla práctica: los benchmarks sirven para descartar modelos malos, no para elegir el mejor. El mejor modelo es el que funciona en tu caso de uso con tu presupuesto.

**Recomendación del curso:** `qwen3:8b` o `qwen3:4b` via Ollama

### Ejercicios
1. Visualizar gráfico calidad vs costo (matplotlib) (5')

---

## Bloque 5 — Prompting como ingeniería

### Contenido teórico

**System prompt**
```
┌─────────────────────────────────────────────┐
│  SYSTEM (invisible para el usuario)         │
│  "Sos un asistente experto en derecho..."   │
├─────────────────────────────────────────────┤
│  USER                                       │
│  "¿Qué pasa si no pago el alquiler?"        │
├─────────────────────────────────────────────┤
│  ASSISTANT                                  │
│  "Según el Código Civil, Art. 1222..."      │
└─────────────────────────────────────────────┘
```

Define: rol, restricciones, formato de respuesta, contexto.

**Las 4 técnicas fundamentales**

| Técnica | Cuándo usarla | Costo de tokens |
|---|---|---|
| Zero-shot | Tarea simple, modelo capaz | Mínimo |
| Few-shot | Formato específico, tarea nueva | Medio |
| Chain of Thought (CoT) | Razonamiento, matemática, lógica | Alto |
| Structured output | Integración con código, JSON | Medio |

**Chain of Thought — ejemplo de dominio de sistemas**

Problema: *"Una base de datos tiene 1.000.000 de usuarios. El 0,3% son premium. Cada usuario premium genera 15 eventos/día de 512 bytes. ¿Cuántos GB de logs se generan por semana?"*

- Sin CoT: puede dar respuesta directa incorrecta
- Con CoT: calcula paso a paso → `3.000 × 15 × 7 × 512 bytes = 0.16 GB`

**Context window — el límite que genera RAG**
```
System + Historial + Documento + Query
◄────────── context window ──────────►
Claude: 200K tokens  ≈ 150.000 palabras
Gemini: 1M tokens    ≈ 750.000 palabras
```

Problemas: costo, latencia, "Lost in the Middle" (olvido del medio del contexto), volumen de documentos empresariales.
Esta limitación es exactamente por qué existe RAG — Clase 3.

### Ejercicios
1. Zero-shot: clasificador de sentimientos (4')
2. Few-shot: clasificador de bugs por severidad (5')
3. CoT: problema de logs de sistemas — comparar sin/con CoT (8')
4. Structured output: analizador de requerimientos → JSON (6')
5. **Ejercicio integrador:** diseñar system prompt para revisor de pull requests (10')

### Código
```python
def llamar_llm(messages, model=None, temperature=0.7):
    """Wrapper unificado Ollama / Groq."""
    if USE_OLLAMA:
        import ollama
        resp = ollama.chat(model=model or 'qwen3:8b', messages=messages,
                           options={'temperature': temperature})
        return resp['message']['content']
    else:
        from groq import Groq
        client = Groq(api_key=os.environ.get('GROQ_API_KEY'))
        resp = client.chat.completions.create(
            model=model or 'llama-3.1-8b-instant',
            messages=messages, temperature=temperature)
        return resp.choices[0].message.content
```

---

## Archivos generados

| Archivo | Descripción |
|---|---|
| `ecosistema_llm_2026.png` | Gráfico calidad vs costo de modelos |
| `attention_heatmap.png` | Heatmap de attention weights (bonus) |

---

## Stack de la clase

```
transformers · tiktoken · torch · bertviz · ollama · groq · matplotlib
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
- Vaswani, A., et al. (2017). Attention Is All You Need. *NeurIPS 2017*.  
  https://arxiv.org/abs/1706.03762
- Devlin, J., et al. (2018). BERT. *NAACL 2019*.  
  https://arxiv.org/abs/1810.04805
- Radford, A., et al. (2018). Improving Language Understanding by Generative Pre-Training. *OpenAI*.  
  https://openai.com/index/language-unsupervised/
- Brown, T., et al. (2020). Language Models are Few-Shot Learners (GPT-3). *NeurIPS 2020*.  
  https://arxiv.org/abs/2005.14165
- Ouyang, L., et al. (2022). Training language models to follow instructions with human feedback. *NeurIPS 2022*.  
  https://arxiv.org/abs/2203.02155
- Rafailov, R., et al. (2023). Direct Preference Optimization. *NeurIPS 2023*.  
  https://arxiv.org/abs/2305.18290
- Wei, J., et al. (2022). Chain-of-Thought Prompting Elicits Reasoning in Large Language Models. *NeurIPS 2022*.  
  https://arxiv.org/abs/2201.11903
- Shazeer, N., et al. (2017). Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer. *ICLR 2017*.  
  https://arxiv.org/abs/1701.06538
- Liu, N. F., et al. (2023). Lost in the Middle: How Language Models Use Long Contexts. *TACL 2023*.  
  https://arxiv.org/abs/2307.03172
- Moonshot AI (2026). Kimi K2.5: Visual Agentic Intelligence.  
  https://github.com/MoonshotAI/Kimi-K2.5
