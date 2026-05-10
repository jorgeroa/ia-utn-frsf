# IA · UTN-FRSF

Material de la asignatura **Inteligencia Artificial** de la
Universidad Tecnológica Nacional, Facultad Regional Santa Fe.

**Docentes:** Dr. Jorge Roa · Dra. Milagros Gutiérrez

---

## Contenido

### Clase 1 — Tokenización y Vectorización

- `clase01/vectores_slides.ipynb` — slides de la clase en formato Jupyter (reveal.js).
- `clase01/figures/` — recursos gráficos.

### Clase 2 — LLMs

- `clase02/clase2_slides.ipynb` — slides de la clase en formato Jupyter (reveal.js).
- `clase02/figures/` — figuras SVG.
- `clase02/notebooks/` — notebooks de práctica para Colab (Groq):
  - `01_groq_intro.ipynb` — primera llamada al LLM.
  - `03_sampling_params.ipynb` — temperature, top_p, top_k.
  - `04_prompting_techniques.ipynb` — zero/one/few-shot.
- `clase02/scripts/` — generadores: `build_notebook.py` (slides) y `build_colabs.py` (notebooks).

### Clase 3 — RAG + Evaluación y monitoreo

- `clase03/clase3_slides.ipynb` — slides de la clase (reveal.js).
- `clase03/figures/` — figuras SVG.
- `clase03/notebooks/` — notebooks de práctica para Colab:
  - `01_arize_eval_handson.ipynb` — end-to-end eval + monitoring de un
    chatbot Q&A con Arize AX (OpenTelemetry tracing, LLM-as-judge,
    dashboards, drift simulation).

La clase incluye dos grandes bloques: RAG (B1-B4: pipeline naive,
búsqueda híbrida, RAG avanzado) y un bloque profundo sobre evaluación,
monitoreo y benchmarks (B5: del prompt al producto, organizado por
fase del ciclo).

### Próximas clases

- Clase 4 — Agentes

---

## Generar las slides

Las notebooks están pensadas para presentarse como reveal.js. Para exportar a HTML:

```bash
cd claseXX
jupyter nbconvert --to slides <archivo>.ipynb
```

Esto genera `<archivo>.slides.html`, que se abre en cualquier navegador y se navega con flechas (← → para slides; ↓ ↑ para subslides opcionales).

Para servirlas en localhost:

```bash
jupyter nbconvert --to slides <archivo>.ipynb --post serve
```

### Clase 2 — pipeline de generación

La notebook de slides y las notebooks de Colab se regeneran desde scripts:

```bash
cd clase02
python scripts/build_notebook.py    # regenera clase2_slides.ipynb
python scripts/build_colabs.py      # regenera las notebooks de Colab
jupyter nbconvert clase2_slides.ipynb --to slides --no-input
```

### Clase 3 — pipeline de generación

Las slides de clase 3 contienen celdas de código que se muestran en
las slides (demos de RAG), por lo que se regeneran sin `--no-input`:

```bash
cd clase03
jupyter nbconvert clase3_slides.ipynb --to slides
```

Los notebooks de Colab de clase 3 se editan directamente (no hay
build script todavía).

### Dependencias mínimas

Recomendamos un entorno virtual aislado para no contaminar el Python del
sistema:

```bash
python3 -m venv venv
source venv/bin/activate    # en Windows: venv\Scripts\activate
pip install jupyter nbconvert
```

El directorio `venv/` está en `.gitignore`. Para salir del entorno:
`deactivate`.

---

## Estructura del repositorio

```
.
├── clase01/
│   ├── vectores_slides.ipynb
│   └── figures/
├── clase02/
│   ├── clase2_slides.ipynb
│   ├── clase2_slides.slides.html
│   ├── figures/
│   ├── notebooks/
│   └── scripts/
├── clase03/
│   ├── clase3_slides.ipynb
│   ├── clase3_slides.slides.html
│   ├── figures/
│   └── notebooks/
│       └── 01_arize_eval_handson.ipynb
├── CITATION.cff
├── LICENSE
└── README.md
```

---

## Licencia

Este material se distribuye bajo la licencia
[Creative Commons Attribution 4.0 International (CC BY 4.0)](LICENSE).

Sos libre de compartir, adaptar y reusar el material para cualquier
propósito, incluso comercial, siempre que cites a los autores. Si
querés citar este repo en un trabajo académico, usá el archivo
[`CITATION.cff`](CITATION.cff).
