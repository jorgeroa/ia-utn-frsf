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

### Próximas clases

- Clase 3 — RAG
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

### Dependencias mínimas

```bash
pip install jupyter nbconvert
```

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
