# IA · UTN-FRSF

Material de la asignatura **Inteligencia Artificial** de la
Universidad Tecnológica Nacional, Facultad Regional Santa Fe.

**Docentes:** Dr. Jorge Roa · Dra. Milagros Gutiérrez

---

## Contenido

### Clase 1 — Tokenización y Vectorización

- `clase01/vectores_slides.ipynb` — slides de la clase en formato Jupyter (reveal.js).
- `clase01/figures/` — recursos gráficos.

### Próximas clases

- Clase 2 — LLMs
- Clase 3 — RAG
- Clase 4 — Agentes

---

## Generar las slides

La notebook está pensada para presentarse como reveal.js. Para
exportarla a HTML:

```bash
cd clase01
jupyter nbconvert --to slides vectores_slides.ipynb
```

Eso genera `vectores_slides.slides.html`, que se abre en cualquier
navegador y se navega con flechas (← → para slides; ↓ ↑ para
subslides opcionales con material de profundización).

Para servirlas en localhost en lugar de descargar el archivo:

```bash
jupyter nbconvert --to slides vectores_slides.ipynb --post serve
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
│       └── embeddings.png
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
