"""
Genera los 5 notebooks Colab de clase02/notebooks/.
Cada uno con badge "Open in Colab" apuntando a main.

Para regenerar:
    python clase02/scripts/build_colabs.py
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent  # clase02/
NB_DIR = ROOT / "notebooks"
NB_DIR.mkdir(exist_ok=True)

REPO = "jorgeroa/ia-utn-frsf"
BRANCH = "main"


def colab_badge(notebook_filename: str) -> str:
    path = f"clase02/notebooks/{notebook_filename}"
    return (
        f'<a href="https://colab.research.google.com/github/{REPO}/blob/{BRANCH}/{path}" '
        f'target="_blank">'
        f'<img src="https://colab.research.google.com/assets/colab-badge.svg" '
        f'alt="Open In Colab"/></a>'
    )


def md(content: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in content.splitlines()],
    }


def code(content: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": [line + "\n" for line in content.splitlines()],
    }


def write_notebook(filename: str, cells: list):
    notebook = {
        "cells": cells,
        "metadata": {
            "colab": {"provenance": []},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    out = NB_DIR / filename
    out.write_text(json.dumps(notebook, indent=1, ensure_ascii=False))
    print(f"OK {out.name}")


# ─────────────────────────────────────────────────────────────────────────────
# 01 — Groq intro
# ─────────────────────────────────────────────────────────────────────────────

write_notebook("01_groq_intro.ipynb", [
    md(f"""\
# 01 — Hola Groq desde Python

{colab_badge("01_groq_intro.ipynb")}

**Objetivo.** Conectarse a un LLM por código (no por UI) y observar tres comportamientos básicos:
1. Una consulta simple.
2. El efecto de `temperature` en la respuesta.
3. El efecto de un `system` prompt en el comportamiento.

**Requisitos.**
- API key gratuita en https://console.groq.com (registrate con email).
- En Colab, guardá la key en el panel de "Secrets" con nombre `GROQ_API_KEY` (icono de la llave en la sidebar)."""),

    code("""\
%pip install --quiet groq"""),

    md("""\
## Setup

Si estás en Colab, esto lee la API key del panel de Secrets. Si estás local, exportá `GROQ_API_KEY` antes de abrir Jupyter."""),

    code("""\
import os
from groq import Groq

# En Colab:
try:
    from google.colab import userdata
    os.environ["GROQ_API_KEY"] = userdata.get("GROQ_API_KEY")
except ImportError:
    # Local: asume que ya está exportada
    assert os.environ.get("GROQ_API_KEY"), "Exportá GROQ_API_KEY antes de correr."

client = Groq()
MODEL = "qwen/qwen3-32b"  # rápido y capaz. Alternativas: llama-3.3-70b-versatile

print("Cliente listo, modelo:", MODEL)"""),

    md("""\
## 1. Consulta simple

Mandamos un único mensaje del usuario y leemos la respuesta."""),

    code("""\
resp = client.chat.completions.create(
    model=MODEL,
    messages=[{"role": "user", "content": "Explicame en 2 oraciones qué es un LLM."}],
)

print(resp.choices[0].message.content)"""),

    md("""\
## 2. Efecto de `temperature`

La misma pregunta, dos veces, con temperaturas distintas. Observá cómo cambia el estilo de la respuesta."""),

    code("""\
PROMPT = "Escribime un haiku sobre los lunes."

for temp in [0.1, 1.0]:
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": PROMPT}],
        temperature=temp,
    )
    print(f"--- temperature = {temp} ---")
    print(resp.choices[0].message.content)
    print()"""),

    md("""\
## 3. Cambiar el "carácter" con un `system` prompt

El system prompt define el rol del modelo antes de cualquier interacción del usuario."""),

    code("""\
PIRATA = "Sos un pirata del Caribe del siglo XVII. Hablás siempre en primera persona y usás expresiones piratas."

resp = client.chat.completions.create(
    model=MODEL,
    messages=[
        {"role": "system", "content": PIRATA},
        {"role": "user", "content": "¿Cómo declaro un array en Python?"},
    ],
    temperature=0.6,
)

print(resp.choices[0].message.content)"""),

    md("""\
## Para experimentar después

- Cambiá el `system` prompt: profesor de física, abogado, chef.
- Combiná `system` + `temperature` baja → asistente técnico predecible.
- Probá pasar varios mensajes en `messages` simulando una conversación.

> El próximo notebook (`03_sampling_params`) profundiza en `temperature`, `top_p` y los efectos del sampling."""),
])


# ─────────────────────────────────────────────────────────────────────────────
# 03 — Sampling params
# ─────────────────────────────────────────────────────────────────────────────

write_notebook("03_sampling_params.ipynb", [
    md(f"""\
# 03 — Sampling: temperature, top_p, top_k

{colab_badge("03_sampling_params.ipynb")}

**Objetivo.** Tocar los parámetros de sampling y ver cómo cambia el output. La idea es que después puedas elegirlos a conciencia para tu caso de uso.

**Requisitos.** API key de Groq en `GROQ_API_KEY`."""),

    code("""\
%pip install --quiet groq"""),

    code("""\
import os
from groq import Groq

try:
    from google.colab import userdata
    os.environ["GROQ_API_KEY"] = userdata.get("GROQ_API_KEY")
except ImportError:
    assert os.environ.get("GROQ_API_KEY"), "Exportá GROQ_API_KEY."

client = Groq()
MODEL = "qwen/qwen3-32b"

def generar(prompt, **kwargs):
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        **kwargs,
    )
    return resp.choices[0].message.content"""),

    md("""\
## 1. `temperature` — el termostato de la creatividad

Mismo prompt, distintas temperaturas. Observá la diferencia de tono y vocabulario."""),

    code("""\
PROMPT = "Escribime un poema corto (4 versos) sobre el otoño en Buenos Aires."

for temp in [0.0, 0.3, 0.7, 1.2]:
    print(f"--- temperature = {temp} ---")
    print(generar(PROMPT, temperature=temp))
    print()"""),

    md("""\
- `temperature=0` → el modelo elige siempre el token más probable. Determinista.
- Valores altos → distribución más plana, salidas diversas (y a veces incoherentes)."""),

    md("""\
## 2. Reproducibilidad: el bug del "siempre lo mismo"

Con temperatura baja, dos llamadas seguidas dan respuestas casi idénticas. Útil cuando necesitás determinismo (testing, código)."""),

    code("""\
PROMPT = "Listame 3 razones por las que se prefiere PostgreSQL sobre MySQL."

for i in range(2):
    print(f"--- Llamada {i+1} (temperature=0) ---")
    print(generar(PROMPT, temperature=0))
    print()"""),

    md("""\
## 3. `top_p` — nucleus sampling

Muestrea solo del conjunto de tokens cuya probabilidad acumulada llega a P. Recorta la "cola larga"."""),

    code("""\
PROMPT = "Inventame el nombre de una banda de rock progresivo argentina."

for p in [0.1, 0.5, 1.0]:
    print(f"--- top_p = {p} ---")
    for _ in range(3):
        print("  ·", generar(PROMPT, temperature=0.9, top_p=p))
    print()"""),

    md("""\
## 4. El bug del repetition loop

Con `temperature` muy baja **y** un prompt ambiguo, el modelo puede entrar en bucle repitiendo frases. Lo provocamos a propósito:"""),

    code("""\
# Prompt cuasi-vacío + temperatura mínima -> riesgo de loop
print(generar("repetí: ", temperature=0.0, max_tokens=100))"""),

    md("""\
## Cuándo usar qué

| Caso | temperature | top_p |
|---|---|---|
| Código, factual, extracción | 0.0 – 0.3 | 1.0 |
| Conversación natural | 0.6 – 0.8 | 0.9 |
| Creatividad, brainstorming | 0.9 – 1.2 | 0.95 |
| Determinismo (tests) | 0.0 | 1.0 |"""),
])


# ─────────────────────────────────────────────────────────────────────────────
# 04 — Prompting techniques
# ─────────────────────────────────────────────────────────────────────────────

write_notebook("04_prompting_techniques.ipynb", [
    md(f"""\
# 04 — Zero-shot, one-shot, few-shot

{colab_badge("04_prompting_techniques.ipynb")}

**Objetivo.** Comparar las tres técnicas en una tarea concreta: clasificación de sentimiento. Vas a ver cómo unos pocos ejemplos cambian el comportamiento."""),

    code("""\
%pip install --quiet groq"""),

    code("""\
import os
from groq import Groq

try:
    from google.colab import userdata
    os.environ["GROQ_API_KEY"] = userdata.get("GROQ_API_KEY")
except ImportError:
    assert os.environ.get("GROQ_API_KEY"), "Exportá GROQ_API_KEY."

client = Groq()
MODEL = "qwen/qwen3-32b"

def clasificar(messages, temperature=0.1):
    resp = client.chat.completions.create(model=MODEL, messages=messages, temperature=temperature)
    return resp.choices[0].message.content.strip()"""),

    md("""\
## Reseñas de prueba

Mezclamos casos fáciles y ambiguos para ver dónde fallan las distintas técnicas."""),

    code("""\
RESEÑAS = [
    "El servicio fue impecable, vuelvo seguro.",                                 # claramente positiva
    "Pésimo, no me devolvieron la plata.",                                       # claramente negativa
    "Está OK, nada del otro mundo pero cumple.",                                 # neutral
    "El servicio fue lento pero la comida estuvo buena.",                        # ambiguo
    "Llegó a tiempo, eso es lo único bueno que puedo decir.",                    # sarcástico/negativo
    "Hace lo que promete, ni más ni menos.",                                     # neutral
]"""),

    md("""\
## Zero-shot

Solo le decimos qué hacer, sin ejemplos."""),

    code("""\
SYSTEM_ZS = "Sos un clasificador de sentimientos. Respondés SOLO con una palabra: POSITIVA, NEUTRAL o NEGATIVA."

print("ZERO-SHOT")
print("-" * 60)
for r in RESEÑAS:
    out = clasificar([
        {"role": "system", "content": SYSTEM_ZS},
        {"role": "user", "content": f"Reseña: {r}"},
    ])
    print(f"  {out:<10} <- {r}")"""),

    md("""\
## Few-shot

Le damos 3 ejemplos resueltos. Observá cómo mejora la consistencia en los casos ambiguos."""),

    code("""\
SYSTEM_FS = "Sos un clasificador de sentimientos. Respondés SOLO con: POSITIVA, NEUTRAL o NEGATIVA."

EJEMPLOS = '''Reseña: Excelente atención y precio justo.
Etiqueta: POSITIVA

Reseña: Lo recibí roto y nadie responde.
Etiqueta: NEGATIVA

Reseña: Cumple, pero tarda más que la competencia.
Etiqueta: NEUTRAL

'''

print("FEW-SHOT")
print("-" * 60)
for r in RESEÑAS:
    user = EJEMPLOS + f"Reseña: {r}\\nEtiqueta:"
    out = clasificar([
        {"role": "system", "content": SYSTEM_FS},
        {"role": "user", "content": user},
    ])
    print(f"  {out:<10} <- {r}")"""),

    md("""\
## Para experimentar

- Cambiá los ejemplos del few-shot por casos más cercanos a tu dominio. Probablemente mejore más.
- Probá con 1 solo ejemplo (one-shot) y compará con few-shot.
- Pedí al modelo que devuelva además una **explicación** de por qué clasificó así. Esto es CoT, lo vemos en el próximo notebook.
- Probá con `temperature=0.7` en zero-shot — ¿se vuelve más inconsistente?"""),
])


# ─────────────────────────────────────────────────────────────────────────────
# 05 — CoT + Structured output
# ─────────────────────────────────────────────────────────────────────────────

write_notebook("05_cot_structured.ipynb", [
    md(f"""\
# 05 — Chain of Thought + Structured Output

{colab_badge("05_cot_structured.ipynb")}

**Objetivo.**
1. Ver cómo CoT mejora respuestas de razonamiento.
2. Generar JSON parseable directamente desde el modelo, listo para integrar con código."""),

    code("""\
%pip install --quiet groq"""),

    code("""\
import os
import json
from groq import Groq

try:
    from google.colab import userdata
    os.environ["GROQ_API_KEY"] = userdata.get("GROQ_API_KEY")
except ImportError:
    assert os.environ.get("GROQ_API_KEY"), "Exportá GROQ_API_KEY."

client = Groq()
MODEL = "qwen/qwen3-32b"

def chat(messages, **kwargs):
    resp = client.chat.completions.create(model=MODEL, messages=messages, **kwargs)
    return resp.choices[0].message.content"""),

    md("""\
## Parte 1 — Chain of Thought

Problema típico de sistemas: cuántos GB de logs genera tu API."""),

    code("""\
PROBLEMA = '''
Tenemos un sistema con 1.000.000 de usuarios registrados.
El 0,3% son usuarios premium.
Cada usuario premium genera en promedio 15 eventos por día.
Cada evento se almacena como una fila de 512 bytes en una tabla de logs.

Pregunta: ¿cuántos GB de logs de eventos premium se generan por SEMANA?
'''
print(PROBLEMA)"""),

    md("""\
### Sin CoT — respuesta directa"""),

    code("""\
print(chat(
    [{"role": "user", "content": PROBLEMA}],
    temperature=0.1,
))"""),

    md("""\
### Con CoT — forzando paso a paso"""),

    code("""\
SYSTEM_COT = (
    "Sos un ingeniero de sistemas. Resolvé los problemas paso a paso, "
    "mostrando cada operación con su unidad antes de dar la respuesta final. "
    "Terminá con una línea 'Respuesta: <valor>'."
)

print(chat(
    [
        {"role": "system", "content": SYSTEM_COT},
        {"role": "user", "content": PROBLEMA},
    ],
    temperature=0.1,
))"""),

    md("""\
### Verificación numérica

Calculamos a mano para comparar."""),

    code("""\
usuarios_premium = 1_000_000 * 0.003
eventos_dia      = usuarios_premium * 15
eventos_semana   = eventos_dia * 7
bytes_semana     = eventos_semana * 512
gb_semana        = bytes_semana / (1024**3)

print(f"Premium:           {usuarios_premium:>15,.0f} usuarios")
print(f"Eventos por semana:{eventos_semana:>15,.0f}")
print(f"Bytes por semana:  {bytes_semana:>15,.0f}")
print(f"GB por semana:     {gb_semana:>15.3f}")"""),

    md("""\
## Parte 2 — Structured Output (JSON)

Forzamos al modelo a devolver JSON parseable. Se usa muchísimo en producción para integrar LLMs con código tradicional."""),

    code("""\
SYSTEM_JSON = '''Analizás requerimientos de software.
Respondés SOLO con un JSON válido con esta estructura exacta:
{
  "tipo": "funcional" | "no_funcional" | "restriccion",
  "prioridad": "alta" | "media" | "baja",
  "componente": string,
  "resumen": string (máx 15 palabras)
}
No incluyas nada más que el JSON. Sin markdown, sin explicaciones.'''

REQUERIMIENTOS = [
    "El sistema debe responder en menos de 200ms al 95% de las solicitudes bajo carga normal.",
    "Los usuarios pueden recuperar su contraseña por email.",
    "Toda la información sensible debe almacenarse cifrada en reposo (AES-256).",
]

for req in REQUERIMIENTOS:
    raw = chat(
        [
            {"role": "system", "content": SYSTEM_JSON},
            {"role": "user", "content": req},
        ],
        temperature=0.1,
    )

    # parse defensivo: a veces el modelo mete ```json ... ```
    clean = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        data = json.loads(clean)
        print(f"OK -> {req}")
        for k, v in data.items():
            print(f"    {k}: {v}")
    except json.JSONDecodeError as e:
        print(f"Error parseando: {e}")
        print(f"  raw: {raw}")
    print()"""),

    md("""\
## Tips para production

- **Tolerá errores de parseo**: usá `json-repair` o un retry con un mensaje "el JSON anterior estaba mal".
- **Validá con un schema** (pydantic, jsonschema). El modelo se equivoca: si el schema no se cumple, descartá la respuesta.
- **`response_format={"type": "json_object"}`**: muchos modelos (incluido Qwen vía Groq) tienen un modo JSON forzado. Probalo.
- **Para tareas de razonamiento + JSON**: pedí primero el razonamiento (en un campo `reasoning`) y después la respuesta. Mejora calidad."""),
])


print()
print("Listo.")
