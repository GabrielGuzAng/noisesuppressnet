# AGENTS_GUIDE.md — Guía de agentes y workflows

Este archivo describe workflows compuestos donde múltiples roles de ROLES.md
interactúan en secuencia. Sirve para tareas complejas que involucran varias etapas.

**Uso**: al iniciar una tarea compleja, invocar el workflow correspondiente:

> "Seguí el workflow [nombre_workflow] de AGENTS_GUIDE.md"

O si querés que Claude Code sugiera:

> "¿Qué workflow de AGENTS_GUIDE.md aplica para esta tarea?"

---

## Índice de workflows

| Workflow | Cuándo usarlo | Duración típica |
|----------|----------------|------------------|
| `nueva_variante` | Al planear y ejecutar una variante nueva (V6, V7...) | Multi-sesión (días) |
| `sesion_diagnostica` | Al retomar el proyecto después de tiempo sin tocarlo | 30-60 min |
| `preparacion_reunion` | Antes de reunión con profesores | 45-90 min |
| `preparacion_defensa` | Antes de defensa parcial o final | Multi-sesión (días) |
| `commit_significativo` | Al preparar un tag de versión | 30-45 min |
| `analisis_hallazgo` | Cuando aparece resultado inesperado | 30-60 min |
| `pivot_metodologico` | Al considerar un cambio de rumbo | 60-90 min |

---

## Workflow: `nueva_variante`

**Objetivo**: planear, implementar, ejecutar y documentar una variante nueva del ablation.

### Etapas

**Etapa 1 — Diseño (usar rol: `experiment_designer`)**
1. Definir hipótesis a validar
2. Definir variable a cambiar (una sola)
3. Configuración CONFIG_V<n> heredando de _BASE
4. Predicción del resultado esperado (falsable)
5. Costo estimado en horas de compute

**Etapa 2 — Validación con literatura (usar rol: `researcher`)**
1. Buscar papers que sustenten la hipótesis
2. Verificar que no haya trabajo previo que ya la refute
3. Documentar referencias en la config

**Etapa 3 — Implementación (usar rol: `implementer`)**
1. Modificar config.py con CONFIG_V<n>
2. Si requiere cambios en trainer/losses: implementar
3. Sanity test: correr 1 época sobre 200 pares para verificar que no explota

**Etapa 4 — Ejecución en background**
1. Correr con nohup + log a archivo
2. Verificar primeros minutos (nvidia-smi, primera época)
3. Dejar corriendo, actualizar CLAUDE.md con "en curso"

**Etapa 5 — Evaluación (usar rol: `evaluator`)**
1. Cuando termine, correr evaluate_variant
2. Reporte de costo con training_cost_report
3. Comparar con variantes previas (Delta explícito)
4. Análisis por bucket y categoría

**Etapa 6 — Documentación (usar rol: `documenter`)**
1. Actualizar docs/EXPERIMENTS.md con sección V<n>
2. Actualizar results/summary_metrics.csv
3. Si aparecieron decisiones metodológicas: docs/decisions.md
4. Actualizar CHANGELOG.md

**Etapa 7 — Commit (usar rol: `code_reviewer` + commit)**
1. Review de código antes de commit
2. Commit + tag v<n>.0.0
3. Push a GitHub
4. Actualizar CLAUDE.md con checkbox completado

### Deliverables al final

- checkpoints/v<n>/best.pt
- checkpoints/v<n>/history.json
- results/v<n>_test_sealed.json
- results/v<n>_training_cost.json
- Sección V<n> en docs/EXPERIMENTS.md
- Fila V<n> en results/summary_metrics.csv
- Tag git v<n>.0.0 pushed

---

## Workflow: `sesion_diagnostica`

**Objetivo**: retomar el proyecto después de días o semanas sin tocarlo. Recuperar contexto rápido.

### Etapas

**Etapa 1 — Sync del estado (5 min)**