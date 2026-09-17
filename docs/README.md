# Documentación de NeuralSuite

## Para empezar

| Documento | Para qué |
| --- | --- |
| [QUICKSTART.md](QUICKSTART.md) | Compilar, comprobar y generar tu primera imagen, en diez minutos |
| [MAPA_DEL_CODIGO.md](MAPA_DEL_CODIGO.md) | Qué hay en cada carpeta y **en qué orden leerlo**, con tres recorridos según lo que busques |
| [ARQUITECTURA.md](ARQUITECTURA.md) | Cómo encaja todo, con diagramas: del `Tensor` al modelo que genera imágenes |
| [GLOSARIO.md](GLOSARIO.md) | Los términos del código y los logs, en cuatro líneas cada uno y diez bloques temáticos |

## Para usar

| Documento | Para qué |
| --- | --- |
| [GUIA_LLM.md](GUIA_LLM.md) | Entrenar, evaluar y generar texto |
| [GUIA_DIFUSION.md](GUIA_DIFUSION.md) | Entrenar la difusión, reanudar y generar imágenes |
| [GUIA_AUTOENCODER.md](GUIA_AUTOENCODER.md) | Comprimir a un latente y por qué difundir ahí cuesta 6× menos |
| [MODELOS.md](MODELOS.md) | Los tres modelos entrenados: datos, comandos, resultados y límites |

## Para estudiar

| Documento | Para qué |
| --- | --- |
| [TEORIA_E_IMPLEMENTACION.md](TEORIA_E_IMPLEMENTACION.md) | Cada concepto: la idea, la matemática, dónde está en el código, cómo se verifica y de dónde sale |
| [REFERENCIAS.md](REFERENCIAS.md) | **Libros, cursos, explicaciones e implementaciones de referencia**, más qué artículo implementa cada pieza y en qué nos apartamos de él. Incluye cuatro rutas de estudio |
| [../DOCS_MATHEMATICS.md](../DOCS_MATHEMATICS.md) | Derivación completa del Transformer |
| [../DOCS_PROGRAMMING_CPP.md](../DOCS_PROGRAMMING_CPP.md) | Patrones de C++17 usados |

## Para contribuir o auditar

| Documento | Para qué |
| --- | --- |
| [VERIFICACION.md](VERIFICACION.md) | Las siete capas de verificación y **los fallos reales que encontró cada una** |
| [ESTADO.md](ESTADO.md) | Qué existe, qué está verificado, qué está **integrado** y qué ha **entrenado** de verdad |
| [ROADMAP.md](ROADMAP.md) | Lo que falta y en qué orden |
| [history/DIARIO_FASES.md](history/DIARIO_FASES.md) | El diario de ingeniería fase a fase, ya cerrado: defectos, mediciones y decisiones |
| [AUTOGRAD_CAPAS.md](AUTOGRAD_CAPAS.md) | Por qué las capas no se migraron al autograd |

---

## Por dónde empezar según quién seas

- **Quiero probarlo ya** → [QUICKSTART.md](QUICKSTART.md): compilar, probar y generar una imagen.
- **Quiero entender el código** → [MAPA_DEL_CODIGO.md](MAPA_DEL_CODIGO.md), y luego la guía del modelo que te interese.
- **Quiero saber qué está hecho de verdad** → [ESTADO.md](ESTADO.md).
- **Quiero estudiar redes neuronales** → [GLOSARIO.md](GLOSARIO.md) → [TEORIA_E_IMPLEMENTACION.md](TEORIA_E_IMPLEMENTACION.md), con el código delante.
- **Quiero evaluar si es fiable** → [VERIFICACION.md](VERIFICACION.md) y los resultados de [MODELOS.md](MODELOS.md).
- **Quiero contribuir** → [ARQUITECTURA.md](ARQUITECTURA.md), y su sección final sobre cómo añadir una capa nueva.
