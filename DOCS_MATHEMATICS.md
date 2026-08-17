# 📖 Documentación Matemática Completa: "Attention Is All You Need" y Transformers

Esta documentación contiene la **derivación matemática exhaustiva y paso a paso** de todos los componentes de la arquitectura **Transformer (Attention Is All You Need, Vaswani et al., 2017)** y su variante autorregresiva **Decoder-Only (GPT)**.

Su objetivo es servir como una **guía técnica de referencia permanente** que te permita entender, verificar e implementar cada ecuación matemática de una red neuronal desde cero, sin necesidad de consultar una IA o librerías externas.

---

## 📑 Tabla de Contenidos
1. [Incrustaciones y Codificación Posicional](#1-incrustaciones-y-codificación-posicional)
2. [Atención por Producto Escalar Escalado (Scaled Dot-Product Attention)](#2-atención-por-producto-escalar-escalado)
3. [¿Por qué se escala dividiendo por $\sqrt{d_k}$? (Demostración de Varianza)](#3-demostración-del-escalado-por-sqrt-d_k)
4. [Máscara Causal (Causal Masking)](#4-máscara-causal)
5. [Softmax Estable (Softmax Stabilization)](#5-softmax-estable)
6. [Atención Multi-Cabeza (Multi-Head Attention)](#6-atención-multi-cabeza)
7. [Normalización de Capa (Pre-Layer Normalization)](#7-normalización-de-capa-pre-layernorm)
8. [Redes Feed-Forward (MLP) y Activación GELU](#8-redes-feed-forward-mlp-y-activación-gelu)
9. [Conexiones Residuales (Skip Connections)](#9-conexiones-residuales)
10. [Función de Pérdida (Cross-Entropy Loss)](#10-función-de-pérdida-cross-entropy-loss)
11. [Retropropagación y Derivadas Analíticas (Backward Pass)](#11-retropropagación-y-derivadas-analíticas)

---

## 1. Incrustaciones y Codificación Posicional

Un Transformer procesa secuencias de tokens $X = (x_1, x_2, \dots, x_T)$ de longitud $T$.

### 1.1 Incrustación de Tokens (Token Embedding)
Cada token $x_t \in \{1, \dots, V\}$ (donde $V$ es el tamaño del vocabulario) se convierte en un vector continuo de dimensión $d_{model}$ mediante una matriz de pesos entrenables $W_{te} \in \mathbb{R}^{V \times d_{model}}$:

$$E_{\text{tok}}(x_t) = W_{te}[x_t, :] \in \mathbb{R}^{d_{model}}$$

### 1.2 Codificación Posicional (Positional Encoding)
Dado que la atención opera en paralelo sobre todos los tokens de forma agnóstica al orden, debemos inyectar información de posición $p \in \{0, \dots, T-1\}$.

#### Opción A: Codificación Senoidal Fija (Paper Original 2017)
Para cada posición $pos$ y dimensión $i \in \{0, \dots, \frac{d_{model}}{2}-1\}$:

$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i / d_{model}}}\right)$$
$$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i / d_{model}}}\right)$$

#### Opción B: Incrustación Posicional Aprendida (Estilo GPT)
Una matriz entrenable $W_{pe} \in \mathbb{R}^{T_{max} \times d_{model}}$:

$$E_{\text{pos}}(pos) = W_{pe}[pos, :] \in \mathbb{R}^{d_{model}}$$

### Entrada Final al Transformer:
$$Z_t = E_{\text{tok}}(x_t) + E_{\text{pos}}(t) \in \mathbb{R}^{d_{model}}$$
Para toda la secuencia, la matriz de entrada es $Z \in \mathbb{R}^{T \times d_{model}}$.

---

## 2. Atención por Producto Escalar Escalado

La atención mapea tres matrices: **Query ($Q$)**, **Key ($K$)** y **Value ($V$)**.

### 2.1 Proyecciones Lineales
Dado $Z \in \mathbb{R}^{T \times d_{model}}$, proyectamos mediante matrices entrenables $W_Q, W_K, W_V \in \mathbb{R}^{d_{model} \times d_k}$:

$$Q = Z W_Q \in \mathbb{R}^{T \times d_k}$$
$$K = Z W_K \in \mathbb{R}^{T \times d_k}$$
$$V = Z W_V \in \mathbb{R}^{T \times d_v}$$

### 2.2 Ecuación de Atención Fundamental:
$$\text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V$$

Donde $Q K^T \in \mathbb{R}^{T \times T}$ es la matriz de puntuaciones de similitud entre todos los pares de tokens.

---

## 3. Demostración del Escalado por $\sqrt{d_k}$

### ¿Por qué dividimos $Q K^T$ por $\sqrt{d_k}$?

Supongamos que los componentes de $q \in \mathbb{R}^{d_k}$ y $k \in \mathbb{R}^{d_k}$ son variables aleatorias independientes con media $\mu = 0$ y varianza $\sigma^2 = 1$.

El producto escalar es:
$$q \cdot k = \sum_{i=1}^{d_k} q_i k_i$$

1. **Media del producto escalar**:
   $$\mathbb{E}[q \cdot k] = \sum_{i=1}^{d_k} \mathbb{E}[q_i k_i] = \sum_{i=1}^{d_k} \mathbb{E}[q_i] \mathbb{E}[k_i] = 0$$

2. **Varianza del producto escalar**:
   $$\text{Var}(q \cdot k) = \sum_{i=1}^{d_k} \text{Var}(q_i k_i) = \sum_{i=1}^{d_k} \mathbb{E}[q_i^2 k_i^2] - (\mathbb{E}[q_i k_i])^2 = \sum_{i=1}^{d_k} \mathbb{E}[q_i^2] \mathbb{E}[k_i^2] = \sum_{i=1}^{d_k} (1)(1) = d_k$$

**Conclusión**: La varianza del producto escalar crece linealmente con la dimensión $d_k$. 
Para valores grandes de $d_k$ (ej. $d_k = 64$ o $128$), los productos escalares tienen magnitudes muy grandes, lo que empuja a la función Softmax hacia regiones de saturación con gradientes extremadamente pequeños ($\approx 0$), provocando el **desvanecimiento del gradiente (vanishing gradient)**.

Al dividir por $\sqrt{d_k}$, la varianza se escala a $1$:
$$\text{Var}\left(\frac{q \cdot k}{\sqrt{d_k}}\right) = \frac{1}{d_k} \text{Var}(q \cdot k) = \frac{d_k}{d_k} = 1$$

---

## 4. Máscara Causal (Causal Masking)

En un modelo autorregresivo (Decoder-Only), el token en la posición $i$ no puede atender a tokens en posiciones futuras $j > i$.

Definimos la matriz de máscara causal $M \in \mathbb{R}^{T \times T}$:

$$M_{i,j} = \begin{cases} 0 & \text{si } j \le i \\ -\infty & \text{si } j > i \end{cases}$$

Sumamos la máscara a las puntuaciones antes de aplicar Softmax:

$$S_{\text{masked}} = \frac{Q K^T}{\sqrt{d_k}} + M$$

Como $e^{-\infty} = 0$, la probabilidad de atención a posiciones futuras se convierte exactamente en $0$.

---

## 5. Softmax Estable (Softmax Stabilization)

Para evitar el desbordamiento numérico en punto flotante (`overflow` cuando $e^x > 10^{38}$), restamos el valor máximo de cada fila:

Dada una fila de puntuaciones $s \in \mathbb{R}^T$:

$$m = \max_{j} (s_j)$$
$$P_j = \frac{e^{s_j - m}}{\sum_{k=1}^{T} e^{s_k - m}}$$

Esta propiedad es numéricamente idéntica a la fórmula original pero inmune a desbordamientos:
$$\frac{e^{s_j - m}}{\sum e^{s_k - m}} = \frac{e^{s_j} \cdot e^{-m}}{\sum e^{s_k} \cdot e^{-m}} = \frac{e^{s_j}}{\sum e^{s_k}}$$

---

## 6. Atención Multi-Cabeza (Multi-Head Attention)

En lugar de calcular una sola atención con dimensión $d_{model}$, dividimos las proyecciones en $h$ cabezas paralelas de dimensión $d_k = d_{model} / h$.

Para cada cabeza $i \in \{1, \dots, h\}$:
$$Q_i = Z W_Q^{(i)}, \quad K_i = Z W_K^{(i)}, \quad V_i = Z W_V^{(i)}$$
$$\text{head}_i = \text{Softmax}\left(\frac{Q_i K_i^T}{\sqrt{d_k}} + M\right) V_i \in \mathbb{R}^{T \times d_k}$$

Concatenamos todas las cabezas y proyectamos con la matriz $W_O \in \mathbb{R}^{d_{model} \times d_{model}}$:

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \text{head}_2, \dots, \text{head}_h) W_O$$

---

## 7. Normalización de Capa (Pre-Layer Normalization)

Dada una entrada $x \in \mathbb{R}^{d_{model}}$, LayerNorm normaliza las características a través de la dimensión del modelo (no a través del batch):

1. **Media ($\mu$)**:
   $$\mu = \frac{1}{d_{model}} \sum_{i=1}^{d_{model}} x_i$$

2. **Varianza ($\sigma^2$)**:
   $$\sigma^2 = \frac{1}{d_{model}} \sum_{i=1}^{d_{model}} (x_i - \mu)^2$$

3. **Normalización ($\hat{x}$)**:
   $$\hat{x}_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}}$$
   *(donde $\epsilon = 10^{-5}$ evita la división por cero)*.

4. **Escalado y Desplazamiento Entrenable ($y$)**:
   $$y_i = \gamma_i \cdot \hat{x}_i + \beta_i$$
   *(donde $\gamma, \beta \in \mathbb{R}^{d_{model}}$ son vectores de parámetros aprendidos)*.

---

## 8. Redes Feed-Forward (MLP) y Activación GELU

Cada bloque Transformer contiene una red Feed-Forward de dos capas lineales con expansión interna a $4 \times d_{model}$:

$$\text{FFN}(x) = \text{GELU}(x W_1 + b_1) W_2 + b_2$$

Donde:
- $W_1 \in \mathbb{R}^{d_{model} \times 4 d_{model}}, \quad b_1 \in \mathbb{R}^{4 d_{model}}$
- $W_2 \in \mathbb{R}^{4 d_{model} \times d_{model}}, \quad b_2 \in \mathbb{R}^{d_{model}}$

### Activación GELU (Gaussian Error Linear Unit):
$$\text{GELU}(x) = x \cdot \Phi(x) = x \cdot P(X \le x), \quad X \sim \mathcal{N}(0, 1)$$

Aproximación rápida analítica (usada en GPT-2/3 y en nuestro C++):
$$\text{GELU}(x) \approx 0.5 \cdot x \cdot \left(1 + \tanh\left(\sqrt{\frac{2}{\pi}} \left(x + 0.044715 x^3\right)\right)\right)$$

---

## 9. Conexiones Residuales (Skip Connections)

Para permitir que el gradiente fluya sin obstáculos a través de decenas de capas, utilizamos Pre-LN residuales:

$$x^{(1)} = x + \text{MultiHeadAttention}(\text{LayerNorm}(x))$$
$$x^{(2)} = x^{(1)} + \text{MLP}(\text{LayerNorm}(x^{(1)}))$$

---

## 10. Función de Pérdida (Cross-Entropy Loss)

Dada la salida del modelo $z \in \mathbb{R}^V$ (logits para el vocabulario) y la clase real del token objetivo $y \in \{1, \dots, V\}$:

Probabilidad predicha mediante Softmax:
$$p_i = \frac{e^{z_i}}{\sum_{k=1}^V e^{z_k}}$$

La pérdida de Entropía Cruzada es:
$$\mathcal{L} = -\log(p_y) = -\log \left( \frac{e^{z_y}}{\sum_{k=1}^V e^{z_k}} \right) = -z_y + \log \left( \sum_{k=1}^V e^{z_k} \right)$$

---

## 11. Retropropagación y Derivadas Analíticas (Backward Pass)

Para entrenar la red en C++ puro sin Autodiff automático, calculamos las derivadas analíticas manualmente:

### 11.1 Gradiente de Cross-Entropy + Softmax Combinado
Derivada de la pérdida $\mathcal{L}$ respecto al logit de entrada $z_i$:

$$\frac{\partial \mathcal{L}}{\partial z_i} = p_i - \mathbf{1}_{\{i = y\}}$$

Donde $\mathbf{1}_{\{i = y\}}$ vale $1$ si $i$ es la clase verdadera $y$, y $0$ en caso contrario.

### 11.2 Gradiente de Capa Lineal (MatMul)
Dada la proyección hacia adelante $Y = X W$:
- Gradiente respecto a la entrada $X$:
  $$\frac{\partial \mathcal{L}}{\partial X} = \frac{\partial \mathcal{L}}{\partial Y} W^T$$
- Gradiente respecto a la matriz de pesos $W$:
  $$\frac{\partial \mathcal{L}}{\partial W} = X^T \frac{\partial \mathcal{L}}{\partial Y}$$

### 11.3 Gradiente del Optimizador AdamW
Para cada parámetro $w$ con gradiente $g_t = \frac{\partial \mathcal{L}}{\partial w}$:

1. Actualizar momento de 1.er orden (Media):
   $$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$$
2. Actualizar momento de 2.º orden (Varianza):
   $$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$
3. Corrección de sesgo:
   $$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$
4. Actualización del parámetro con Weight Decay desacoplado $\lambda$:
   $$w_t = w_{t-1} - \eta \cdot \left( \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda w_{t-1} \right)$$

---
*Esta documentación está guardada permanentemente en `cpp/DOCS_MATHEMATICS.md` para consulta y estudio independiente.*
