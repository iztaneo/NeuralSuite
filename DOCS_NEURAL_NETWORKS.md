# 📖 Documentación Teórica: Redes Neuronales en NeuralSuite

Esta guía contiene la **explicación teórica y matemática de las arquitecturas de redes neuronales** soportadas por NeuralSuite: MLPs, CNNs, LSTMs, Funciones de Pérdida y Optimizadores.

---

## 1. Perceptrón Multicapa (MLP / Feed-Forward Networks)

Un MLP es una red neuronal densa donde cada neurona de la capa $l$ está conectada con todas las neuronas de la capa $l-1$.

### Forward Pass (Capa Densa / Linear)
Dada una entrada $X \in \mathbb{R}^{B \times d_{in}}$, la proyección lineal es:

$$Y = X W + b$$

Donde $W \in \mathbb{R}^{d_{in} \times d_{out}}$ es la matriz de pesos y $b \in \mathbb{R}^{d_{out}}$ es el vector de sesgos (bias).

### Backward Pass (Gradientes)
Dada la derivada parcial de la pérdida respecto a la salida $\frac{\partial \mathcal{L}}{\partial Y} \in \mathbb{R}^{B \times d_{out}}$:

$$\frac{\partial \mathcal{L}}{\partial X} = \frac{\partial \mathcal{L}}{\partial Y} W^T \in \mathbb{R}^{B \times d_{in}}$$
$$\frac{\partial \mathcal{L}}{\partial W} = X^T \frac{\partial \mathcal{L}}{\partial Y} \in \mathbb{R}^{d_{in} \times d_{out}}$$
$$\frac{\partial \mathcal{L}}{\partial b} = \sum_{\text{batch}} \frac{\partial \mathcal{L}}{\partial Y} \in \mathbb{R}^{d_{out}}$$

---

## 2. Redes Neuronales Convolucionales (CNNs)

Las redes convolucionales procesan tensores 2D/3D (como imágenes de dimensiones $C \times H \times W$) aprovechando la invarianza espacial.

### Operación Convolucional 2D (`Conv2D`)
Dado un mapa de entrada $X \in \mathbb{R}^{C_{in} \times H \times W}$ y un filtro $K \in \mathbb{R}^{C_{out} \times C_{in} \times k_h \times k_w}$:

$$Y_{o, i, j} = b_o + \sum_{c=1}^{C_{in}} \sum_{m=0}^{k_h-1} \sum_{n=0}^{k_w-1} X_{c, i+m, j+n} \cdot K_{o, c, m, n}$$

### Pooling Máximo (`MaxPool2D`)
Reduce la dimensión espacial tomando el valor máximo en una ventana $k \times k$:

$$Y_{i, j} = \max_{m, n \in [0, k-1]} X_{i \cdot s + m, j \cdot s + n}$$

---

## 3. Redes Recurrentes (LSTM - Long Short-Term Memory)

Las LSTMs resuelven el problema del desvanecimiento del gradiente en secuencias mediante un estado de celda ($C_t$) y tres puertas (*gates*):

$$\begin{aligned}
f_t &= \sigma(W_f \cdot [h_{t-1}, x_t] + b_f) \quad &&\text{(Forget Gate)} \\
i_t &= \sigma(W_i \cdot [h_{t-1}, x_t] + b_i) \quad &&\text{(Input Gate)} \\
\tilde{C}_t &= \tanh(W_c \cdot [h_{t-1}, x_t] + b_c) \quad &&\text{(Candidate State)} \\
C_t &= f_t \odot C_{t-1} + i_t \odot \tilde{C}_t \quad &&\text{(New Cell State)} \\
o_t &= \sigma(W_o \cdot [h_{t-1}, x_t] + b_o) \quad &&\text{(Output Gate)} \\
h_t &= o_t \odot \tanh(C_t) \quad &&\text{(Hidden State Output)}
\end{aligned}$$

---

## 4. Funciones de Pérdida (Loss Functions)

### Cross-Entropy Loss (Entropía Cruzada)
Para tareas de clasificación y predicción de tokens en LLMs:

$$\mathcal{L} = -\sum_{i=1}^{V} y_i \log(p_i)$$

### Mean Squared Error (MSE)
Para tareas de regresión numérica:

$$\mathcal{L} = \frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2$$

---

## 5. Optimizadores

### AdamW (Adaptive Moment Estimation con Weight Decay Desacoplado)
$$m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t, \quad v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$$
$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$
$$\theta_t = \theta_{t-1} - \eta \cdot \left( \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda \theta_{t-1} \right)$$
