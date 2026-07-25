import numpy as np
import matplotlib.pyplot as plt

# --- Activación Sigmoide ---
class Sigmoid:
    @staticmethod
    def forward(Z):
        return 1 / (1 + np.exp(-Z))
    @staticmethod
    def derivative(A):
        return A * (1 - A)

# --- Capa densa ---
class Dense:
    def __init__(self, units, input_shape, activation=None):
        self.W = np.random.randn(input_shape[0], units) * 0.1
        self.b = np.zeros((units,), dtype=float)
        self.activation = activation

    def forward(self, X):
        self.X = X
        Z = X.dot(self.W) + self.b
        return self.activation.forward(Z) if self.activation else Z

    def backward(self, dA):
        if self.activation:
            Z = self.X.dot(self.W) + self.b
            A = self.activation.forward(Z)
            dZ = dA * self.activation.derivative(A)
        else:
            dZ = dA
        dW = self.X.T.dot(dZ)
        db = np.sum(dZ, axis=0)
        dX = dZ.dot(self.W.T)
        return dW, db, dX

# --- Optimizador Adam ---
class Adam:
    def __init__(self, lr, beta1=0.9, beta2=0.999, eps=1e-7):
        self.lr, self.b1, self.b2, self.eps = lr, beta1, beta2, eps
        self.m_w = self.v_w = self.m_b = self.v_b = None
        self.t = 0

    def update(self, W, b, dW, db):
        if self.m_w is None:
            self.m_w, self.v_w = np.zeros_like(dW), np.zeros_like(dW)
            self.m_b, self.v_b = np.zeros_like(db), np.zeros_like(db)
        self.t += 1
        self.m_w = self.b1 * self.m_w + (1 - self.b1) * dW
        self.v_w = self.b2 * self.v_w + (1 - self.b2) * (dW ** 2)
        self.m_b = self.b1 * self.m_b + (1 - self.b1) * db
        self.v_b = self.b2 * self.v_b + (1 - self.b2) * (db ** 2)
        m_w_corr = self.m_w / (1 - self.b1 ** self.t)
        v_w_corr = self.v_w / (1 - self.b2 ** self.t)
        m_b_corr = self.m_b / (1 - self.b1 ** self.t)
        v_b_corr = self.v_b / (1 - self.b2 ** self.t)
        W -= self.lr * m_w_corr / (np.sqrt(v_w_corr) + self.eps)
        b -= self.lr * m_b_corr / (np.sqrt(v_b_corr) + self.eps)
        return W, b

# --- Modelo secuencial ---
class Sequential:
    def __init__(self, layers):
        self.layers = layers

    def compile(self, optimizers, loss='mean_squared_error'):
        self.opts = [optimizers] * len(self.layers) if not isinstance(optimizers, list) else optimizers

    def fit(self, X, Y, epochs=1000, verbose=False):
        self._scale = 255.0
        X_norm, Y_norm = X / self._scale, Y
        n = X_norm.shape[0]
        for ep in range(1, epochs + 1):
            A = X_norm
            for layer in self.layers:
                A = layer.forward(A)
            y_pred = A.flatten()
            error = y_pred - Y_norm
            dA = (2 / n) * error.reshape(-1, 1)
            for layer, opt in zip(reversed(self.layers), reversed(self.opts)):
                dW, db, dA = layer.backward(dA)
                layer.W, layer.b = opt.update(layer.W, layer.b, dW, db)
            if verbose and ep % 200 == 0:
                loss = np.mean(error ** 2)
                print(f"Epoch {ep}: loss = {loss:.6f}")

    def predict(self, X_new):
        X_norm = X_new / self._scale
        A = X_norm
        for layer in self.layers:
            A = layer.forward(A)
        return A.flatten()

# ======================================================
# 1) Datos de entrenamiento: Rojo (1) vs No rojo (0)
# ======================================================
X_train = np.array([
    [255, 0, 0],    # rojo puro
    [250, 0, 0],
    [240, 0, 0],
    [230, 0, 0],
    [220, 0, 0],
    [210, 0, 0],
    [200, 0, 0],
    [180, 0, 0],
    [175, 0, 0],

    [0, 0, 255],    # azul
    [0, 255, 0],    # verde
    [120, 120, 120],# gris
    [255, 255, 255],# blanco
    [0, 0, 0],      # negro
    [255, 255, 0],  # amarillo
    [128, 0, 128], #morado
    [255, 165, 0], #naranja
    [0, 255, 255], #cian
    (255, 0, 255), # magenta
    (150, 75, 0), #cafe
    (255, 192, 203), #rosado
], dtype=float)

Y_train = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=float)

# ======================================================
# 2) Construcción del modelo
# ======================================================
capa1 = Dense(units=4, input_shape=[3], activation=Sigmoid)
capa2 = Dense(units=1, input_shape=[4], activation=Sigmoid)
opt1, opt2 = Adam(lr=0.05), Adam(lr=0.05)

modelo = Sequential([capa1, capa2])
modelo.compile(optimizers=[opt1, opt2])
modelo.fit(X_train, Y_train, epochs=2000, verbose=False)

# ======================================================
# 3) Probar con un color nuevo y mostrarlo
# ======================================================
color_test = np.array([[225, 0, 0]], dtype=float)
prob = modelo.predict(color_test)[0]

print(f"Probabilidad de ser ROJO: {prob}")

# Mostrar visualmente
plt.figure(figsize=(2,2))
plt.imshow([[color_test[0]/255.0]])
plt.axis("off")
plt.title(f"Prob. rojo: {prob}")
plt.show()