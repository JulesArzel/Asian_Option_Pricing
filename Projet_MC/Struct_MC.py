#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Projet Monte Carlo – Traduction du code R en Python
Karim Bennoura et Jules Arzel (code original en R)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Paramètres globaux
T = 1
K = 1
n = 100
sigma = 0.25
S_0 = 1
delta = T / n  # pour la partie échantillonnage préférentiel

# Création de la grille de temps Tk
def create_grid(T, n):
    # np.linspace crée n+1 points entre 0 et T
    Tk = np.linspace(0, T, n + 1)
    return Tk

Tk = create_grid(T, n)

# =====================================================================
# QUESTION 1
# =====================================================================

# Simulation d'un mouvement brownien sur la grille Tk
def simul_brown():
    Wn = np.zeros(n + 1)
    Wn[0] = 0
    for i in range(n):
        Z = np.random.randn()  # générer un N(0,1)
        Wn[i + 1] = Wn[i] + np.sqrt(Tk[i + 1] - Tk[i]) * Z
    return Wn

# Simulation des actifs (processus géométrique de Black–Scholes)
def simul_asset():
    Sn = np.zeros(n + 1)
    Sn[0] = S_0
    for i in range(n):
        Z = np.random.randn()
        # La formule utilisée est : S_{i+1} = S_i * exp(( - sigma^2/2 ) * dt + sigma * sqrt(dt) * Z)
        Sn[i + 1] = Sn[i] * np.exp(( - sigma**2 / 2) * (Tk[i + 1] - Tk[i]) + sigma * np.sqrt(Tk[i + 1] - Tk[i]) * Z)
    return Sn

# Calcul de A(T) qui est la moyenne des actifs simulés (en omettant la valeur initiale)
def calcul_A():
    Sn = simul_asset()
    # En R, Sn[2:n+1] correspond à ignorer le premier élément (S0)
    return np.mean(Sn[1:])

# Estimation du prix par Monte Carlo classique (sur i réalisations)
def estim_class_G(i):
    G = []
    for j in range(i):
        payoff = max(calcul_A() - K, 0)
        G.append(payoff)
    # On retourne la moyenne et la variance (variance de l'estimateur MC = sample variance / i)
    return {"estim": np.mean(G), "var": np.var(G, ddof=1) / i}

# Exemple d'estimation classique
print("Estimation classique de G (MC):", estim_class_G(1000))


# =====================================================================
# QUESTION 3 – Variable de contrôle
# =====================================================================

# Simulation de la variable Y à partir d'une réalisation des actifs
def simulate_Y(Sn):
    prod = 1.0
    # En R, la boucle multiplie les Sn[1] à Sn[n] (en indexation R)
    # Ici, on utilise les indices 0 à n-1 (avec Sn[0] = S_0)
    for i in range(n):
        prod *= Sn[i]
    return max(prod**(1 / n) - K, 0)

# Calcul de l'espérance théorique de Y
a = (sigma / 4) * np.sqrt(6 * (n + 1) / (2 * n + 1))
mu_val = (sigma / n) * np.sqrt((n + 1) * (2 * n + 1) / 6)
const = np.exp((-sigma**2) * (n + 1) / (4 * n) + 0.5 * (mu_val**2))
esp_Y = const * (1 - norm.cdf(a, loc=mu_val, scale=1)) - (1 - norm.cdf(a, loc=0, scale=1))
print("esp_Y =", esp_Y)

# Estimation par MC de E(Y) sur i réalisations
def estimate_Y(i):
    Yk = []
    for j in range(i):
        Sn = simul_asset()
        Yk.append(simulate_Y(Sn))
    return np.mean(Yk)

# Estimation du coefficient de contrôle b_optimal
def estimate_b_new(i):
    Phi = np.zeros(i)
    Yi = np.zeros(i)
    for j in range(i):
        Sn = simul_asset()
        # En R, on utilise mean(Sn[2:n+1]), ici on utilise Sn[1:]
        Phi[j] = max(np.mean(Sn[1:]) - K, 0)
        Yi[j] = simulate_Y(Sn)
    Phi_mean = np.mean(Phi)
    Yi_mean = np.mean(Yi)
    num = np.sum((Phi - Phi_mean) * (Yi - Yi_mean))
    denom = np.sum((Yi - Yi_mean) ** 2)
    return -num / denom

# Pour choisir un bon b_opt, on peut tracer b en fonction du nombre de réalisations
b_values = []
# On fait varier i de 1 à 200
for i_val in range(1, 201):
    b_values.append(estimate_b_new(i_val))
plt.plot(b_values)
plt.title("Estimation de b")
plt.xlabel("Nombre de réalisations (i)")
plt.ylabel("b")
plt.show()
# On choisit par exemple b_opt = b_values[49] correspondant à i=50 (index 49 en Python)
b_opt = b_values[49]

# Méthode 1 : avec la valeur exacte de E(Y)
def estim_control_exact_G(i):
    S_vals = []
    for j in range(i):
        Sn = simul_asset()
        Y_val = simulate_Y(Sn)
        A_val = np.mean(Sn[1:])
        payoff = max(A_val - K, 0) + b_opt * (Y_val - esp_Y)
        S_vals.append(payoff)
    return {"estim": np.mean(S_vals), "var_estim": np.var(S_vals, ddof=1) / i}

print("MC avec variable de contrôle (E(Y) exacte):", estim_control_exact_G(1000))

# Méthode 2 : avec la valeur estimée de E(Y)
def estim_control_estime_G(i):
    S_vals = []
    for j in range(i):
        Sn = simul_asset()
        Y_val = simulate_Y(Sn)
        A_val = np.mean(Sn[1:])
        # Remarque : estimate_Y(1000) est recalculé à chaque itération, ce qui peut être lent
        payoff = max(A_val - K, 0) + b_opt * (Y_val - estimate_Y(1000))
        S_vals.append(payoff)
    return {"estim": np.mean(S_vals), "var_estim": np.var(S_vals, ddof=1) / i}

print("MC avec variable de contrôle (E(Y) estimée):", estim_control_estime_G(1000))


# =====================================================================
# QUESTION 4 : Méthode antithétique et échantillonnage préférentiel
# =====================================================================

# (a) Reformulation de G en fonction d'un vecteur gaussien Z

# Fonction calculant W_t_k à partir d'un vecteur Z
def W_t_k(k, Z):
    s = 0
    for i in range(k):  # i = 0,...,k-1
        s += np.sqrt(Tk[i + 1] - Tk[i]) * Z[i]
    return s

# Fonction g qui calcule G à partir d'un vecteur Z de dimension n  
def g(Z):
    total = 0
    # On fait i=1,...,n (en Python, range(1, n+1))
    for i in range(1, n + 1):
        total += np.exp(-0.5 * sigma**2 * Tk[i] + sigma * W_t_k(i, Z))
    return max((1 / n) * total - K, 0)

# Estimation de E(g(Z)) par la méthode de Monte Carlo classique
def estim_class_g_Z(i):
    Gz = []
    for j in range(i):
        # On génère un vecteur Z de dimension n (en R, rnorm(n+1) était utilisé, ici on utilise n)
        Z = np.random.randn(n)
        Gz.append(g(Z))
    return np.mean(Gz)

# Méthode antithétique :
def estim_antith_G(i):
    S_vals = []
    for j in range(i):
        Z = np.random.randn(n)
        S_vals.append(g(Z) + g(-Z))
    return {"estim": np.mean(S_vals) / 2, "var": np.var(S_vals, ddof=1) / (4 * i)}

print("MC antithétique de G:", estim_antith_G(1000))


# (c) Échantillonnage préférentiel

# La discrétisation delta a été définie plus haut (delta = T/n)

# Fonction qui calcule z_{j+1}(y) selon y, j, S_j et z_j
def suite_z(y, j, Sj, zj):
    if j == 0:
        return sigma * np.sqrt(delta) * (y + K) / y
    else:
        return zj - sigma * np.sqrt(delta) * Sj / (n * y)

# Fonction qui calcule S_j en fonction de j et du vecteur [z_1, ..., z_j]
def suite_S(j, y, zj):
    # On utilise Tk[j+1] pour correspondre à l'indexation R
    return S_0 * np.exp(-0.5 * sigma**2 * Tk[j + 1] + sigma * np.sqrt(delta) * np.sum(zj))

# Fonction qui renvoie les vecteurs Z et S (de longueur n) pour un certain y
def vect_z(y):
    Z = np.zeros(n)
    S_vals = np.zeros(n)
    # En R, Z[1] = suite_z(y,0,0,0)
    Z[0] = suite_z(y, 0, 0, 0)
    # Pour k allant de 2 à n en R, ici k de 1 à n-1 (indexation Python)
    for k in range(1, n):
        # On passe les k premiers éléments de Z
        Sk = suite_S(k, y, Z[:k])
        S_vals[k - 1] = Sk
        Z[k] = suite_z(y, k, Sk, Z[k - 1])
    # Dernier élément de S_vals
    S_vals[n - 1] = suite_S(n, y, Z)
    return Z, S_vals

# On définit la fonction prob(y) dont on cherche la racine
def prob(y):
    _, S_vals = vect_z(y)
    return np.mean(S_vals) - K - y

# Recherche graphique de la racine de prob(y)
y_values = np.linspace(0.156, 0.158, 100)
vals = [prob(y) for y in y_values]
plt.plot(y_values, np.abs(vals))
plt.title("Recherche de la racine de prob(y)")
plt.xlabel("y")
plt.ylabel("|prob(y)|")
plt.show()

# D'après le graphe, on choisit y_chapeau = 0.1572
y_chapeau = 0.1572
mu_opt, _ = vect_z(y_chapeau)  # mu_opt est le vecteur Z obtenu

# Fonction w = f/g, où f est la densité d'un vecteur gaussien standard et g celle d'un gaussien de moyenne mu_opt
def w(Z, mu):
    return np.exp(0.5 * np.sum(mu**2) - np.sum(Z * mu))

# Estimateur par échantillonnage préférentiel de G
def estim_pref_G(i, mu_opt):
    S_vals = []
    for j in range(i):
        Z = np.random.randn(n)
        Z_tilde = Z + mu_opt
        S_vals.append(g(Z_tilde) * w(Z_tilde, mu_opt))
    return {"esp_pref": np.mean(S_vals), "var_pref": np.var(S_vals, ddof=1) / i}

print("MC par échantillonnage préférentiel:", estim_pref_G(1000, mu_opt))


# =====================================================================
# CONCLUSION : Comparaison des estimateurs
# =====================================================================

print("\nComparaison des méthodes d'estimation de E(G) :")
print("MC classique          :", estim_class_G(1000))
print("MC antithétique       :", estim_antith_G(1000))
print("MC contrôle (E(Y) exacte):", estim_control_exact_G(1000))
print("MC échantillonnage préf.:", estim_pref_G(1000, mu_opt))
