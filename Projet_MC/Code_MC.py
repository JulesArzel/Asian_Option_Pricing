#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module Monte Carlo – Estimation de prix par simulation
Traduit et amélioré à partir d’un code R initial.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


class MonteCarloEstimator:
    def __init__(self, T=1, K=1, n=100, sigma=0.25, S0=1):
        """
        Initialise les paramètres du modèle.
          - T : horizon
          - K : strike
          - n : nombre de pas de temps
          - sigma : volatilité
          - S0 : valeur initiale de l’actif
        """
        self.T = T
        self.K = K
        self.n = n
        self.sigma = sigma
        self.S0 = S0
        self.delta = T / n
        self.dt = T / n
        self.Tk = np.linspace(0, T, n + 1)

    def create_grid(self):
        """Renvoie la grille de temps Tk."""
        return self.Tk

    def simul_brown(self):
        """Simule une trajectoire du mouvement brownien sur la grille Tk."""
        dt_increments = np.diff(self.Tk)
        increments = np.sqrt(dt_increments) * np.random.randn(self.n)
        W = np.empty(self.n + 1)
        W[0] = 0
        W[1:] = np.cumsum(increments)
        return W

    def simul_asset(self):
        """Simule une trajectoire de l’actif (processus géométrique de Black–Scholes)."""
        dt = self.dt
        Z = np.random.randn(self.n)
        # Calcul des incréments multiplicatifs
        increments = np.exp((-self.sigma**2 / 2) * dt + self.sigma * np.sqrt(dt) * Z)
        S = np.empty(self.n + 1)
        S[0] = self.S0
        S[1:] = self.S0 * np.cumprod(increments)
        return S

    def simul_asset_paths(self, num_paths):
        """
        Simule num_paths trajectoires de l’actif de longueur n+1 de façon vectorisée.
        Renvoie un tableau de dimension (num_paths, n+1).
        """
        dt = self.dt
        Z = np.random.randn(num_paths, self.n)
        increments = np.exp((-self.sigma**2 / 2) * dt + self.sigma * np.sqrt(dt) * Z)
        S_paths = np.empty((num_paths, self.n + 1))
        S_paths[:, 0] = self.S0
        S_paths[:, 1:] = self.S0 * np.cumprod(increments, axis=1)
        return S_paths

    def calcul_A(self, S):
        """
        Pour une trajectoire d'actif S (vecteur 1D de taille n+1),
        renvoie A = moyenne(S[1:]).
        """
        return np.mean(S[1:])

    def estim_class_G(self, num_simulations):
        """
        Estimation classique du payoff G = max(A - K, 0) sur num_simulations.
        Renvoie un dictionnaire avec l'estimation et la variance de l'estimateur.
        """
        S_paths = self.simul_asset_paths(num_simulations)
        A_vals = np.mean(S_paths[:, 1:], axis=1)
        payoffs = np.maximum(A_vals - self.K, 0)
        return {"estim": np.mean(payoffs), "var": np.var(payoffs, ddof=1) / num_simulations}

    def simulate_Y(self, S):
        """
        Pour une trajectoire d'actif S (1D array), calcule Y = max( (géom. moyenne de S[1:]) - K, 0).
        On utilise la moyenne des logarithmes pour plus de stabilité.
        """
        gm = np.exp(np.mean(np.log(S[1:])))
        return max(gm - self.K, 0)

    def estimate_Y(self, num_simulations):
        """
        Estimation vectorisée de E(Y) sur num_simulations trajectoires.
        """
        S_paths = self.simul_asset_paths(num_simulations)
        gm = np.exp(np.mean(np.log(S_paths[:, 1:]), axis=1))
        Y_vals = np.maximum(gm - self.K, 0)
        return np.mean(Y_vals)

    def estimate_b_new(self, num_simulations):
        """
        Estimation vectorisée du coefficient de contrôle b_optimal à partir de num_simulations.
        b_opt = - Cov(Phi, Y) / Var(Y) où
          Phi = max(mean(S[1:]) - K, 0)
          Y = max( (géom. moyenne de S[1:]) - K, 0)
        """
        S_paths = self.simul_asset_paths(num_simulations)
        Phi = np.maximum(np.mean(S_paths[:, 1:], axis=1) - self.K, 0)
        gm = np.exp(np.mean(np.log(S_paths[:, 1:]), axis=1))
        Y_vals = np.maximum(gm - self.K, 0)
        Phi_mean = np.mean(Phi)
        Y_mean = np.mean(Y_vals)
        cov = np.mean((Phi - Phi_mean) * (Y_vals - Y_mean))
        var_Y = np.mean((Y_vals - Y_mean) ** 2)
        return -cov / var_Y if var_Y != 0 else 0

    def estim_control_exact_G(self, num_simulations, b_opt, esp_Y):
        """
        Estimation par variable de contrôle (avec la valeur exacte de E(Y)) :
        G = max(A - K, 0) + b_opt*(Y - esp_Y)
        """
        S_paths = self.simul_asset_paths(num_simulations)
        A_vals = np.mean(S_paths[:, 1:], axis=1)
        gm = np.exp(np.mean(np.log(S_paths[:, 1:]), axis=1))
        Y_vals = np.maximum(gm - self.K, 0)
        payoffs = np.maximum(A_vals - self.K, 0) + b_opt * (Y_vals - esp_Y)
        return {"estim": np.mean(payoffs), "var_estim": np.var(payoffs, ddof=1) / num_simulations}

    def estim_control_estime_G(self, num_simulations, b_opt, inner_simulations=1000):
        """
        Estimation par variable de contrôle (avec estimation de E(Y) à chaque itération).
        Méthode moins efficace (et plus lente).
        """
        payoffs = []
        for _ in range(num_simulations):
            S = self.simul_asset()
            Y_val = self.simulate_Y(S)
            esp_Y_est = self.estimate_Y(inner_simulations)
            A_val = np.mean(S[1:])
            payoff = max(A_val - self.K, 0) + b_opt * (Y_val - esp_Y_est)
            payoffs.append(payoff)
        payoffs = np.array(payoffs)
        return {"estim": np.mean(payoffs), "var_estim": np.var(payoffs, ddof=1) / num_simulations}

    def g(self, Z):
        """
        Calcule g(Z) pour un vecteur gaussien Z de dimension n.
        On utilise la formule :
          g(Z) = max( (1/n) * somme_{i=1}^{n} exp(-0.5*sigma^2*Tk[i] + sigma * W_t_k(i,Z)) - K, 0)
        où W_t_k(i,Z) = sqrt(dt)*cumsum(Z)[i-1].
        """
        cs = np.cumsum(Z) * np.sqrt(self.dt)
        exponent = -0.5 * self.sigma**2 * self.Tk[1:] + self.sigma * cs
        total = np.sum(np.exp(exponent))
        return max(total / self.n - self.K, 0)

    def g_vec(self, Z_mat):
        """
        Version vectorisée de g(Z) pour un tableau Z_mat de dimension (num_simulations, n).
        """
        cs = np.cumsum(Z_mat, axis=1) * np.sqrt(self.dt)
        exponent = -0.5 * self.sigma**2 * self.Tk[1:] + self.sigma * cs
        total = np.sum(np.exp(exponent), axis=1)
        return np.maximum(total / self.n - self.K, 0)

    def estim_class_g_Z(self, num_simulations):
        """
        Estimation de E(g(Z)) par MC classique en générant num_simulations vecteurs gaussiens.
        """
        Z = np.random.randn(num_simulations, self.n)
        payoffs = self.g_vec(Z)
        return np.mean(payoffs)

    def estim_antith_G(self, num_simulations):
        """
        Estimation antithétique de G.
        Pour chaque vecteur Z simulé, on utilise aussi -Z et on prend la moyenne.
        """
        Z = np.random.randn(num_simulations, self.n)
        payoff1 = self.g_vec(Z)
        payoff2 = self.g_vec(-Z)
        S_vals = payoff1 + payoff2
        return {"estim": np.mean(S_vals) / 2, "var": np.var(S_vals, ddof=1) / (4 * num_simulations)}

    # --- Méthodes pour l'échantillonnage préférentiel (récurrence non vectorisée) ---

    def suite_z(self, y, j, Sj, zj):
        if j == 0:
            return self.sigma * np.sqrt(self.delta) * (y + self.K) / y
        else:
            return zj - self.sigma * np.sqrt(self.delta) * Sj / (self.n * y)

    def suite_S(self, j, y, zj):
        return self.S0 * np.exp(-0.5 * self.sigma**2 * self.Tk[j + 1] + self.sigma * np.sqrt(self.delta) * zj)

    def vect_z(self, y):
        """
        Renvoie un tuple (Z, S) où :
          - Z est le vecteur [z_1, ..., z_n] obtenu par récurrence,
          - S est le vecteur [S_1, ..., S_n] calculé avec la récurrence.
        """
        Z = np.empty(self.n)
        S_vals = np.empty(self.n)
        Z[0] = self.suite_z(y, 0, 0, 0)
        for k in range(1, self.n):
            # On utilise la somme des z précédents pour calculer S
            Sk = self.suite_S(k, y, np.sum(Z[:k]))
            S_vals[k - 1] = Sk
            Z[k] = self.suite_z(y, k, Sk, Z[k - 1])
        S_vals[self.n - 1] = self.suite_S(self.n, y, np.sum(Z))
        return Z, S_vals

    def prob(self, y):
        """
        Fonction à racine : prob(y) = moyenne(S) - K - y, où S est obtenu par vect_z(y).
        """
        _, S_vals = self.vect_z(y)
        return np.mean(S_vals) - self.K - y

    def w(self, Z, mu):
        """
        Calcule le poids w = exp(0.5*||mu||² - mu·Z).
        Z et mu sont des vecteurs 1D.
        """
        return np.exp(0.5 * np.sum(mu**2) - np.sum(Z * mu))

    def estim_pref_G(self, num_simulations, mu_opt):
        """
        Estimation par échantillonnage préférentiel de G.
          - mu_opt est le vecteur optimal (de dimension n).
        """
        Z = np.random.randn(num_simulations, self.n)
        Z_tilde = Z + mu_opt  # décalage optimal
        payoffs = self.g_vec(Z_tilde)
        weights = np.exp(0.5 * np.sum(mu_opt**2) - np.sum(Z_tilde * mu_opt, axis=1))
        adjusted_payoffs = payoffs * weights
        return {"esp_pref": np.mean(adjusted_payoffs), "var_pref": np.var(adjusted_payoffs, ddof=1) / num_simulations}

    def compute_esp_Y_theoretical(self):
        """
        Calcule l'espérance théorique de Y d'après la formule utilisée dans le code R original.
        """
        a = (self.sigma / 4) * np.sqrt(6 * (self.n + 1) / (2 * self.n + 1))
        mu_val = (self.sigma / self.n) * np.sqrt((self.n + 1) * (2 * self.n + 1) / 6)
        const = np.exp(-self.sigma**2 * (self.n + 1) / (4 * self.n) + 0.5 * mu_val**2)
        esp_Y = const * (1 - norm.cdf(a, loc=mu_val, scale=1)) - (1 - norm.cdf(a, loc=0, scale=1))
        return esp_Y
