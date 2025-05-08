# Import des bibliothèques nécessaires
import streamlit as st                # Pour créer l'interface web interactive
import numpy as np                   # Pour les calculs numériques
import matplotlib.pyplot as plt      # Pour afficher les courbes

# Définition de la classe pour pricer une option barrière via Monte Carlo
class MonteCarloBarrierOption:
    def __init__(self, S0, K, barrier, T, r, sigma, option_type, barrier_type, N=10000, n=252, seed=42):
        # Initialisation des paramètres de l'option
        self.S0 = S0                      # Prix initial du sous-jacent
        self.K = K                        # Prix d'exercice
        self.barrier = barrier            # Niveau de la barrière
        self.T = T                        # Maturité en années
        self.r = r                        # Taux sans risque
        self.sigma = sigma                # Volatilité du sous-jacent
        self.option_type = option_type    # 'call' ou 'put'
        self.barrier_type = barrier_type  # 'up-and-in', 'down-and-out', etc.
        self.N = N                        # Nombre de simulations Monte Carlo
        self.n = n                        # Nombre de pas de temps
        self.dt = T / n                   # Pas de temps (delta t)
        self.sqrt_dt = np.sqrt(self.dt)   # Racine de dt pour Brownien
        self.seed = seed                  # Graine aléatoire pour reproductibilité

    # Simulation de trajectoires pour le sous-jacent
    def simulate_paths(self, S0=None):
        if S0 is None:
            S0 = self.S0
        np.random.seed(self.seed)
        z = np.random.randn(self.N, self.n)  # Génération des chocs aléatoires
        increments = (self.r - 0.5 * self.sigma**2) * self.dt + self.sigma * self.sqrt_dt * z
        increments = np.cumsum(increments, axis=1)  # Intégration des chocs
        log_paths = np.log(S0) + increments         # Log des trajectoires
        paths = np.exp(log_paths)                   # Conversion en prix
        paths = np.insert(paths, 0, S0, axis=1)      # Ajout du point initial
        return paths

    # Calcul du prix de l'option en utilisant les trajectoires simulées
    def price(self, S0=None):
        paths = self.simulate_paths(S0)
        payoff = np.zeros(self.N)
        for j in range(self.N):
            S_path = paths[j, :]
            S_T = S_path[-1]  # Prix final du sous-jacent
            breached = np.any(S_path >= self.barrier) if 'up' in self.barrier_type else np.any(S_path <= self.barrier)
            in_condition = breached
            out_condition = not breached

            # Calcul du payoff intrinsèque
            intrinsic = max(S_T - self.K, 0) if self.option_type == 'call' else max(self.K - S_T, 0)

            # Attribution du payoff selon le type de barrière
            if 'in' in self.barrier_type and in_condition:
                payoff[j] = intrinsic
            elif 'out' in self.barrier_type and out_condition:
                payoff[j] = intrinsic

        return np.exp(-self.r * self.T) * np.mean(payoff)  # Actualisation du payoff

    # Estimation du Gamma via méthode des différences finies centrées
    def estimate_gamma(self, eps=1):
        base_seed = self.seed
        price_up = MonteCarloBarrierOption(self.S0 + eps, self.K, self.barrier, self.T, self.r, self.sigma,
                                           self.option_type, self.barrier_type, self.N, self.n, base_seed).price()
        price_0 = MonteCarloBarrierOption(self.S0, self.K, self.barrier, self.T, self.r, self.sigma,
                                          self.option_type, self.barrier_type, self.N, self.n, base_seed).price()
        price_down = MonteCarloBarrierOption(self.S0 - eps, self.K, self.barrier, self.T, self.r, self.sigma,
                                             self.option_type, self.barrier_type, self.N, self.n, base_seed).price()
        return (price_up - 2 * price_0 + price_down) / (eps ** 2)

    # Calcul du Gamma pour différents niveaux de sous-jacent
    def compute_gamma_curve(self, S_min, S_max, steps=15, eps=1.0):
        S_vals = np.linspace(S_min, S_max, steps)
        gammas = []
        for S in S_vals:
            pricer = MonteCarloBarrierOption(S, self.K, self.barrier, self.T, self.r, self.sigma,
                                             self.option_type, self.barrier_type, self.N, self.n, seed=self.seed)
            gamma = pricer.estimate_gamma(eps)
            gammas.append(gamma)

        # Lissage de la courbe pour meilleure lisibilité
        gammas = np.convolve(gammas, np.ones(3)/3, mode='same')
        return S_vals, gammas

# Interface utilisateur avec Streamlit
st.title("Visualisation du Gamma autour d'une barrière")

# Paramètres à ajuster via sliders
option_type = st.selectbox("Type d'option", ['call', 'put'])
barrier_type = st.selectbox("Type de barrière", ['up-and-in', 'up-and-out', 'down-and-in', 'down-and-out'])

S0 = st.slider("Prix initial du sous-jacent (S0)", 50, 150, 100)
K = st.slider("Strike (K)", 50, 150, 100)
barrier = st.slider("Barrière", 50, 150, 110)
T = st.slider("Maturité (T en années)", 0.25, 2.0, 1.0)
r = st.slider("Taux sans risque (r)", 0.0, 0.1, 0.03)
sigma = st.slider("Volatilité (sigma)", 0.1, 1.0, 0.3)

S_min = st.slider("Minimum S pour le gamma", 50, 130, 50)
S_max = st.slider("Maximum S pour le gamma", 70, 170, 150)

# Bouton pour afficher la courbe de Gamma
if st.button("Afficher le Gamma"):
    option = MonteCarloBarrierOption(S0, K, barrier, T, r, sigma, option_type, barrier_type)
    S_vals, gammas = option.compute_gamma_curve(S_min, S_max)

    # Affichage du graphe
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(S_vals, gammas, label='Gamma')
    ax.axvline(barrier, color='red', linestyle='--', label='Barrière')
    ax.set_xlabel('Prix du sous-jacent')
    ax.set_ylabel('Gamma')
    ax.set_title(f'Gamma autour de la barrière ({barrier}) - {option_type} {barrier_type}')
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)
