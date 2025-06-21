import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from typing import Tuple, Dict


regular_spiking_params = {
    'a': 0.02,
    'b': 0.2,
    'c': -65,
    'd': 8,
    'v0': -65,
    'I': 10,
    'tt': 'Regular spiking'
}

fast_spiking_params = {
    'a': 0.1,
    'b': 0.2,
    'c': -65,
    'd': 2,
    'v0': -70,
    'I': 15,
    'tt': 'Fast spiking'
}

low_threshold_spiking_params = {
    'a': 0.02,
    'b': 0.25,
    'c': -65,
    'd': 2,
    'v0': -70,
    'I': 7,
    'tt': 'Low-threshold spiking'
}

resonator_params = {
    'a': 0.1,
    'b': 0.26,
    'c': -65,
    'd': 2,
    'v0': -65,
    'I': 10,
    'tt': 'Resonator'
}

intrinsically_bursting_params = {
    'a': 0.02,
    'b': 0.2,
    'c': -55,
    'd': 4,
    'v0': -60,
    'I': 10,
    'tt': 'Intrinsically bursting'
}

chattering_params = {
    'a': 0.02,
    'b': 0.2,
    'c': -50,
    'd': 2,
    'v0': -65,
    'I': 10,
    'tt': 'Chattering'
}


def izhikevich_model(state, t, params):
    v, u = state
    a, b, I = params['a'], params['b'], params['I']

    dvdt = 0.04 * v**2 + 5 * v + 140 - u + I
    dudt = a * (b * v - u)
    return [dvdt, dudt]


def simulate_izhikevich(params, t_max=200, dt=0.001):

    a, b, c, d = params['a'], params['b'], params['c'], params['d']
    v0, u0 = params['v0'], params['b'] * params['v0']
    I = params['I']

    t = np.arange(0, t_max, dt)
    n_steps = len(t)

    v = np.zeros(n_steps)
    u = np.zeros(n_steps)
    v[0], u[0] = v0, u0

    for i in range(1, n_steps):
        state = [v[i-1], u[i-1]]
        new_state = odeint(izhikevich_model, state, [0, dt], args=(params,))[1]

        v[i], u[i] = new_state

        if v[i] >= 30:
            v[i] = c
            u[i] += d

    return t, v, u


t, v, u = simulate_izhikevich(chattering_params)


DARK_RED = '#c91010'
DARK_GREEN = '#23a118'

plt.figure(figsize=(12, 5))
plt.suptitle(chattering_params['tt'], fontsize=14, y=1.02)

plt.subplot(1, 2, 1)
plt.plot(t, v, color=DARK_RED)
plt.title('Membrane Potential')
plt.legend('v(t)')
plt.xlabel('t, ms')
plt.ylabel('Potential, mV')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(t, u, color=DARK_GREEN)
plt.title('Recovery Variable')
plt.legend('u(t)')
plt.xlabel('t, ms')
plt.ylabel('Value')
plt.grid(True)

plt.tight_layout()
plt.savefig('chattering.png', dpi=300, bbox_inches='tight')
plt.show()

from google.colab import files
files.download('chattering.png')

I_values = [0, 5, 10, 15, 20, 25]
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

plt.figure(figsize=(15, 10))

for idx, I in enumerate(I_values):
    params = regular_spiking_params.copy()
    params['I'] = I
    params['tt'] = f'I = {I}'
    
    t, v, u = simulate_izhikevich(params)
    
    plt.subplot(2, 3, idx+1)
    plt.plot(t, v, color=colors[idx])
    plt.title(f'I = {I} nA')
    plt.xlabel('t, ms')
    plt.ylabel('Potential, mV')
    plt.legend([f'v_{idx}(t)'])
    plt.grid(True)
    plt.ylim(-90, 40)


plt.suptitle(f"{regular_spiking_params['tt']}: Membrane Potential", fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('rs_different_I_potentials.png', dpi=300, bbox_inches='tight')

files.download('rs_different_I_potentials.png')


plt.figure(figsize=(15, 10))

for idx, I in enumerate(I_values):
    params = regular_spiking_params.copy()
    params['I'] = I
    params['tt'] = f'I = {I}'
    
    t, v, u = simulate_izhikevich(params)
    
    plt.subplot(2, 3, idx+1)
    plt.plot(t, u, color=colors[idx])
    plt.title(f'I = {I} nA')
    plt.xlabel('t, ms')
    plt.ylabel('Value')
    plt.legend([f'u_{idx}(t)'])
    plt.grid(True)

plt.suptitle(f"{regular_spiking_params['tt']}: Recovery Variable", fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('rs_different_I_recovery.png', dpi=300, bbox_inches='tight')
files.download('rs_different_I_recovery.png')

plt.show()