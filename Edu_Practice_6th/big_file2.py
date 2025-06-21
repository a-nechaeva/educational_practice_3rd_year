import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from google.colab import files


rs_params = { 
    'a': 0.02,
    'b': 0.2,
    'c': -65,
    'd': 8,
    'v0': -65,
    'I': 10  
}

fs_params = {  
    'a': 0.1,   
    'b': 0.2,
    'c': -65,
    'd': 2,     
    'v0': -65,
    'I': 10
}

sigmas = [0, 0.5, 1, 2, 10]

def izhikevich_model_net(state, t, a1, b1, I1, a2, b2, I2, sigma_):
    v1, u1, v2, u2 = state
    dv1dt = 0.04*v1**2 + 5*v1 + 140 - u1 + I1 + sigma_*(v2 - v1)
    du1dt = a1*(b1*v1 - u1)

    dv2dt = 0.04*v2**2 + 5*v2 + 140 - u2 + I2 + sigma_*(v1 - v2)
    du2dt = a2*(b2*v2 - u2)
    return [dv1dt, du1dt, dv2dt, du2dt]

def simulate_izhikevich_net(sigma_, t_max=200, dt=0.01):
    t = np.arange(0, t_max, dt)
    state = np.array([
        rs_params['v0'],  
        rs_params['b'] * rs_params['v0'],
        fs_params['v0'],  
        fs_params['b'] * fs_params['v0']
    ])
    
    v_rs = np.zeros_like(t)
    v_fs = np.zeros_like(t)
    
    for i in range(len(t)):
        new_state = odeint(
            izhikevich_model_net, 
            state, 
            [0, dt], 
            args=(
                rs_params['a'], rs_params['b'], rs_params['I'],
                fs_params['a'], fs_params['b'], fs_params['I'],
                sigma_
            )
        )[-1]
        
  
        if new_state[0] >= 30: 
            new_state[0] = rs_params['c']
            new_state[1] += rs_params['d']
        if new_state[2] >= 30:  
            new_state[2] = fs_params['c']
            new_state[3] += fs_params['d']
        
        v_rs[i], v_fs[i] = new_state[0], new_state[2]
        state = new_state
    
    return t, v_rs, v_fs


def sync_level(v1, v2):
    return np.mean((v1 - v2)**2) 

DARK_RED = '#c91010'
DARK_GREEN = '#23a118'

plt.figure(figsize=(15, 12))

for i, sigma_ in enumerate(sigmas):
    t, v_rs, v_fs = simulate_izhikevich_net(sigma_)
    sync = sync_level(v_rs, v_fs)
    
    plt.subplot(len(sigmas), 1, i+1)
    plt.plot(t, v_rs, color=DARK_RED, label='RS neuron')
    plt.plot(t, v_fs, color=DARK_GREEN, label='FS neuron',  linestyle=':')
    plt.title(f'σ={sigma_}, s={sync:.2f}')
    plt.ylabel('v, mV')
    plt.legend()
    plt.grid(True)
    plt.yticks([-75, -50, -25, 0, 25])

plt.xlabel('t, ms')
plt.suptitle(f"Membrane Potential for RS-FS", fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('v_rsfs.png', dpi=300, bbox_inches='tight')
files.download('v_rsfs.png')
plt.show()


sync_values = [sync_level(*simulate_izhikevich_net(sigma_)[1:]) for sigma_ in sigmas]

plt.figure(figsize=(10, 5))
plt.plot(sigmas, sync_values, 'co-', label=r'$s(\sigma)$')
plt.legend()
plt.xlabel('σ')
plt.ylabel('s')
plt.title('Synchronization for RS-FS')
plt.grid(True)
plt.savefig('sync.png', dpi=300, bbox_inches='tight')
files.download('sync.png')
plt.show()