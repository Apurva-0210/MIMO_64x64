import numpy as np

# ─── Constants ───────────────────────────────────────────────
TIME_STEPS = 10
ULA_RHO    = 0.7
JAKES_N    = 32
VELOCITY   = 120
FC         = 28e9


# ─── ULA spatial correlation ──────────────────────────────────
def ula_corr(n, rho=ULA_RHO):
    i = np.arange(n)
    return (rho ** np.abs(i[:, None] - i[None, :])).astype(np.complex128)


def apply_spatial(H, R_tx, R_rx):
    L_tx = np.linalg.cholesky(R_tx + 1e-12 * np.eye(len(R_tx)))
    L_rx = np.linalg.cholesky(R_rx + 1e-12 * np.eye(len(R_rx)))
    return L_rx @ H @ L_tx.conj().T


# ─── Jakes Doppler ────────────────────────────────────────────
def jakes(fd, t, N=JAKES_N, rng=None):
    if rng is None: rng = np.random.default_rng()
    alpha  = 2 * np.pi * np.arange(1, N+1) / N
    phi    = rng.uniform(-np.pi, np.pi, N)
    phases = 2*np.pi*fd*np.cos(alpha)[:,None]*t[None,:] + phi[:,None]
    return np.sum(np.exp(1j*phases), axis=0) / np.sqrt(N)


# ─── Jakes channel (Stage I) ──────────────────────────────────
def gen_jakes_channel(Nr, Nt, velocity_kmh=VELOCITY, fc=FC,
                       num_steps=TIME_STEPS):
    """Simple Jakes channel: H_0 * exp(j 2pi fd t)."""
    c = 3e8; v = velocity_kmh / 3.6; fd = v * fc / c
    t = np.arange(num_steps) / 1e6
    H_base = (np.random.randn(Nr, Nt) +
              1j * np.random.randn(Nr, Nt)) / np.sqrt(2)
    H = []
    for step in t:
        ph = np.exp(1j * 2 * np.pi * fd * step)
        Hs = H_base * ph
        H.append(np.stack([Hs.real, Hs.imag], axis=-1))
    return np.array(H, dtype=np.float32)


# ─── 3GPP TR 38.901 TDL tap tables ────────────────────────────
TDL_A = [(0.0000,-13.4),(0.3819,0.0),(0.4025,-2.2),(0.5868,-4.0),
         (0.4610,-6.0),(0.5375,-8.2),(0.6708,-9.9),(0.5750,-10.5),
         (0.7618,-7.5),(1.5375,-15.9),(1.8978,-6.6),(2.2242,-16.7),
         (2.1718,-12.4),(2.4942,-15.2),(2.5119,-10.8),(3.0582,-11.3),
         (4.0810,-12.7),(4.4579,-16.2),(4.5695,-18.3),(4.7966,-18.9),
         (5.0066,-16.6),(5.3043,-19.9),(9.6586,-29.7)]
TDL_B = [(0.0000,0.0),(0.1072,-2.2),(0.2155,-4.0),(0.2095,-3.2),
         (0.2870,-9.8),(0.2986,-1.2),(0.3752,-3.4),(0.5055,-5.2),
         (0.3681,-7.6),(0.3697,-3.0),(0.5700,-8.9),(0.5283,-9.0),
         (1.1021,-4.8),(1.2756,-5.7),(1.5474,-7.5),(1.7842,-1.9),
         (2.0169,-7.6),(2.8294,-12.2),(3.0219,-9.8),(3.6187,-11.4),
         (4.1067,-14.9),(4.2790,-9.2),(4.7834,-11.3)]
TDL_C = [(0.0000,-4.4),(0.2099,-1.2),(0.2219,-3.5),(0.2329,-5.2),
         (0.2176,-2.5),(0.6366,0.0),(0.6448,-2.2),(0.6560,-3.9),
         (0.6584,-7.4),(0.7935,-7.1),(0.8213,-10.7),(0.9336,-11.1),
         (1.2285,-5.1),(1.3083,-6.8),(2.1704,-8.7),(2.7105,-13.2),
         (4.2589,-13.9),(4.6003,-13.9),(5.4902,-15.8),(5.6077,-17.1),
         (6.3065,-16.0),(6.6374,-15.7),(7.0427,-21.6),(8.6523,-22.8)]
TDL_D = [(0.000,-0.2,True),(0.000,-13.5,False),(0.035,-18.8,False),
         (0.612,-21.0,False),(1.363,-22.8,False),(1.405,-17.9,False),
         (1.804,-20.1,False),(2.596,-21.9,False),(1.775,-22.9,False),
         (4.042,-27.8,False),(7.937,-23.6,False),(9.424,-24.8,False),
         (9.708,-30.0,False),(12.525,-27.7,False)]
TDL_E = [(0.0000,-0.03,True),(0.0000,-22.03,False),
         (0.5133,-15.8,False),(0.5440,-18.1,False),(0.5630,-19.8,False),
         (0.5440,-22.9,False),(0.7112,-22.4,False),(1.9092,-18.6,False),
         (1.9293,-20.8,False),(1.9589,-22.6,False),(2.6426,-22.3,False),
         (3.7136,-25.6,False),(5.4524,-20.2,False),(12.0034,-29.8,False),
         (20.6519,-29.2,False)]


# ─── 10 deployment scenarios (TR 38.901) ──────────────────────
SCENARIOS = {
    'InH_LOS':  {'taps':TDL_D,'v':3,  'fc':28e9, 'los':True, 'K':13.3,'group':'LOS', 'desc':'Indoor Hotspot LOS'},
    'UMi_LOS':  {'taps':TDL_D,'v':30, 'fc':28e9, 'los':True, 'K':9.0, 'group':'LOS', 'desc':'Urban Micro LOS'},
    'UMa_LOS':  {'taps':TDL_D,'v':120,'fc':28e9, 'los':True, 'K':9.0, 'group':'LOS', 'desc':'Urban Macro LOS 120 km/h'},
    'RMa_LOS':  {'taps':TDL_D,'v':300,'fc':3.5e9,'los':True, 'K':7.0, 'group':'LOS', 'desc':'Rural Macro LOS 300 km/h'},
    'InH_NLOS': {'taps':TDL_A,'v':3,  'fc':28e9, 'los':False,'K':None,'group':'NLOS','desc':'Indoor Hotspot NLOS'},
    'UMi_NLOS': {'taps':TDL_A,'v':30, 'fc':28e9, 'los':False,'K':None,'group':'NLOS','desc':'Urban Micro NLOS'},
    'UMa_NLOS': {'taps':TDL_B,'v':120,'fc':28e9, 'los':False,'K':None,'group':'NLOS','desc':'Urban Macro NLOS 120 km/h'},
    'RMa_NLOS': {'taps':TDL_C,'v':300,'fc':3.5e9,'los':False,'K':None,'group':'NLOS','desc':'Rural Macro NLOS 300 km/h'},
    'UMi_O2I':  {'taps':TDL_C,'v':3,  'fc':28e9, 'los':False,'K':None,'group':'O2I', 'desc':'Urban Micro Outdoor-to-Indoor'},
    'RMa_HST':  {'taps':TDL_E,'v':500,'fc':3.5e9,'los':True, 'K':22.0,'group':'HST', 'desc':'Rural Macro High-Speed Train 500 km/h'},
}
SNAMES = list(SCENARIOS.keys())


# ─── TDL channel generator ────────────────────────────────────
def gen_tdl_channel(name, Nr, Nt, num_steps=TIME_STEPS):
    s = SCENARIOS[name]
    taps, v_kmh, fc = s['taps'], s['v'], s['fc']
    is_los, K_db = s['los'], s['K']
    c = 3e8; v = v_kmh / 3.6; fd = v * fc / c
    t = np.arange(num_steps) / 1e6
    rng = np.random.default_rng()
    R_tx = ula_corr(Nt); R_rx = ula_corr(Nr)
    tot  = sum(10**(e[1]/10) for e in taps)
    H    = np.zeros((num_steps, Nr, Nt), dtype=np.complex128)
    for tap in taps:
        is_los_tap = tap[2] if len(tap)==3 else False
        pwr = 10**(tap[1]/10)
        Hi  = (rng.standard_normal((Nr,Nt)) +
               1j*rng.standard_normal((Nr,Nt)))/np.sqrt(2)
        Hi  = apply_spatial(Hi, R_tx, R_rx)
        if is_los_tap and is_los and K_db is not None:
            K = 10**(K_db/10)
            lph = np.exp(1j*2*np.pi*fd*t)
            Hl  = apply_spatial(np.ones((Nr,Nt),dtype=np.complex128),
                                R_tx, R_rx)
            Hd = np.sqrt(K/(K+1)) * np.einsum('t,ij->tij', lph, Hl)
            hj = jakes(fd, t, rng=rng)
            Hs = np.sqrt(1/(K+1)) * np.einsum('t,ij->tij', hj, Hi)
            Ht = Hd + Hs
        else:
            hj = jakes(fd, t, rng=rng)
            Ht = np.einsum('t,ij->tij', hj, Hi)
            if is_los and K_db is not None:
                Ht *= np.sqrt(1/(10**(K_db/10)+1))
        H += np.sqrt(pwr) * Ht
    H /= np.sqrt(tot)
    return np.stack([H.real, H.imag], axis=-1).astype(np.float32)


def gen_tdl_random(Nr, Nt, num_steps=TIME_STEPS):
    """Random scenario each call — used in TDL training."""
    name = np.random.choice(SNAMES)
    return gen_tdl_channel(name, Nr, Nt, num_steps)


def gen_combined_channel(Nr, Nt, num_steps=TIME_STEPS, p_jakes=0.5):
    if np.random.random() < p_jakes:
        return gen_jakes_channel(Nr, Nt, num_steps=num_steps)
    return gen_tdl_random(Nr, Nt, num_steps=num_steps)
