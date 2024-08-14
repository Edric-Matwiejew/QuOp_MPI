import numpy as np
import h5py


# Values from Particle Data Group: https://pdg.lbl.gov/
# SI units
G = 6.67430e-11
c = 299792458
Msol = 1.98841e30
MSOL_TO_SEC = G * Msol / c**3


# Mass coordinate transformations

def m1m2_to_Meta(m1, m2):
    return m1 + m2, m1*m2/(m1+m2)**2

def Meta_to_m1m2(M, eta):
    return (M/2)*(1 + np.sqrt(1-4*eta)), (M/2)*(1 - np.sqrt(1-4*eta))

def Meta_to_theta1eta(M, eta, f0):
    return (5/128) * (np.pi * M * MSOL_TO_SEC * f0)**(-5/3) / eta, eta

def theta1eta_to_Meta(theta1, eta, f0):
    return (eta * theta1 * 128/5)**(-3/5) / (np.pi * MSOL_TO_SEC * f0), eta

def m1m2_to_theta1eta(m1, m2, f0):
    M, eta = m1m2_to_Meta(m1, m2)
    return Meta_to_theta1eta(M, eta, f0)

def theta1eta_to_m1m2(theta1, eta, f0):
    M, eta = theta1eta_to_Meta(theta1, eta, f0)
    return Meta_to_m1m2(M, eta)


# TaylorF2 template phase function is from https://doi.org/10.1103/PhysRevD.80.084043

def template_taylorf2( freqs, M, eta, tc=0 ):
    v_lso = 1/np.sqrt(6)
    v = np.real(( np.pi * M * MSOL_TO_SEC * freqs )**(1/3))
    phase = 2*np.pi*freqs*tc + (3/(128*eta*v**5)) * ( 1 + (20/9)*(743/336 + 11*eta/4)*v**2 - 16*np.pi*v**3 + 10*(3058673/1016064 + 5429*eta/1008 + 617*eta**2/144)*v**4 + np.pi*(38645/756 - 65*eta/9)*(1 + 3*np.log(v/v_lso))*v**5 + (11583231236531/4694215680 - 640*np.pi**2/3 - 6848*np.euler_gamma/21 - 6848*np.log(4*v)/21 + (-15737765635/3048192 + 2255*np.pi**2/12)*eta + 76055*eta**2/1728 - 127825*eta**3/1296)*v**6 + np.pi*(77096675/254016 + 378515*eta/1512 - 74045*eta**2/756)*v**7 )
    return {"waveform": np.exp(-1j * phase) / freqs**(7/6), "freqs": freqs, "df": freqs[1]-freqs[0]}

def template_overlap( h1, h2, noise_psd ):
    h1w = h1["waveform"]
    h2w = h2["waveform"]
    nw = noise_psd["waveform"]
    df = noise_psd["df"]
    return 4 * df * sum( np.conj(h1w)*h2w  / nw )


# Grid setup functions
# Shifted to ensure signal parameters correspond to one of the grid points

def grid_setup_m1m2(Ns, min_mass, max_mass, m1_signal, m2_signal):

    Ntotal = np.prod(Ns)

    m1_min, m1_max = min_mass[0], max_mass[0]
    m2_min, m2_max = min_mass[1], max_mass[1]
    m1_list = np.linspace(m1_min, m1_max, Ns[0])
    m2_list = np.linspace(m2_min, m2_max, Ns[1])

    m1_nearest_ix = np.argmin( np.abs(m1_list - m1_signal) )
    m1_shift = m1_signal - m1_list[m1_nearest_ix]
    m1s = m1_list + m1_shift

    m2_nearest_ix = np.argmin( np.abs(m2_list - m2_signal) )
    m2_shift = m2_signal - m2_list[m2_nearest_ix]
    m2s = m2_list + m2_shift

    m1_grid, m2_grid = np.meshgrid(m1s, m2s, indexing='ij')
    m1_grid = m1_grid.reshape(Ntotal)
    m2_grid = m2_grid.reshape(Ntotal)

    return m1_grid, m2_grid

def grid_setup_theta1eta(Ns, min_mass, max_mass, m1_signal, m2_signal, f0):

    Ntotal = np.prod(Ns)

    M_max = np.sum(max_mass)
    M_min = np.sum(min_mass)
    eta_max = 0.25
    eta_min = min(min_mass)*max(max_mass)/(min(min_mass) + max(max_mass))**2

    theta1_0, eta_0 = Meta_to_theta1eta(M_max, eta_max, f0)
    theta1_1, eta_1 = Meta_to_theta1eta(M_max, eta_min, f0)
    theta1_2, eta_2 = Meta_to_theta1eta(M_min, eta_max, f0)
    theta1_3, eta_3 = Meta_to_theta1eta(M_min, eta_min, f0)

    theta1_min = min(theta1_0, theta1_1, theta1_2, theta1_3)
    eta_min = min(eta_0, eta_1, eta_2, eta_3)
    theta1_max = max(theta1_0, theta1_1, theta1_2, theta1_3)
    eta_max = max(eta_0, eta_1, eta_2, eta_3)

    theta1_list = np.linspace(theta1_min, theta1_max, Ns[0])
    eta_list = np.linspace(eta_min, eta_max, Ns[1])
    M_signal, eta_signal = m1m2_to_Meta(m1_signal, m2_signal)
    theta1_signal, eta_signal = Meta_to_theta1eta(M_signal, eta_signal, f0)

    theta1_nearest_ix = np.argmin( np.abs(theta1_list - theta1_signal) )
    theta1_shift = theta1_signal - theta1_list[theta1_nearest_ix]
    theta1s = theta1_list + theta1_shift

    eta_nearest_ix = np.argmin( np.abs(eta_list - eta_signal) )
    eta_shift = eta_signal - eta_list[eta_nearest_ix]
    etas = eta_list + eta_shift

    theta1_grid, eta_grid = np.meshgrid(theta1s, etas, indexing='ij')
    theta1_grid = theta1_grid.reshape(Ntotal)
    eta_grid = eta_grid.reshape(Ntotal)

    return theta1_grid, eta_grid


# Generating cost function files

def write_cost_values(cost_file, input_path_strain, input_path_psd, cds, Ns, m1_signal, m2_signal, tc_signal, min_mass, max_mass):

    Ntotal = np.prod(Ns)

    f_strain = np.loadtxt(input_path_strain)
    time = f_strain[0,:]
    strain = f_strain[1,:]

    N_time = len(strain)
    fs = 1/(time[1]-time[0])
    df = fs/N_time

    strain_dft = np.fft.fft(strain) / fs
    freqs_dft = np.fft.fftfreq(N_time, 1/fs)

    # Truncate frequencies
    fL = 20
    kL = int( max(1, np.ceil(N_time*fL/fs)) )

    v_lso = 1/np.sqrt(6)
    M_max = 12
    f_lso = v_lso**3 / (np.pi * M_max * MSOL_TO_SEC)
    kH = int( min(np.floor((N_time-1)/2), np.floor(N_time*f_lso/fs)) )

    freqs = freqs_dft[kL:kH+1]
    data_series = {'waveform': strain_dft[kL:kH+1], 'freqs': freqs, 'df': df}

    f_psd = np.loadtxt(input_path_psd)
    freqs_psd = f_psd[0,:]
    psd = f_psd[1,:]

    if not np.array_equal(freqs, freqs_psd[kL:kH+1]):
        raise Exception('Signal and noise frequencies do not match')

    noise_psd = {'waveform': psd[kL:kH+1], 'freqs': freqs, 'df': df}


    if cds == 'm1m2':
        cd1_grid, cd2_grid = grid_setup_m1m2(Ns, min_mass, max_mass, m1_signal, m2_signal)
        M_grid, eta_grid = m1m2_to_Meta(cd1_grid, cd2_grid)
        cd1_signal, cd2_signal = m1_signal, m2_signal
    elif cds == 'theta1eta':
        cd1_grid, cd2_grid = grid_setup_theta1eta(Ns, min_mass, max_mass, m1_signal, m2_signal, fs)
        M_grid, eta_grid = theta1eta_to_Meta(cd1_grid, cd2_grid, fs)
        cd1_signal, cd2_signal = m1m2_to_theta1eta(m1_signal, m2_signal, fs)
    else:
        raise Exception('Coordinates ' + cds + ' not recognized.')


    C = np.empty(Ntotal)
    for ix in range(Ntotal):
        h = template_taylorf2(freqs, M_grid[ix], eta_grid[ix], tc_signal)
        h['waveform'] = h['waveform'] / np.sqrt( abs(template_overlap(h, h, noise_psd)) )
        C[ix] = -abs(template_overlap(data_series, h, noise_psd))
    
    hf = h5py.File(cost_file, 'w')
    hf.create_dataset('cost_values', data=C)
    hf.create_dataset('cd1', data=cd1_grid)
    hf.create_dataset('cd2', data=cd2_grid)
    hf.create_dataset('cd1_signal', data=cd1_signal)
    hf.create_dataset('cd2_signal', data=cd2_signal)
    hf.create_dataset('tc_signal', data=tc_signal)
    hf.close()

    return


# Functions used in QVAs to read cost function values

def cost_values_from_file(
        local_i,
        local_i_offset,
        MPI_COMM,
        Ns,
        strides,
        deltas,
        mins,
        d,
        cost_file):

    hf = h5py.File(cost_file)
    C = hf['cost_values'][...]
    hf.close()

    return C[local_i_offset:local_i_offset+local_i]


def cost_values_from_file_binary(
        local_i,
        local_i_offset,
        MPI_COMM,
        Ns,
        strides,
        deltas,
        mins,
        d,
        threshold,
        cost_file):

    hf = h5py.File(cost_file)
    C = hf['cost_values'][...]
    hf.close()

    return -1*( C[local_i_offset:local_i_offset+local_i] < -threshold )

