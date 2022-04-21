import numpy as np

def gray_scott_np(F, k, v_np, u_np):
    v2 = np.power(v_np, 2)
    uv2 = u_np * v2

    v_update = uv2 - (F + k) * v_np # Gray-Scott
    u_update = - uv2 + F * (1 - u_np) # Gray-Scott
    return v_update, u_update

def laplacian(a):
    return a[2:, 1:-1] + a[1:-1, 2:] + a[0:-2, 1:-1] + a[1:-1, 0:-2] - 4 * a[1:-1, 1:-1]

def gierer_mienhardt(v, u, rho, kappa, mu, nu):
    a_view = v[1:-1, 1:-1]
    h_view = u[1:-1, 1:-1]

    a2 = np.power(a_view, 2)
    ah2 = h_view * (1 + kappa * a2)
    a2_ah2 = a2 / ah2 

    a_update = rho * (a2_ah2 - mu * a_view) 
    h_update = rho * (a2 - nu * h_view)

    return a_update, h_update

def generalized_np(rho_np, kap_np, v_np, u_np):
    v_pows, u_pows = {}, {}
    for pow in range(3):
        v_pows[pow] = np.power(v_np, pow)
        u_pows[pow] = np.power(u_np, pow)

    updates = np.array((np.zeros(v_np.shape), np.zeros(u_np.shape)))

    for k in range(2):
        for l in range(3):
            num, den = 0, 0
            for i in range(3):
                for j in range(3):
                    num += rho_np[k][l][i][j] * v_pows[i] * u_pows[j]
                    den += kap_np[k][l][i][j] * v_pows[i] * u_pows[j]

            updates[k] += np.divide(num, den, where=np.abs(den)>0.0001)

    return updates