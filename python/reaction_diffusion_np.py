import numpy as np

def gray_scott_np(F, k, v_np, u_np):
    v2 = np.power(v_np, 2)
    uv2 = u_np * v2

    v_update = uv2 - (F + k) * v_np # Gray-Scott
    u_update = - uv2 + F * (1 - u_np) # Gray-Scott
    return v_update, u_update


def generalized_np(rho_np, kap_np, v_np, u_np):
    v_pows, u_pows = {}, {}
    for pow in range(-2, 3):
        v_pows[pow] = np.power(v_np, abs(pow))
        u_pows[pow] = np.power(u_np, abs(pow))

    updates = np.array((np.zeros(v_np.shape), np.zeros(u_np.shape)))

    for k in range(2):
        for i in range(-2, 3):
            for j in range(-2, 3):
                rho, kappa = rho_np[k][i+2, j+2], kap_np[k][i+2, j+2]

                if i >= 0 and j >= 0:
                    num = rho * v_pows[i] * u_pows[j]
                    den = 1

                elif i >= 0:
                    num = rho * v_pows[i]
                    den = kappa + u_pows[j]

                elif j >= 0:
                    num = rho * u_pows[j]
                    den = kappa + v_pows[i]
                    
                else:
                    num = rho 
                    den = kappa + v_pows[i] * u_pows[j]

                updates[k] += np.divide(num, den, where=np.abs(den)>0.0001)

    return updates