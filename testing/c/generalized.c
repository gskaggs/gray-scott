#define idx(x, y) x + block_size *y

__kernel void iterate(
    __global const float *rho_g, __global const float *kap_g,
    __global const float *v_g, __global const float *u_g,
    __global float *v_update_g)
{
    uint gidx = get_global_id(0);
    uint gidy = get_global_id(1);
    uint block_size = get_global_size(0);

    float v = v_g[idx(gidx, gidy)];
    float v2 = v * v;

    float u = u_g[idx(gidx, gidy)];
    float u2 = u * u;

    float pow_v[3] = {1, v, v2};
    float pow_u[3] = {1, u, u2};

    float result = 0;
    float eps = .0001;

    int term = 0;
    for (term = 0; term < 3; term++) {
        int i = 0;
        float nom = 0;
        float den = 0;
        for (i = 0; i < 3; i++) {
            int j = 0;
            for (j = 0; j < 3; j++) {
                float rho = rho_g[9 * term + 3 * i + j];
                float kap = kap_g[9 * term + 3 * i + j];

                nom += rho * pow_v[i] * pow_u[j];
                den += kap * pow_v[i] * pow_u[j];

                
            }
        }
        if (den < -eps || den > eps) 
            result += nom / den;
    }

    v_update_g[idx(gidx, gidy)] += result;
}