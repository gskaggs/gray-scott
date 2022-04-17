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

    v_update_g[idx(gidx, gidy)] = 0;

    int i = -2;
    for (i = -2; i < 3; i++)
    {
        int j = -2;
        for (j = -2; j < 3; j++)
        {
            int i2 = i + 2;
            int j2 = j + 2;
            float rho = rho_g[j2 + 5 * i2];
            float kap = kap_g[j2 + 5 * i2];
            float num = rho;
            float den = 1;

            if (i == -2) den *= v2;
            if (i == -1) den *= v;
            if (i == 1) num *= v;
            if (i == 2) num *= v2;

            if (j == -2) den *= u2;
            if (j == -1) den *= u;
            if (j == 1) num *= u;
            if (j == 2) num *= u2;

            if (i < 0 || j < 0) den += kap;
            
            if (!(den < 0.0001 && den > -0.0001))
                v_update_g[idx(gidx, gidy)] += num / den;
        }
    }
}