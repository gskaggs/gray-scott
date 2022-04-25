#define idx(x, y) x + block_size *y

__kernel void iterate(
    __global const double *rho_g, __global const double *kap_g,
    __global const double *v_g, __global const double *u_g,
    __global double *v_update_g)
{
    uint gidx = get_global_id(0);
    uint gidy = get_global_id(1);
    uint block_size = get_global_size(0);

    double v = v_g[idx(gidx, gidy)];
    double v2 = v * v;

    double u = u_g[idx(gidx, gidy)];
    double u2 = u * u;

    double pow_v[3] = {1, v, v2};
    double pow_u[3] = {1, u, u2};

    double result = 0;
    double eps = .001;

    int term = 0;
    for (term = 0; term < 3; term++) {
        int i = 0;
        double nom = 0;
        double den = 0;
        for (i = 0; i < 3; i++) {
            int j = 0;
            for (j = 0; j < 3; j++) {
                double rho = rho_g[9 * term + 3 * i + j];
                double kap = kap_g[9 * term + 3 * i + j];

                nom += rho * pow_v[i] * pow_u[j];
                den += kap * pow_v[i] * pow_u[j];
            }
        }

        if (den < -eps || den > eps)  {
            double current = nom / den;
            result += current;
        }
    }

    v_update_g[idx(gidx, gidy)] += result;
}