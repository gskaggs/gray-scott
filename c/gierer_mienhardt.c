#define idx(x, y) x + block_size *y

__kernel void iterate(
    __global const double *v_g, __global const double *u_g,
    const double rho, const double kap,
    const double mu, const double nu,
    __global double *v_update_g, __global double *u_update_g)
{
    uint gidx = get_global_id(0);
    uint gidy = get_global_id(1);
    uint block_size = get_global_size(0);

    double v = v_g[idx(gidx, gidy)];
    double u = u_g[idx(gidx, gidy)];

    double v2 = v * v;
    double uv2 = u * (1 + kap * v2);
    double v2_uv2 = v2 / uv2;

    double result_v = rho * (v2_uv2 - mu * v);
    double result_u = rho * (v2 - nu * u);

    v_update_g[idx(gidx, gidy)] += result_v;
    u_update_g[idx(gidx, gidy)] += result_u;
}