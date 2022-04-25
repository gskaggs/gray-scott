#define idx(x, y) x + block_size *y

__kernel void iterate(
    __global const double *v_g, __global const double *u_g,
    const double F, const double k,
    __global double *v_update_g, __global double *u_update_g)
{
    uint gidx = get_global_id(0);
    uint gidy = get_global_id(1);
    uint block_size = get_global_size(0);

    double v = v_g[idx(gidx, gidy)];
    double v2 = v * v;

    double u = u_g[idx(gidx, gidy)];
    double uv2 = u * v2;

    double result_v = uv2 - (F + k) * v;
    double result_u = - uv2 + F * (1 - u);
    v_update_g[idx(gidx, gidy)] += result_v;
    u_update_g[idx(gidx, gidy)] += result_u;

}