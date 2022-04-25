#define idx(x, y) x + block_size * y

__kernel void iterate(
    __global const double *v_g,
    __global double *v_update_g,
    const double dv)
{
    uint gidx = get_global_id(0);
    uint gidy = get_global_id(1);
    uint block_size = get_global_size(0);
    double v = v_g[idx(gidx, gidy)];

    double result = 0;
    result -= 4 * v;

    if (gidx < block_size-1)
        result += v_g[idx((gidx+1), gidy)];
    if (gidx > 0)
        result += v_g[idx((gidx-1), gidy)];
    if (gidy < block_size-1)
        result += v_g[idx(gidx, (gidy+1))];
    if (gidy > 0)
        result += v_g[idx(gidx, (gidy-1))];

    v_update_g[idx(gidx, gidy)] = dv * result;
}