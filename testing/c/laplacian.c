#define idx(x, y) x + block_size * y

__kernel void iterate(
    __global const float *v_g,
    __global float *v_update_g)
{
    uint gidx = get_global_id(0);
    uint gidy = get_global_id(1);
    uint block_size = get_global_size(0);
    float v = v_g[idx(gidx, gidy)];

    float result = 0;
    result -= 4 * v;

    if (gidx < block_size-1)
        result += v_g[idx((gidx+1), gidy)];
    if (gidx > 0)
        result += v_g[idx((gidx-1), gidy)];
    if (gidy < block_size-1)
        result += v_g[idx(gidx, (gidy+1))];
    if (gidy > 0)
        result += v_g[idx(gidx, (gidy-1))];

    v_update_g[idx(gidx, gidy)] = result;
}