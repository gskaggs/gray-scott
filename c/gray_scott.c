#define idx(x, y) x + block_size *y

__kernel void iterate(
    __global const float *v_g, __global const float *u_g,
    const float F, const float k,
    __global float *v_update_g, __global float *u_update_g)
{
    uint gidx = get_global_id(0);
    uint gidy = get_global_id(1);
    uint block_size = get_global_size(0);

    float v = v_g[idx(gidx, gidy)];
    float v2 = v * v;

    float u = u_g[idx(gidx, gidy)];
    float uv2 = u * v2;

    v_update_g[idx(gidx, gidy)] = uv2 - (F + k) * v;
    u_update_g[idx(gidx, gidy)] = - uv2 + F * (1 - u);

}