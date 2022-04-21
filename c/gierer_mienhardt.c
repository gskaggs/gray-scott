#define idx(x, y) x + block_size *y

__kernel void iterate(
    __global const float *v_g, __global const float *u_g,
    const float rho, const float kap,
    const float mu, const float nu,
    __global float *v_update_g, __global float *u_update_g)
{
    uint gidx = get_global_id(0);
    uint gidy = get_global_id(1);
    uint block_size = get_global_size(0);

    float v = v_g[idx(gidx, gidy)];
    float u = u_g[idx(gidx, gidy)];

    float v2 = v * v;
    float uv2 = u * (1 + kap * v2);
    float v2_uv2 = v2 / uv2;

    float result_v = rho * (v2_uv2 - mu * v);
    float result_u = rho * (v2 - nu * u);

    v_update_g[idx(gidx, gidy)] += result_v;
    u_update_g[idx(gidx, gidy)] += result_u;

}