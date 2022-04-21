#define idx(x, y) x + block_size * y

__kernel void iterate(
    __global float *v_g,
    __global const float *v_update_g, 
    const float dt)
{
    uint gidx = get_global_id(0);
    uint gidy = get_global_id(1);
    uint block_size = get_global_size(0);

    if (gidx == 0 || gidy == 0 || gidx == block_size-1 || gidy == block_size-1)
        return;

    float v = v_g[idx(gidx, gidy)];
    float v_update = v_update_g[idx(gidx, gidy)];
    v_g[idx(gidx, gidy)] = v + dt * v_update; //v + dt * v_update;
}