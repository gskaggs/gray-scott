#define idx(x, y) x + block_size * y

__kernel void iterate(
    __global double *v_g,
    __global double *v_update_g, 
    const double dt)
{
    uint gidx = get_global_id(0);
    uint gidy = get_global_id(1);
    uint block_size = get_global_size(0);

    if (gidx == 0 || gidy == 0 || gidx == block_size-1 || gidy == block_size-1) 
        return;

    double v = v_g[idx(gidx, gidy)];
    double v_update = v_update_g[idx(gidx, gidy)];
    double result = v + dt * v_update; 
    
    if (result < 0) result = 0;
    if (result > 5) result = 5;

    v_g[idx(gidx, gidy)] = result;
}