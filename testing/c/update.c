#define idx(x, y) x + block_size * y

__kernel void iterate(
    __global double *v_g,
    __global double *v_update_g, 
    const double dt)
{
    uint gidx = get_global_id(0);
    uint gidy = get_global_id(1);
    uint block_size = get_global_size(0);

    double v = v_g[idx(gidx, gidy)];
    double v_update = v_update_g[idx(gidx, gidy)];

    if (gidx == 0 || gidy == 0 || gidx == block_size-1 || gidy == block_size-1) {
        // Wrap-around functionality
        uint gidx_prime = gidx;
        uint gidy_prime = gidy;

        if (gidx == 0) gidx_prime = block_size-2;
        if (gidy == 0) gidy_prime = block_size-2;
        if (gidx == block_size-1) gidx_prime = 1;
        if (gidy == block_size-1) gidy_prime = 1;

        v = v_g[idx(gidx_prime, gidy_prime)];
        v_update = v_update_g[idx(gidx_prime, gidy_prime)];
    }

    double result = v + dt * v_update; 
    
    if (result < 0) result = 0;
    if (result > 5) result = 5;

    v_g[idx(gidx, gidy)] = result;
}