def process_function_param_search(param_seeds, images, successful_params):
    while True:
        i, j, F, k = param_seeds.get()
        if i == "DONE":
            break

        print(f"Beginning sim: F={F}, k={k}")

        sim = GrayScott(F=F, kappa=k, movie=False, outdir="./garbage", name=f"{F}_{k}")
        pattern, _, image = sim.integrate(0, 2000, dump_freq=100, report=250, should_dump=False)
        images.put((i, j, image))

        print(f"Done with sim F={F}, k={k}")

    images.put(('DONE', None, None))

def param_search(args):
    """
    Searchers the space of parameters for Turing patterns
    """
    F0, F1, k0, k1 = args.F0, args.F1, args.k0, args.k1
    Nf, Nk = args.Nf, args.Nk  # We'll have Nf * Nk chromosomes
    df, dk = (F1 - F0) / Nf, (k1 - k0) / Nk

    successul_params = []
    images = [[None for _ in range(Nk)] for _ in range(Nf)]
    image_queue = Queue()
    param_seeds = [(i, j, round(F0 + i * df, 3), round(k0 + j * dk, 3)) for i in range(Nf) for j in range(Nk)]
    for _ in range(args.num_threads):
        param_seeds.append(("DONE", None, None, None))

    q = Queue()
    for param in param_seeds:
        q.put(param)

    param_seeds = q

    start = time.time()
    processes = start_processes(args.num_threads, process_function, (param_seeds, image_queue, successul_params))
    num_successes = len(successul_params)

    img_text = [[f'F={round(F0 + i * df, 3)}, K={round(k0 + j * dk, 3)}' for j in range(Nk)] for i in range(Nf)]

    count = 0
    while count < args.num_threads:
        i, j, image = image_queue.get()
        if i == 'DONE':
            count+=1
            print(i)
            continue
        images[i][j] = image
        print(i,j)

    end_processes(processes)
    end = time.time()
    print(f'Processes time taken {start-end}')

    grid = create_img_grid(images, img_text)
    grid.save('param_search.png')
    print(f"Param search terminated with {num_successes} turing patterns")
    for params in successul_params:
        print(f"F={params[0]}, k={params[1]}")