from PIL import Image as im
from PIL import ImageFont, ImageDraw
import os
import pickle

def grid_w_h(chromosomes):
    N = len(chromosomes)
    for W in range(N):
        if W**2 >= N and N % W == 0:
            return W, N//W

    return N, 1

def create_img_grid(images, text):
    W, H = images[0][0].width, images[0][0].height
    rows, cols = len(images), len(images[0])
    grid = im.new("RGB", (rows*W, cols*H))
    draw = ImageDraw.Draw(grid)
    font = ImageFont.load_default()
    
    for i in range(rows):
        for j in range(cols):
            x, y = i*W, j*H
            grid.paste(images[i][j], (x, y))
            draw.rectangle((x,y,x+W,y+10),fill=(0))
            draw.text((x, y),text[i][j],(255, 255, 255),font=font)

    return grid

def present_chromosomes(chromosomes, cur_iter, args):
    W, H = grid_w_h(chromosomes)
    img_text = [['' for _ in range(H)] for _ in range(W)]
    images   = [[None for _ in range(H)] for _ in range(W)]
    successful_params = []

    for i in range(W):
        for j in range(H):
            cur = H*i+j
            c = chromosomes[cur]
            F, k, fitness = round(c.F, 4), round(c.k, 4), round(c.fitness, 2)
            img_text[i][j] = f'#{cur+1}:'
            if args.dirichlet_vis:
                img_text[i][j] = img_text[i][j] + f' Fit={fitness}'
            images[i][j]   = chromosomes[cur].image
            if c.pattern:
                successful_params.append((cur, c.get_params()))

    sim_type = 'Paramater search' if args.param_search else 'Genetic algorithm'
    last_gen = cur_iter == args.num_iters or args.param_search
    if last_gen:
        if not args.test_speed: 
            print(f"{sim_type} terminated with {len(successful_params)} turing patterns out of {len(chromosomes)} chromosomes")
        for idx, params in successful_params:
            print(f'Chromosome #{idx+1}:')
            print(params)

    grid = create_img_grid(images, img_text)
    sim_type = 'param_search' if args.param_search else args.fitness
    rd_string = '_'.join(sorted(args.rd))
    sim_dir =  f'./results/{rd_string}'
    if not os.path.exists(sim_dir):
        os.makedirs(sim_dir)

    sim_id = f'{sim_dir}/{sim_type}_{len(chromosomes)}_{args.end_time}_{cur_iter}'
    img_file, param_file = sim_id + '.png', sim_id + '.pkl'

    count = 1
    while os.path.exists(img_file) or os.path.exists(param_file):
        img_file, param_file = sim_id + f'v{count}' + '.png', sim_id + f'v{count}' + '.pkl'
        count += 1

    have_display = bool(os.environ.get('DISPLAY', None))
    if (last_gen or args.fitness == 'user') and have_display:
        try:
            grid.show()
        except:
            pass

    print('Saving simulation image at', img_file)
    grid.save(img_file)

    with open(param_file, 'wb') as file:
        pickle.dump((chromosomes, cur_iter, args), file)

    return grid