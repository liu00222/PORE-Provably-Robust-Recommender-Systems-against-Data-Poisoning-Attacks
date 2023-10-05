import os

if not os.path.exists("./log"):
    os.makedirs("./log/")

def run(dataset, s, N_prime, T, Alg, series_num=1):
    log_dir = "./log"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    assert (Alg in ['ir', 'bpr'])

    os.system(f"nohup python3 -u sample_T.py  \
    --dataset {dataset} \
    --s {s} \
    --N_prime {N_prime} \
    --T {T} \
    --Alg {Alg} \
    --series {series_num} \
    > {log_dir}/sample_{Alg}_{dataset}_s{s}_N_prime{N_prime}_T{T}_{series_num}.log &")


run('100k', 200, 1, 100000, 'ir')
