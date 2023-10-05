import os

if not os.path.exists("./log"):
    os.makedirs("./log/")


os.system("nohup python3 -u pore.py --dataset 100k --s 200 --N_prime 1 --T 100000 --N 10 --Alg ir --alpha 0.001 > ./log/ir_100k_results.log &")
