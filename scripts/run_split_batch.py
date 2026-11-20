import os, subprocess, sys, time
cases = [
 "L1_d1_c0_k1_n200",
 "L1_d1_c0_k1_n200_s0.08",
 "L1_d2_c0.3_k4_n150",
 "L1_d4_c0_k1_n150",
 "L1_d1_c0_k8_n150",
 "L1_d1_c0_k1_n200_s0.08_a0.18_f0.05",
]
out = []
for c in cases:
    env = dict(os.environ, CASE=c)
    t0 = time.time()
    r = subprocess.run([sys.executable, "scripts/exp_split_merged.py"], capture_output=True, text=True, env=env, timeout=300)
    dt = round(time.time()-t0,1)
    # parse lines
    base = split = total = None
    for line in r.stdout.splitlines():
        if "matched truth points:" in line:
            if base is None:
                base = line.split(":")[1].strip()
            else:
                split = line.split(":")[1].strip()
    # total from truth? parse from first occurrence of / 
    print(f"{c}: baseline={base} split={split} ({dt}s)", flush=True)
    out.append((c,base,split,dt))
# write csv
with open("split_batch_results.csv","w") as f:
    f.write("case,baseline,split,time_s\n")
    for c,b,s,t in out:
        f.write(f"{c},{b},{s},{t}\n")
print("WROTE split_batch_results.csv")
