import json
import matplotlib.pyplot as plt

psnr_entropy = []
psnr_no_entropy = []

with open("results.json") as f:
    for line in f:
        r = json.loads(line)
        if r["entropy"]:
            psnr_entropy.append(r["psnr"])
        else:
            psnr_no_entropy.append(r["psnr"])

plt.plot(psnr_entropy, label="With Entropy")
plt.plot(psnr_no_entropy, label="Without Entropy")
plt.xlabel("Test Image Index")
plt.ylabel("PSNR (dB)")
plt.legend()
plt.grid()
plt.show()
