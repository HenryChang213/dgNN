import matplotlib.pyplot as plt


plt.style.use("seaborn-v0_8-deep")

fig, ax = plt.subplots(ncols=3, figsize=(10, 3), layout='constrained')

ax[0].plot([1,2,4], [1.0452361514073447,1.1751391443902097,1.6633961376176407], 'o-', label='equal')
ax[0].plot([1,2,4], [1.045271073647946,1.5203397671876089,2.2230969968337897], 'o-', label='binary search')
ax[0].set_title("reddit")
ax[0].set_ylim(0,2.5)
ax[0].legend(loc="lower left")

ax[1].plot([1,2,4], [0.598344028294727,0.5015964157893461,0.4034852272944557], 'o-', label='equal')
ax[1].plot([1,2,4], [0.5997697003775234,0.5133146702073711,0.4034148813341305], 'o-', label='binary search')
ax[1].set_title("arxiv")
ax[1].set_ylim(0,1)
ax[1].legend(loc="lower left")


ax[2].plot([1,2,4], [0.8163594802049566,0.9691842975695945,1.6107960496703988], 'o-', label='equal')
ax[2].plot([1,2,4], [0.8167663527996795,1.0107433536968131,1.8934696840504697], 'o-', label='binary search')
ax[2].set_title("products")
ax[2].set_ylim(0,2.5)
ax[2].legend(loc="lower left")


fig.supxlabel("Number of GPUs")
fig.supylabel("Throughput (Tflops/sec)")
plt.savefig("./figure3.pdf", dpi=300)

