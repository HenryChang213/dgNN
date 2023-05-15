import matplotlib.pyplot as plt
import numpy as np

plot_cates = [
    "reddit",
    "ogbn-arxiv",
    "ogbn-products",
    "ogbn-papers100M"
]

groups = [
    "Memory-0",
    "Memory-1",
    "Memory-2",
    "MG-GCN"
]

series = [{
        'equal': [0,0,0,0],
        'binary_search': [0,0,0,0],
        'unpermuted': [0,0,0,38309489486/1e12/0.0227693],
        'permuted': [0,0,0,38309489486/1e12/0.0146227],
    },
    {
        'equal': [0,0,0,0],
        'binary_search': [0,0,0,0],
        'unpermuted': [0,0,0,3557177320/1e12/0.00427404],
        'permuted': [0,0,0,3557177320/1e12/0.00376192],
    },
    {
        'equal': [0,0,0,0],
        'binary_search': [0,0,0,0],
        'unpermuted': [0,0,0,73842153998/1e12/0.0463017],
        'permuted': [0,0,0,73842153998/1e12/0.033549],
    },
    {
        'equal': [0,0,0,0],
        'binary_search': [0,0,0,0],
        'unpermuted': [0,0,0,0],
        'permuted': [0,0,0,0],
    },

]


with open("figure1_new_new.csv") as f:
    for line in f:
        linsp = line.split(",")
        # print(linsp)
        series[plot_cates.index(linsp[3])][linsp[5]][int(linsp[4])] = float(linsp[6])

# print(series)

x = np.arange(len(groups))  # the label locations
width = 1 / 5  # the width of the bars
plt.style.use("seaborn-v0_8-deep")

fig, ax = plt.subplots(figsize=(15, 3), layout='constrained', nrows=1, ncols=4)
# ax.set_color_cycle(['red', 'black', 'yellow'])

for sp in range(len(plot_cates)):
    multiplier = 0
    for predictor, result in series[sp].items():
        if multiplier == 2: multiplier = 0
        offset = width * multiplier
        rects = ax[sp].bar(x + offset, result, width, label=predictor)
        # ax.bar_label(rects, label_type='edge', color='snow', padding=-15)
        multiplier += 1
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax[sp].set_title(plot_cates[sp])
    ax[sp].legend(loc='upper left', ncols=1)
    ax[sp].set_xticks(x + width * 0.5, groups)
    ax[sp].set_ylim(0, 3)

fig.supxlabel("Memory Management Method")
fig.supylabel("Throughput (TFlops/sec)")
plt.savefig('./figure1.pdf', dpi=300)