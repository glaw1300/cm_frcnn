import pickle
import tqdm
import matplotlib.pyplot as plt
import glob
import pandas as pd
import numpy as np

def pick(x):
    return x.iloc[0]

def to_arr(d):
    return [(int(k.split("_")[0]), int(k.split("_")[-1]), *v) for k, v in d.items()]

comps = glob.glob("compares_low/*.pickle")

df = pd.DataFrame()
print("Loading in data...")
for c in comps:
    #if int(c.split("/")[1].split("_")[-1].strip(".pickle")) > 70:
    #    continue
    with open(c, "rb") as f:
        tmp = pd.DataFrame(to_arr(pickle.load(f)), columns=["sid", "iter", "x", "y", "pid", "tif", "pred", "gt", "dist", "score"])
        df = pd.concat((df, tmp))

df = df[~df.pid.isna()]
df = df.astype({"pid":"int64"})

print("Data loaded!")
min_overflights = 1

sl = pd.read_csv("../GAO_Summer2020_source_list_multimodal.csv")
sl = sl[(sl["source_lat"] != -9999) & (sl["source_lon"] != -9999) & (~sl["source_id"].isna())]
sl = sl[sl["N"] >= min_overflights]
# match up to 2020 survey
sl = sl[(sl["source_lat"] < 36) & (sl["source_lon"] > -106)]
pl = pd.read_csv("../GAO_Summer2020_plume_source_list_multimodal.csv")
pl = pl[(pl["lat"] != -9999) & (pl["lon"] != -9999) & (~pl["source_id"].isna())]
df["q"] = 0.
df["qs"] = 0.

g = pl.groupby(["source_id", "date"]).agg({"EmissionRate_kgh_HRRR_10m":"mean", "EmissionUncertainty_kgh_HRRR_10m":"mean", "plume_id":pick})
for i, row in g.iterrows():
    df.loc[df["pid"] == row["plume_id"], "q"] = row["EmissionRate_kgh_HRRR_10m"]
    df.loc[df["pid"] == row["plume_id"], "qs"] = row["EmissionUncertainty_kgh_HRRR_10m"]

df = df[df["q"]!=0.]

newdf = []
st = "2020"
for i,row in tqdm.tqdm(sl.iterrows()):
    sid = int(row["source_id"])

    # if no images
    if len(glob.glob(f"../DIMAC_Context_Images/{sid}_*.tif")) == 0:
        #newdf.append((sid, "no images", st, row["fmean"], row["unc"], row["unc"]**2, 0., 0.))
        continue

    # get all entries from df
    spers = row["f"]
    tot = len(df.loc[df["sid"] == sid])

    # if no predictions
    if tot == 0:
        newdf.append((sid, "no predictions", st, row["fmean"], row["unc"], row["unc"]**2, 0., 0.))

    for name, group in df.loc[df["sid"] == sid].groupby(["pred"]):
        ppers = spers * len(group) / tot
        grp = group.groupby(["iter"]).mean()
        ppers = len(group) / tot * spers
        std = grp.q.std()
        mn = grp.q.mean() * ppers
        qs = (grp.qs.mean() * ppers) # standard deviation
        newdf.append((sid, name, st, mn, qs, qs**2, std, grp.dist.mean()))


ndf = pd.DataFrame(newdf, columns=["sid", "pred", "gt", "q", "qs", "qv", "std", "d"])
#ndf["std"] = 0.
#for name, group in ndf.groupby(["sid"]):
#    ndf.loc[ndf["sid"] == name, "std"] = group["q"].std()

# parse by distance
dthresh = 215. # 150 m @ .7 m resolution
dlabel = f">{int(dthresh*.7)} m"
ndf.loc[ndf["d"] > dthresh, "pred"] = dlabel

fig, ax = plt.subplots()
cs = ["tab:orange", "tab:green", "tab:blue", "tab:red", "tab:purple", "tab:pink", "tab:olive", "tab:cyan", "lightgray", "lightgray", "lightgray"]
labels = ["tank", "compressor", "processing", "wellhead", "flare", "pond", "slugcatcher", "pumpjack", "no predictions", "pipeline", "no images"]
hatchmap = {"pipeline": "///", "no predictions": "+++", "no images":"...", dlabel:"xxx"}
xticksl = []
xticks = []
plt.style.use("default")
ds = set()
tot_ems = ndf.groupby(["gt","pred"]).sum()
facs = ndf["gt"].unique()
curx = 0
conv = 8.76e-06
les = []
outlines = ["slugcatcher", "flare", "wellhead"]
print(tot_ems)
for i, f in enumerate(facs):
    ax.annotate(f, (curx, max(tot_ems.q)*1.4*conv))
    for j, q in enumerate(sorted(tot_ems.q[f], reverse=True)):
        l = tot_ems[tot_ems["q"] == q].q[f].keys()[0]
        if l in hatchmap:
            continue
        curx += 2
        xticksl.append(l)
        xticks.append(curx-2)
        a = ax.errorbar(curx - 1, tot_ems["q"][f][l]*conv, yerr=conv*(tot_ems["qs"][f][l]), label="emission uncertainty" if "qs" not in ds else "", color="darkgrey", capsize=2, fmt=".",markersize=1) # sum of sigma_q
        #ax.errorbar(curx - 1, tot_ems["q"][f][l]*conv, yerr=conv*np.sqrt(tot_ems["qv"][f][l]), label="emission uncertainty" if "qs" not in ds else "", color="darkgrey", capsize=2, fmt=".",markersize=1) # variance sqrt

        if "std" not in ds:
            les.append(a)

        a = ax.errorbar(curx + 1, tot_ems["q"][f][l]*conv, yerr=conv*tot_ems["std"][f][l], label="attribution error" if "std" not in ds else "", color="grey", capsize=2, fmt=".", markersize=1)

        if "qs" not in ds:
            les.append(a)

        if l in outlines:
            a = ax.bar(curx, tot_ems["q"][f][l]*conv, width=4, label=l if l not in ds else "", ec=cs[labels.index(l)], fc="none", hatch="///")
        else:
            a = ax.bar(curx, tot_ems["q"][f][l]*conv, width=4, label=l if l not in ds else "", color=cs[labels.index(l)])

        if l not in ds:
            les.append(a)

        ds.add(l)
        ds.add("std")
        ds.add("qs")
        curx += 2

    # add extras at end
    for h in hatchmap.keys():
        curx += 2
        if h not in tot_ems.q[f]:
            continue

        a = ax.bar(curx, tot_ems["q"][f][h]*conv, width=4, label=h if h not in ds else "", hatch=hatchmap[h], ec="w", color="gray")
        if h not in ds:
            les.append(a)

        # no algorithm error cuz no predictions for extras
        ax.errorbar(curx, tot_ems["q"][f][h]*conv, yerr=conv*tot_ems["qs"][f][h], label="emission uncertainty" if "qs" not in ds else "", color="darkgrey", capsize=2, fmt=".",markersize=1)

        ds.add(h)

        curx += 2

    ax.axvline(curx + 2, color="k", lw=1)
    curx += 4
#ax.set_xticks(xticks)
ax.set_xticks([])
#ax.set_xticklabels(xticksl, rotation=70)
#ax.set_xticks()
ax.legend(handles=les, bbox_to_anchor=(1.05, 1.0), loc='upper left')
ax.set_xlim(-2, curx + 1)
ax.set_ylim(0, max(tot_ems.q)*1.5*conv)
ax.set_title(f"Emissions by sub-facility, N >= {min_overflights}")
ax.set_ylabel("Total emissions (Tg/a)")
fig.tight_layout()
plt.show()
