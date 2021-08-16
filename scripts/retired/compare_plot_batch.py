import pandas as pd
import matplotlib.pyplot as plt
import pickle
import glob
import tqdm

comps = glob.glob("compares/*.pickle")

def to_arr(d):
    return [(int(k.split("_")[0]), int(k.split("_")[-1]), *v) for k, v in d.items()]

df = pd.DataFrame()
for c in comps:
    with open(c, "rb") as f:
        tmp = pd.DataFrame(to_arr(pickle.load(f)), columns=["sid", "iter", "x", "y", "pid", "tif", "pred", "gt"])
        df = pd.concat((df, tmp))

pl = pd.read_excel("../permian_plume_list_EST_03252021.xlsx")
sl = pd.read_excel("../permian_source_list_EST_03252021.xlsx")
combs = []

def pick(x):
    return x.iloc[0]

df.loc[:, "qplume"] = 0.
df.loc[:, "qplume_sigma"] = 0.
g = pl.groupby(["source_id", "date_of_detection"]).agg({"qplume":"mean", "sigma_qplume":"mean", "plume_id":pick})
for name, group in tqdm.tqdm(pl.groupby(["source_id", "date_of_detection"])):
    for i, pid in enumerate(group["plume_id"]):
        if i != 0:
            df = df.loc[df["pid"] != pid]
        else:
            df.loc[df["pid"] == pid, "qplume"] = group.loc[:, "qplume"].mean()
            df.loc[df["pid"] == pid, "qplume_sigma"] = group.loc[:, "sigma_qplume"].mean()

# average plume fluxes for total emissions per source
tots = []
i = 0
for name, group in tqdm.tqdm(df.groupby(["iter", "sid", "pred"])):
    i += 1
    sfct = len(group)
    sid = "P%05d" %name[1]

    srow = sl.loc[sl["source_id"] == sid]
    sct = len(df.loc[(df["sid"] == name[1]) & (df["iter"] == name[0])])
    pers = float(srow["source_persistence"])
    ppers = sfct / sct * pers
    #if sfct != sct:
    #    print(sid, name, f"{sfct}/{sct}", pers, ppers)
    if i % 1000 == 0:
        print((name[0], name[1], name[2], group["gt"].iloc[0], group["qplume"].mean() * ppers, ppers))
    tots.append((name[0], name[1], name[2], group["gt"].iloc[0], group["qplume"].mean() * ppers))

tot = pd.DataFrame(tots, columns=["iter", "sid", "pred", "gt", "pplume"])

comb = tot.groupby(["gt", "pred"])["pplume"].sum()
combs.append(comb)

fig, ax = plt.subplots()
facs = ["production", "compressor", "processing"]

mn = tot.groupby(["sid","pred"], as_index=False).agg({"pplume": "mean", "gt":pick})
mn.groupby(["gt", "pred"])["pplume"].sum()["production"].sum()

for i, label in enumerate(facs):
    pass
