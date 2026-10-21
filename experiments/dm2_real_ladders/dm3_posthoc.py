"""DM-3 post hoc (not in the addendum): does the Kaplan law's data term alone reproduce the D = 20N "dark matter"
signature? Evaluate the fitted K1 surface (no second N term) along D = 20 N_total, then run the same N-only checks on
that model curve as on the data: 1-term law fitted on sizes >= 100M, residuals of the smaller sizes.

    python dm3_posthoc.py       # after dm3.py; writes results/dm3_posthoc_K1_slice.csv
"""
import os

import numpy as np
import pandas as pd

import dm3
import fitlib as F

RES = os.path.join(dm3.HERE, "results")


def main():
    A3, K3 = dm3.load()
    T = pd.read_csv(os.path.join(RES, "dm3_testK.csv"))
    rows = []
    for ds in dm3.CURVES:
        d, p = A3[ds], K3[ds]
        nt, ne, y = d.N_total.values, d.N_nonemb.values, d.y.values
        t = T[T.dataset == ds].iloc[0]
        for tag, c in (("K1 at its best c", t.c_K1), ("K1 at c = 1", 1.0)):
            if tag == "K1 at its best c":
                th = np.array([t.E_K1, t.A_K1, t.p_K1, t.G_K1, t.q_K1, 0.0, 1.0])
            else:
                f = dm3.fitK(dm3.n_of(1.0, p.N_total.values, p.N_nonemb.values), p.D.values / dm3.D_REF, p.L.values,
                             "K1", 40, np.random.default_rng(5))
                th = dm3.theta(f)
            n = dm3.n_of(c, nt, ne)
            m = dm3.kap(th, n, 20 * nt / dm3.D_REF, "K1")  # the model's D = 20N curve
            big = nt >= 1e8
            for src, yy in (("data", y), ("K1 model curve", m)):
                f1 = F.fit1(nt[big] / dm3.N_REF, yy[big])  # the DM-2 reverse ladder, N_total
                r = yy - F.predN(f1, nt / dm3.N_REF)
                g1 = F.fit1(nt / dm3.N_REF, yy)
                g2 = F.fit2(nt / dm3.N_REF, yy, g1)
                for j in range(len(nt)):
                    rows.append(dict(dataset=ds, model=tag, c=c, curve=src, N_total=nt[j], y=yy[j],
                                     data_minus_model=y[j] - m[j], rev_resid=r[j] if not big[j] else np.nan,
                                     dbic_1v2_total=F.bic(g1["rss"], len(nt), 3) - F.bic(g2["rss"], len(nt), 5)))
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(RES, "dm3_posthoc_K1_slice.csv"), index=False)
    s = out[out.N_total < 1e8].groupby(["dataset", "model", "curve"]).agg(
        rev_mean=("rev_resid", "mean"), rev_5M=("rev_resid", "first"), dbic=("dbic_1v2_total", "first"))
    s2 = out[out.curve == "data"].groupby(["dataset", "model"]).data_minus_model.agg(
        lambda v: float(np.sqrt(np.mean(v ** 2))))
    print(s.round(3).to_string())
    print("RMS data - K1 along D=20N:\n" + s2.round(4).to_string())


if __name__ == "__main__":
    main()
