#!/usr/bin/env python3
"""Null test for the shape analysis: replace every real price series with a
random walk (same length, same daily volatility, zero drift), then run the same
k-means clustering and the same named-pattern detectors. If the shapes and the
detection rates match the real data, the "patterns" are a property of random
walks, not of markets."""

import json

import numpy as np
import pandas as pd

from common import RESULTS, ROOT, UNIVERSE_NAME, base_config, load_frames, setup_logging
import shapes

log = setup_logging("shapes_null")


def fake_frames(frames: dict, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    out = {}
    for t, df in frames.items():
        r = np.diff(np.log(df["Close"].to_numpy()))
        sim = rng.normal(0.0, r.std(), size=len(df) - 1)
        px = df["Close"].iloc[0] * np.exp(np.concatenate([[0.0], np.cumsum(sim)]))
        out[t] = pd.DataFrame({"Close": px}, index=df.index)
    return out


def main() -> None:
    real_centers = np.load(RESULTS / "shape_centers.npy")        # saved by shapes.py, ordered by share
    real = json.load(open(RESULTS / "shapes.json"))
    frames = fake_frames(load_frames(base_config()))
    X, _, meta = shapes.build_windows(frames)
    shapes.SUFFIX = shapes.SUFFIX + "_randomwalk"
    cl = shapes.cluster_shapes(X, meta)
    sim_centers = np.load(RESULTS / "shape_centers.npy")
    # match every real centroid to its nearest simulated centroid
    matches = []
    for i, c in enumerate(real_centers):
        d = np.linalg.norm(sim_centers - c, axis=1) / np.sqrt(shapes.WIN)
        j = int(d.argmin())
        matches.append({"real_shape_rank": i, "real_share": real["clusters"]["table"][i]["share_pct"],
                        "sim_shape_rank": j, "sim_share": cl["table"][j]["share_pct"],
                        "distance": float(d[j]), "corr": float(np.corrcoef(c, sim_centers[j])[0, 1])})
    m = pd.DataFrame(matches)
    log.info("real vs random-walk centroids:\n%s", m.round(3).to_string(index=False))
    _, pats = shapes.detect_patterns(frames)
    comp = pd.DataFrame(real["patterns"]["summary"])[["pattern", "per_1000_days"]].rename(columns={"per_1000_days": "real"})
    comp["random_walk"] = pd.DataFrame(pats["summary"]).set_index("pattern").loc[comp.pattern, "per_1000_days"].values
    log.info("named-pattern frequency per 1000 stock-days:\n%s", comp.round(1).to_string(index=False))
    with open(RESULTS / "shapes_null.json", "w") as f:
        json.dump({"centroid_matches": matches, "pattern_rates": comp.to_dict("records"), "sim_clusters": cl}, f, indent=2)
    np.save(RESULTS / "shape_centers.npy", real_centers)   # restore


if __name__ == "__main__":
    main()
