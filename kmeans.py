#!/usr/bin/env python3
"""K-means clustering from scratch with visualization."""
import sys, math, random, csv

def distance(a, b): return math.sqrt(sum((x-y)**2 for x, y in zip(a, b)))

def kmeans(data, k, max_iter=100, seed=42):
    random.seed(seed); n = len(data); d = len(data[0])
    centroids = random.sample(data, k)
    labels = [0] * n
    for iteration in range(max_iter):
        # Assign
        changed = False
        for i, point in enumerate(data):
            dists = [distance(point, c) for c in centroids]
            new_label = dists.index(min(dists))
            if new_label != labels[i]: changed = True; labels[i] = new_label
        if not changed: break
        # Update centroids
        for j in range(k):
            members = [data[i] for i in range(n) if labels[i] == j]
            if members:
                centroids[j] = [sum(p[dim] for p in members) / len(members) for dim in range(d)]
    inertia = sum(distance(data[i], centroids[labels[i]])**2 for i in range(n))
    return labels, centroids, inertia, iteration + 1

def silhouette(data, labels, k):
    n = len(data); scores = []
    for i in range(n):
        cluster = labels[i]
        same = [j for j in range(n) if labels[j] == cluster and j != i]
        a = sum(distance(data[i], data[j]) for j in same) / max(len(same), 1)
        b = float("inf")
        for c in range(k):
            if c == cluster: continue
            others = [j for j in range(n) if labels[j] == c]
            if others:
                avg = sum(distance(data[i], data[j]) for j in others) / len(others)
                b = min(b, avg)
        scores.append((b - a) / max(a, b) if max(a, b) > 0 else 0)
    return sum(scores) / len(scores)

def ascii_plot(data, labels, k):
    if len(data[0]) < 2: return
    xs = [p[0] for p in data]; ys = [p[1] for p in data]
    W, H = 60, 20
    xmin, xmax = min(xs), max(xs); ymin, ymax = min(ys), max(ys)
    xr = xmax - xmin or 1; yr = ymax - ymin or 1
    grid = [[" "] * W for _ in range(H)]
    chars = "0123456789ABCDEF"
    for i, (x, y) in enumerate(zip(xs, ys)):
        px = int((x - xmin) / xr * (W - 1)); py = H - 1 - int((y - ymin) / yr * (H - 1))
        grid[py][px] = chars[labels[i] % len(chars)]
    for row in grid: print("".join(row))

def main():
    import argparse
    p = argparse.ArgumentParser(description="K-means clustering")
    p.add_argument("file", nargs="?"); p.add_argument("-k", type=int, default=3)
    p.add_argument("--plot", action="store_true"); p.add_argument("--demo", action="store_true")
    args = p.parse_args()
    if args.demo or not args.file:
        random.seed(42)
        data = [[random.gauss(cx, 0.5), random.gauss(cy, 0.5)] for cx, cy in [(0,0),(3,3),(6,0)] for _ in range(30)]
        labels, centroids, inertia, iters = kmeans(data, 3)
        print(f"K-means (k=3): converged in {iters} iterations, inertia={inertia:.2f}")
        sil = silhouette(data, labels, 3)
        print(f"Silhouette score: {sil:.3f}")
        for i, c in enumerate(centroids):
            count = labels.count(i)
            print(f"  Cluster {i}: center=({c[0]:.2f}, {c[1]:.2f}), size={count}")
        ascii_plot(data, labels, 3); return
    with open(args.file) as f:
        reader = csv.reader(f); next(reader)
        data = [[float(v) for v in row] for row in reader]
    labels, centroids, inertia, iters = kmeans(data, args.k)
    print(f"K={args.k}: {iters} iterations, inertia={inertia:.2f}")
    if args.plot: ascii_plot(data, labels, args.k)

if __name__ == "__main__": main()
