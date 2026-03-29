#!/usr/bin/env python3
"""kmeans - K-Means clustering with K-Means++ initialization."""
import sys, math, random

def distance(a, b):
    return math.sqrt(sum((x-y)**2 for x,y in zip(a, b)))

def kmeans_pp_init(X, k, rng):
    centers = [X[rng.randint(0, len(X)-1)]]
    for _ in range(1, k):
        dists = [min(distance(x, c)**2 for c in centers) for x in X]
        total = sum(dists)
        r = rng.random() * total
        cumsum = 0
        for i, d in enumerate(dists):
            cumsum += d
            if cumsum >= r:
                centers.append(X[i])
                break
    return centers

def kmeans(X, k, max_iter=100, seed=42):
    rng = random.Random(seed)
    centers = kmeans_pp_init(X, k, rng)
    for _ in range(max_iter):
        clusters = [[] for _ in range(k)]
        labels = []
        for x in X:
            dists = [distance(x, c) for c in centers]
            label = dists.index(min(dists))
            clusters[label].append(x)
            labels.append(label)
        new_centers = []
        for j in range(k):
            if clusters[j]:
                ndim = len(X[0])
                new_centers.append([sum(p[d] for p in clusters[j]) / len(clusters[j]) for d in range(ndim)])
            else:
                new_centers.append(centers[j])
        if new_centers == centers:
            break
        centers = new_centers
    return labels, centers

def inertia(X, labels, centers):
    return sum(distance(X[i], centers[labels[i]])**2 for i in range(len(X)))

def test():
    X = [[0,0],[1,0],[0,1],[10,10],[11,10],[10,11]]
    labels, centers = kmeans(X, 2)
    cluster0 = set(i for i, l in enumerate(labels) if l == labels[0])
    cluster1 = set(i for i, l in enumerate(labels) if l != labels[0])
    assert cluster0 == {0,1,2} or cluster0 == {3,4,5}
    assert len(centers) == 2
    i = inertia(X, labels, centers)
    assert i < 10
    labels3, centers3 = kmeans(X, 3)
    assert len(set(labels3)) <= 3
    single, c = kmeans([[5,5]], 1)
    assert single == [0]
    print("All tests passed!")

if __name__ == "__main__":
    test() if "--test" in sys.argv else print("kmeans: K-Means clustering. Use --test")
