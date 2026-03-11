#!/usr/bin/env python3
"""K-means clustering."""
import sys, random, math
random.seed(42)
k=int(sys.argv[1]) if len(sys.argv)>1 else 3
# Generate sample data
data=[]
for cx,cy in [(2,2),(8,3),(5,8)]:
    data+=[(cx+random.gauss(0,1),cy+random.gauss(0,1)) for _ in range(20)]
centers=random.sample(data,k)
for iteration in range(20):
    clusters=[[] for _ in range(k)]
    for p in data:
        nearest=min(range(k),key=lambda i:math.hypot(p[0]-centers[i][0],p[1]-centers[i][1]))
        clusters[nearest].append(p)
    new_centers=[]
    for cl in clusters:
        if cl: new_centers.append((sum(p[0] for p in cl)/len(cl),sum(p[1] for p in cl)/len(cl)))
        else: new_centers.append(random.choice(data))
    if new_centers==centers: break
    centers=new_centers
print(f"K-means (k={k}), converged in {iteration+1} iterations:")
for i,(cx,cy) in enumerate(centers):
    print(f"  Cluster {i}: center=({cx:.2f},{cy:.2f}), size={len(clusters[i])}")
