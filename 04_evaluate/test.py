from track_graph import ids_to_track_graph
import numpy as np

ids = np.array([
    [[1, 1, 0, 2, 2, 2, 0, 0]],
    [[1, 1, 0, 2, 2, 2, 0, 0]],
    [[1, 1, 0, 2, 2, 0, 0, 0]],
    [[1, 1, 0, 2, 2, 0, 0, 0]],
    [[1, 1, 2, 2, 0, 2, 2, 0]],
    [[1, 1, 2, 2, 0, 2, 2, 0]],
    [[1, 1, 2, 2, 0, 2, 2, 0]],
    [[1, 1, 2, 2, 0, 2, 2, 0]],
    [[1, 1, 2, 2, 0, 2, 2, 0]],
    [[1, 1, 2, 2, 0, 2, 2, 0]],
    [[1, 1, 2, 2, 0, 0, 2, 2]],
    [[1, 1, 2, 2, 0, 0, 2, 2]],
])

tracks, track_ids = ids_to_track_graph(ids)

print
print
print
for t in tracks:
    print(t)

print
print
print(track_ids)
