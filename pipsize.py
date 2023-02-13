# Run `python pipsize.py` in Terminal to show size of pip packages
# Credits: https://stackoverflow.com/a/67914559/11067496

sort_in_descending = True   # Show packages in descending order

import os
import pkg_resources

def calc_container(path):
    total_size = 0
    for dirpath, _, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total_size += os.path.getsize(fp)
    return total_size

dists = [d for d in pkg_resources.working_set]
dists_with_size = {}

for dist in dists:
    try:
        path = os.path.join(dist.location, dist.project_name)
        size = calc_container(path)
        dists_with_size[size] = dist
    except OSError:
        '{} no longer exists'.format(dist.project_name)

# Sort packages size
dists_with_size = dict(sorted(dists_with_size.items(), reverse=sort_in_descending))

for size, dist in dists_with_size.items():
    if size/1000 > 1.0:
        print (f"{dist}: {size/1000000:.2f} MB")
        print("-"*40)