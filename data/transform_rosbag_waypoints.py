import csv, tf, re
import pandas as pd

base4_wp = pd.read_csv('../../fourth/base_waypoints.csv')

base_waypoints4 = dict()
for idx, item in base4_wp.iteritems():
    if "position" in idx or "orientation" in idx:
        waypoint_idx = idx.split(".")[1]
        result = re.search("\d+", waypoint_idx)
        idx_key = int(result.group())

        if idx_key not in base_waypoints4:
            base_waypoints4[idx_key] = dict()
        idx_prefix = idx.split('.')[-2]  # position or orientation
        base_waypoints4[idx_key]["{}_{}".format(idx_prefix, idx[-1])] = item.values[0]

fieldnames = ['x', 'y', 'z', 'yaw']
base_waypoints_data = list()

for idx, wp in base_waypoints4.iteritems():
    _, _, yaw = tf.transformations.euler_from_quaternion([wp['orientation_x'], wp['orientation_y'],
                                                          wp['orientation_z'], wp['orientation_w']])

    base_waypoints_data.append(
        {'x': float(wp['position_x']), 'y': float(wp['position_y']), 'z': float(wp['position_z']),
         'yaw': float(yaw)})

with open('./base_waypoints_site.csv', 'w') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(base_waypoints_data)

print "done"