
from microhive import MicroHive

database_dir = '/tsdb'

columns = [
    'orient_x',
    'orient_y',
    'orient_z',
    'sma',
    'mean_x',
    'mean_y',
    'mean_z',
]

db = MicroHive(database_dir, {
    'metrics': {
        'hop': 1_000_000,
        'columns': columns,
        'dtype': 'int16',
        'granularity': 'hour',
    },
})

info = db.get_info('metrics')

print(info)

# add to check_metrics.py temporarily
import microhive as mh
print("EPOCH_OFFSET:", mh._EPOCH_OFFSET)
info = db.get_info('metrics')
print(info)
# info['start_s'] should be ~1775052000 (Unix 2026), not ~2721736800


base_epoch = 1775052566 # 01.04.2026
start = base_epoch - (2*3600)
end = base_epoch + (2*3600)

for chunk in db.get_timerange('metrics', start, end):
    print(len(chunk))

