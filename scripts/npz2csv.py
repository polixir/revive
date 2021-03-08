from __future__ import with_statement
import numpy as np

if __name__ == '__main__':
    data = np.load('data/ib-medium-99-train.npz')
    data = {k : v for k, v in data.items()}

    csv_data = np.concatenate([data['obs'], data['action']], axis=1)

    current_id = 0
    current_index = 0

    ids = []
    indexes = []

    for i in range(csv_data.shape[0]):
        if i == data['index'][current_id]:
            current_id += 1
            current_index = 0

        ids.append(current_id)
        indexes.append(current_index)

        current_index += 1

    ids = np.array(ids).reshape((-1, 1))
    indexes = np.array(indexes).reshape((-1, 1))
    uuids = np.arange(indexes.shape[0]).reshape((-1, 1))

    csv_data = np.concatenate([uuids, ids, indexes, csv_data], axis=1)

    np.savetxt('data/IndustrialBenchmark.csv', csv_data, fmt='%.3f', delimiter=',')

    head = ['uuid', 'traj_id', 'time_step'] + [f'obs_{i}' for i in range(data['obs'].shape[-1] - 2)] + ['fatigue', 'consumption'] + [f'action_{i}' for i in range(data['action'].shape[-1])]
    head = ','.join(head) + '\n'

    with open('data/IndustrialBenchmarkHead.csv', 'w') as f:
        f.write(head)