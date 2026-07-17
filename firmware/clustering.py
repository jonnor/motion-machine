

from pca import PCA
from scaler import StandardScaler
from reservoir_sampler import ReservoirSampler

import npyfile


def load_samples(sampler, path, hop_samples=1000):

    n_features = sampler.cols

    with npyfile.Reader(path) as reader:
        shape = reader.shape
        assert len(shape) == 2 and shape[1] == n_features, \
            f"Expected (N, {n_features}) array, got {shape}"
        assert reader.typecode == 'h', \
            f"Expected int16 ('h'), got '{reader.typecode}'"

        n_samples_total = shape[0]

        assert n_samples_total >= sampler.k, f'Not enough samples {n_samples_total}'

        hop_items = hop_samples * n_features

        chunks = reader.read_data_chunks(hop_items)

        for chunk in chunks:
            sampler.push(chunk)


def main():

    n_samples = 1000
    n_features = 7

    sampler = ReservoirSampler(k=n_samples, cols=n_features)

    in_path = 'data/pamap2_features.npy'

    load_samples(sampler, in_path)

    data  = sampler.get_flat()

    print('D', len(data))

    scaler = StandardScaler()

    scaled = scaler.fit_transform(data, n_samples=n_samples, out=None)

    pca = PCA(n_components=3)
    transformed = pca.fit_transform(scaled, n_samples=n_samples)

    print('tr', len(transformed))


if __name__ == '__main__':
    main()
