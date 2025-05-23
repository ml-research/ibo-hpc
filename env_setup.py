import os

def setup_environment():
    # set symlink for NASLib data
    data_abs_dir = os.path.abspath('./benchmark_data/')
    if not os.path.exists('/usr/local/lib/python3.9/site-packages/naslib/data/'):
        os.mkdir('/usr/local/lib/python3.9/site-packages/naslib/data/')
        os.system('wget -O /usr/local/lib/python3.9/site-packages/naslib/data/nasbench_only108.pkl https://figshare.com/ndownloader/files/53291267')
        os.system('wget -O /usr/local/lib/python3.9/site-packages/naslib/data/nb201_ImageNet16_full_training.pickle https://figshare.com/ndownloader/files/53291258')
        os.system('wget -O /usr/local/lib/python3.9/site-packages/naslib/data/nb201_cifar10_full_training.pickle https://figshare.com/ndownloader/files/53291264')
        os.system('wget -O /usr/local/lib/python3.9/site-packages/naslib/data/nb201_cifar100_full_training.pickle https://figshare.com/ndownloader/files/53291261')

    if not os.path.exists(data_abs_dir):
        os.mkdir(data_abs_dir)

    if not os.path.exists(os.path.join(data_abs_dir, 'assembled_surrogates/')):
        os.system(f'wget -O {data_abs_dir}/assembled_surrogates.zip https://figshare.com/ndownloader/files/53290088')
        os.system('unzip /app/benchmark_data/assembled_surrogates.zip -d /app/benchmark_data')
        os.system('rm /app/benchmark_data/assembled_surrogates.zip')

    # download hpo-b data
    if not os.path.exists(os.path.join(data_abs_dir, 'hpob-data/')):
        os.system('wget -O /app/benchmark_data/hpob-data.zip https://rewind.tf.uni-freiburg.de/index.php/s/xdrJQPCTNi2zbfL/download/hpob-data.zip')
        os.system('unzip /app/benchmark_data/hpob-data.zip -d /app/benchmark_data')
        os.system('wget -O /app/benchmark_data/hpob-data/meta-dataset-descriptors.json https://github.com/machinelearningnuremberg/HPO-B/raw/main/hpob-data/meta-dataset-descriptors.json')
        os.system('wget -O /app/benchmark_data/hpob-data/meta-test-tasks-per-space.json https://github.com/machinelearningnuremberg/HPO-B/raw/main/hpob-data/meta-test-tasks-per-space.json')
        os.system('rm /app/benchmark_data/hpob-data.zip')

    if not os.path.exists(os.path.join(data_abs_dir, 'hpob-surrogates/')):
        os.mkdir(f'{data_abs_dir}/hpob-surrogates')
        os.system('wget -O /app/benchmark_data/saved-surrogates.zip https://rewind.tf.uni-freiburg.de/index.php/s/rTwPgaxS2Z7NH39/download/saved-surrogates.zip')
        os.system('unzip /app/benchmark_data/saved-surrogates.zip -d /app/benchmark_data/hpob-surrogates')
        os.system('rm /app/benchmark_data/saved-surrogates.zip')

    if not os.path.exists(os.path.join(data_abs_dir, 'lc_bench/')):
        os.mkdir(os.path.join(data_abs_dir, 'lc_bench/'))
        os.system('wget -O /app/benchmark_data/lc_bench/lc_bench.tar.gz https://figshare.com/ndownloader/files/53292767')
        os.system('tar --no-same-owner -xvzf /app/benchmark_data/lc_bench/lc_bench.tar.gz -C /app/benchmark_data/')
        os.system('rm /app/benchmark_data/lc_bench/lc_bench.tar.gz')

    if not os.path.exists(os.path.join(data_abs_dir, 'pd1/')):
        os.mkdir(os.path.join(data_abs_dir, 'pd1/'))
        os.system('wget -O /app/benchmark_data/pd1_with_surrogates.tar.gz https://figshare.com/ndownloader/files/53290085')
        os.system('tar --no-same-owner -xvzf /app/benchmark_data/pd1_with_surrogates.tar.gz -C /app/benchmark_data/')
        os.system('rm /app/benchmark_data/pd1_with_surrogates.tar.gz')

    if not os.path.exists(os.path.join(data_abs_dir, 'fcnet_tabular_benchmarks/')):
        os.system('wget -O /app/benchmark_data/fcnet_tabular_benchmarks.tar.gz http://ml4aad.org/wp-content/uploads/2019/01/fcnet_tabular_benchmarks.tar.gz')
        os.system('tar --no-same-owner -xf /app/benchmark_data/fcnet_tabular_benchmarks.tar.gz -C /app/benchmark_data/')
        os.system('rm /app/benchmark_data/fcnet_tabular_benchmarks.tar.gz')

def create_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

setup_environment()