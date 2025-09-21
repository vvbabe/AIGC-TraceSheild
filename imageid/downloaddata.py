import tensorflow_datasets as tfds


data_dir = '/root/sjt2/synthid-text/mydata/'


builder = tfds.builder("wikipedia/20230601.en", data_dir=data_dir)


builder.download_and_prepare()


ds = builder.as_dataset(split='train')


info = builder.info
