import functools
import seqio
import tensorflow_datasets as tfds
from t5.evaluation import metrics
from t5.data import preprocessors
import subprocess
import tensorflow.compat.v1 as tf
from random import shuffle

vocabulary = seqio.SentencePieceVocabulary('gs://sultan-t5x/M3Vocab.model', extra_ids=100)
output_features = {
    'inputs': seqio.Feature(vocabulary=vocabulary),
    'targets': seqio.Feature(vocabulary=vocabulary)
}
#source=seqio.TfdsDataSource(tfds_name='wikipedia/20230601.ar:1.0.0'),
def arabic_dataset(split, shuffle_files = False):
    del shuffle_files
    files_name_pmc =  list(map(lambda x: x.strip(), subprocess.run(['gsutil', 'ls', 'gs://sultan-t5x/kenlm30gb_mimic_corpora/*.txt'], stdout=subprocess.PIPE).stdout.splitlines()))
    shuffle(files_name_pmc)

    print(files_name_pmc[0])

    ds = tf.data.TextLineDataset(
       files_name_pmc
    )
    ds = ds.map(lambda *ex: dict(zip(['title', 'text'], ['None',ex[0]])))
    ds = ds.shuffle(buffer_size=4000000)

    return ds

seqio.TaskRegistry.remove('arabic_dataset')
seqio.TaskRegistry.add(
    'arabic_dataset',
    source=seqio.FunctionDataSource(dataset_fn=arabic_dataset, splits = ['train']),
    preprocessors=[
        functools.partial(
            preprocessors.rekey, key_map={
                "inputs": None,
                "targets": "text"
            }),
        seqio.preprocessors.tokenize,
        seqio.CacheDatasetPlaceholder(),
        preprocessors.span_corruption,
        seqio.preprocessors.append_eos_after_trim,

    ],
    output_features=output_features ,
    metric_fns=[])
