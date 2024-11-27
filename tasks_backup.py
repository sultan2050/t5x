from random import shuffle
import functools
import seqio
import tensorflow_datasets as tfds
import t5
from t5.evaluation import metrics
from t5.data import preprocessors
import subprocess
import tensorflow.compat.v1 as tf
import sentencepiece as spm
vocabulary = seqio.SentencePieceVocabulary('gs://sultan-t5x/spiece.model', extra_ids=100)
output_features = {
    'inputs': seqio.Feature(vocabulary=vocabulary),
    'targets': seqio.Feature(vocabulary=vocabulary)
}

def arabic_dataset(split, shuffle_files = False):
    del shuffle_files
    files_name_pmc =  list(map(lambda x: x.strip(), subprocess.run(['gsutil', 'ls', 'gs://sultan-t5x/arabic_text/*.txt'], stdout=subprocess.PIPE).stdout.splitlines()))
    shuffle(files_name_pmc)

    print(files_name_pmc[0])

    ds = tf.data.TextLineDataset(
       files_name_pmc
    )
    ds = ds.map(lambda *ex: dict(zip(['title', 'text'], ['None',ex[0]])))
    ds = ds.shuffle(buffer_size=4000000)

    return ds

  
print("A few raw validation examples...")
for ex in tfds.as_numpy(arabic_dataset("train").take(5)):
  print(ex)

t5.data.TaskRegistry.remove('arabic_dataset')
t5.data.TaskRegistry.add(
    'arabic_dataset',
    dataset_fn = arabic_dataset,
    splits = ['train'],
    text_preprocessor =  functools.partial(
        t5.data.preprocessors.rekey,
        key_map = {'inputs': None, 'targets': 'text'},
    ),
    token_preprocessor = t5.data.preprocessors.unsupervised,
    output_features=output_features,
    metric_fns = [],
)
