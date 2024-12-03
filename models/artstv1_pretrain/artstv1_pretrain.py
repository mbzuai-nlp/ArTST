# coding=utf-8
# Copyright 2022 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
from pathlib import Path
from typing import Dict, List, Tuple

import datasets

_MODELNAME = "pretrain_checkpoint"
_HOMEPAGE = "https://github.com/mbzuai-nlp/ArTST"
_DESCRIPTION = """\
Pre-Train checkpoint for ArTST by MBZUAI Speech Team
"""

_CITATION = """\
@inproceedings{toyin2023artst,
  title={ArTST: Arabic Text and Speech Transformer},
  author={Toyin, Hawau and Djanibekov, Amirbek and Kulkarni, Ajinkya and Aldarmaki, Hanan},
  booktitle={Proceedings of ArabicNLP 2023},
  pages={41--51},
  year={2023}
}
"""

_LANGUAGES = ['ar']

_LICENSE = "MIT (mit)"

_LOCAL = False


_URLS = {
    _MODELNAME: "https://huggingface.co/MBZUAI/ArTST/resolve/main/pretrain_checkpoint.pt",
}

_SUPPORTED_TASKS = ['pretrain']
_SOURCE_VERSION = "1.0.0"


class ArTSTModel(datasets.GeneratorBasedBuilder):
    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)

    def _info(self) -> datasets.DatasetInfo:        
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=None,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        urls = _URLS[_MODELNAME]
        checkpoint_file = dl_manager.download(urls)
        
        
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": checkpoint_file,
                },
            )
        ]

    def _generate_examples(self, filepath: Path) -> Tuple[int, Dict]:
        yield 0, {"filepath": filepath}
        
        

if __name__ == "__main__":
    datasets.load_dataset(__file__, trust_remote_code=True)