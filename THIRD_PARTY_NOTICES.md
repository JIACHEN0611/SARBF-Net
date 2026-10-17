# Third-party provenance retained from the supplied archive

This file records evidence in the uploaded source, not an independent audit of
all upstream history. The archive supplied no repository-wide license, upstream
commit locks, or complete copies of upstream license files. They have not been
invented during renaming.

| Source area | Evidence in supplied files |
| --- | --- |
| `model/OLinear.py` and inherited runtime | Original model/backbone name and release notes identify OLinear lineage; exact upstream commit/license was not included |
| `layers/RevIN.py` | Header credits `https://github.com/ts-kim/RevIN` with modifications |
| `layers/SelfAttention_Family.py` | Inline comments credit Flowformer, FlashAttention-PyTorch and Informer2020 paths |
| `layers/FANLayer.py` | Docstring refers to FAN, arXiv:2410.02675 |
| `utils/CKA.py` | Header links to `yuanli2333/CKA-Centered-Kernel-Alignment` |
| `utils/timefeatures.py` | Existing Amazon/GluonTS copyright and Apache 2.0 notice retained |
| `data_provider/m4.py` | Existing Element AI/N-BEATS header states CC BY-NC 4.0 and non-commercial conditions; header retained |

Generic loaders and attention utilities are retained because they are part of
the inherited import closure. No claim is made that renaming these files would
transfer authorship. The current model entry describes SAER/BCIF while keeping
these source credits. Verify upstream licenses and obtain/preserve the required
license texts and notices before assigning a repository-wide distribution
license. This packaging task does not resolve missing provenance.
