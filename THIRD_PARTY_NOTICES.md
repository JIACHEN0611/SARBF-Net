# Third-party sources and notices

We retain source credits and distinguish the inherited main-model implementation from the transfer backbone sources. Paths below describe the integrated tree. Commit IDs identify the archived transfer package's source records; they do not imply that every upstream file is included.

| Source | Integrated area | Recorded revision / terms |
| --- | --- | --- |
| [OLinear](https://github.com/jackyue1994/OLinear) lineage | `model/components/` and inherited main-model runtime | Exact upstream commit and complete license record are not available in the inherited source |
| [CycleNet](https://github.com/ACAT-SCUT/CycleNet) | DLinear, PatchTST, CycleNet and associated transfer layers | `d807e51fc2dcd143885ee639d97965a7ab0926f4`; Apache 2.0 |
| [iTransformer](https://github.com/thuml/iTransformer) | iTransformer and associated transfer layers | `c2426e68ca13f74aaec08045c5c724d8ad328124`; MIT |
| [Time-Series-Library](https://github.com/thuml/Time-Series-Library) | TSMixer | `4e938a1767106324dd753b2a44832bf870a0252e`; MIT |
| [RevIN](https://github.com/ts-kim/RevIN) | Main and transfer RevIN files | Existing attribution retained; the two implementations remain separate |
| FAN, arXiv:2410.02675 | `layers/FANLayer.py` | Existing paper reference retained |
| [CKA-Centered-Kernel-Alignment](https://github.com/yuanli2333/CKA-Centered-Kernel-Alignment) | `utils/CKA.py` | Existing source link retained |
| Amazon/GluonTS time features | Main and transfer time-feature helpers | Existing copyright and Apache 2.0 notices retained |
| Element AI/N-BEATS | `data_provider/m4.py` | Existing CC BY-NC 4.0 / non-commercial notice retained |

Attention utilities retain inline references to Flowformer, FlashAttention-PyTorch and Informer2020. File headers and dependency notices remain part of the source distribution.

Transfer license texts are in `THIRD_PARTY_LICENSES/`: CycleNet-Apache-2.0.txt, iTransformer-MIT.txt and Time-Series-Library-MIT.txt. The inherited main-model provenance still requires a complete upstream license audit before assigning a blanket license to this repository; LICENSE.md records the present status.
