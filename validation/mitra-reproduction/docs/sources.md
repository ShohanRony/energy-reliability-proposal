# Source and access audit

1. Full paper text read through the web reader:
   https://arxiv.org/html/2405.20876v1
2. PDF text inspected, 11 pages:
   https://arxiv.org/pdf/2405.20876
   PDF screenshot requests returned reference text but no usable image payload.
   No numerical figure values were read or digitized.
3. Referenced Liu repository README inspected:
   https://github.com/Eric-mingjie/rethinking-network-pruning
4. Weight-pruning README inspected:
   https://github.com/Eric-mingjie/rethinking-network-pruning/blob/master/cifar/weight-level/README.md
   It identifies the vgg19_bn entry point and convolution-only masks retained
   during fine-tuning. It does not establish Mitra's exact experiment revision.
5. Attempted model/training/pruning source retrieval failed, including the paths
   cifar/weight-level/cifar.py, cifar_prune.py, cifar_finetune.py, and candidate
   model paths. No implementation-level claim about those files is made.
6. Searched the exact paper title plus GitHub and author/pruning combinations.
   No accessible author-specific reproduction repository or checkpoints were
   identified. This is not a claim that none exist.
7. Native download probes are recorded in results/execution-attempt/environment.json.
   Their failures are separate from the web reader's successful textual access.

Recommended follow-up reference: the dataset authors' repository
https://github.com/hendrycks/robustness for the official CIFAR-10-C archive.
Dataset files and package wheels were not downloaded or executed here.
