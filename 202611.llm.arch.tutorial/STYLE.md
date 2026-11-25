# STYLE.md

Style rules for documents in this project.

## Files

- File name pattern: `ch-N-title.ext`, where `N` is the chapter number (no zero-padding),
  `title` is the chapter title in lowercase words joined by hyphens, and `ext` matches the
  file type (`.md`, `.pdf`, `.ppt`, …). Example: `ch-7-feed-forward-network.md`.

## References

- Each chapter ends with a `## References` section listing the published papers or textbooks
  that are the source of its information. Keep at most the 10 most important items.
- Reference papers must be published in a top conference in the field (e.g. NeurIPS, ICML,
  ICLR, ACL, EMNLP, NAACL, CVPR, MLSys, OSDI, SOSP) or in a high-reputation journal (e.g.
  JMLR, TACL, Neural Computation). arXiv-only preprints, technical reports, blog posts and
  workshop papers do not qualify.

## Diagrams

- Draw block diagrams in the horizontal direction (left to right). In Mermaid, use `flowchart LR`.
