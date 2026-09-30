# Security policy

## Supported versions

Only the latest release of this repository receives fixes.

## Reporting a vulnerability

This is a research code base that runs locally and does not handle network input or credentials. If you believe you have found a security problem, for example in how results or model files are loaded, please do not open a public issue. Report it by email to **dfr@esmad.ipp.pt** with a description, the steps to reproduce it and the version or commit affected.

You can expect an acknowledgement within seven days. Once the problem is confirmed and fixed, the fix will be released and the report credited unless you ask otherwise.

Note that `results/models/*.pkl` are Python pickle files. Only load pickle files that come from this repository or from a source you trust.
