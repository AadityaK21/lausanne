# Security and confidentiality

This project's central claim is that privileged audio does not leave the
machine. A defect in that claim is more serious here than an ordinary bug.

## Please report privately

Open a [security advisory](https://github.com/AadityaK21/lausanne/security/advisories/new)
rather than a public issue if you find:

- any path by which run content (audio, transcript, or notes) reaches a
  non-local endpoint while a run is classified `privileged`
- any way for `provenance.json` or `attestation.txt` to report something that
  is not true of the run
- a way to make the app server bind an interface other than loopback

## What is in scope

The attestation records what the system did. It is not a network capture, and
it cannot prove the absence of egress by a component outside the pipeline —
an operating-system telemetry agent, a backup client indexing the work
directory, or a model server configured to forward requests. Those are real
risks and out of this project's control; the manifest is a record of the
pipeline's own behaviour, and is documented as such.

## Not a vulnerability

- Downloading model weights from Hugging Face. This is inbound asset
  retrieval, recorded separately from data egress, and discloses nothing
  about a proceeding.
- The `--material public` flag permitting remote inference. That is the
  documented purpose of the classification.
