# Contributing

## Running the tests

Both suites run with no GPU, no models, and no network:

```bash
pip install "pydantic>=2.6,<3" requests PyYAML python-docx
python tests/test_pipeline.py     # 187 checks
python tests/test_app.py          # 32 checks
```

The application can be exercised the same way:

```bash
python app/server.py --demo
```

Demo mode replaces only the two GPU stages with fixtures. Merge, extraction,
and rendering are the production code paths.

## The tests encode claims, not just behaviour

Several checks exist because the accompanying paper makes a claim that the
code must keep true — that extractive modes are 100% verbatim, that invalid
identifiers are rejected and counted, that a privileged run refuses a
non-local endpoint, that the server will not bind a routable interface.

If a change makes one of those fail, the right response is usually to change
the code back, not the test. If the claim itself is wrong, change the claim in
`docs/DESIGN_PRINCIPLES.md` in the same pull request.

## Style

- Every stage is a separate process communicating through validated JSON. Do
  not add an import of `torch`, `faster_whisper`, or `pyannote` to `run.py` or
  `app/server.py`; the memory design depends on those processes staying clean.
- No new runtime dependencies without a strong reason. The application is
  standard library only on purpose.
- Comments explain *why*, particularly where a choice looks odd. Several do.
