# Publishing to GitHub

Your profile is at **https://github.com/AadityaK21** — the link you sent
(`https:/github/AadityaK21`) was missing a slash and the `.com`, so it will not
resolve.

I cannot push on your behalf: I have no credentials for your account, and the
bridge to your computer is not connected. Everything below runs on your machine
and takes about five minutes.

---

## 1. Check what you are about to publish

Do this before anything else. The repository is configured to exclude run
artefacts, but the cost of being wrong here is publishing a transcript.

```powershell
cd C:\lausanne
git status --short
```

You should see **no** files under `work/`, `output/`, `results/`,
`data/audio/`, or `data/gold/`. Those directories hold audio, transcripts, and
derived notes. If any appear, stop and tell me before continuing.

Also confirm no token leaked into a config file:

```powershell
findstr /S /I "hf_" *.yaml *.md *.py
```

That should return nothing. `HF_TOKEN` belongs in your environment, never in
the repository.

---

## 2. Create the repository on GitHub

Go to https://github.com/new and create it:

- **Name:** `lausanne`
- **Description:** `Local, confidentiality-preserving legal audio-to-notes pipeline with an evaluation harness`
- **Public**
- **Do not** initialise with a README, .gitignore, or licence — the repository
  already has all three, and adding them creates a conflict on the first push.

---

## 3. Push

If you unzipped what I sent, the git history is already there:

```powershell
cd C:\lausanne
git remote add origin https://github.com/AadityaK21/lausanne.git
git branch -M main
git push -u origin main
```

If `git log` shows nothing, initialise first:

```powershell
git init
git add -A
git commit -m "Lausanne: local confidentiality-preserving legal note pipeline"
git remote add origin https://github.com/AadityaK21/lausanne.git
git branch -M main
git push -u origin main
```

GitHub will ask you to authenticate. Use a **personal access token**, not your
password — create one at https://github.com/settings/tokens with `repo` scope.
Or install [GitHub CLI](https://cli.github.com/) and run `gh auth login` once,
which handles it for you.

---

## 4. After the first push

**Continuous integration runs itself.** `.github/workflows/ci.yml` runs both
test suites on Python 3.11 and 3.12 on every push. It installs only the four
lightweight dependencies — not torch, faster-whisper, or pyannote, which are
multi-gigabyte and unnecessary because both suites run without a GPU, without
models, and without network access. The README badge turns green on the first
successful run.

**Add topics** so people can find it. On the repository page, click the gear
next to *About* and add:

```
legal-ai  speech-recognition  speaker-diarization  summarization
hallucination  privacy  local-llm  whisper  ollama  design-science
```

**Turn on the security advisory form.** Settings → Security → enable private
vulnerability reporting. `SECURITY.md` links to it, and a confidentiality
defect in this project deserves a private channel.

**Point the About link** at the paper once it is public.

---

## What people will see first

The README opens with the demo command, because a stranger who can run
something in one line without installing CUDA is a stranger who might actually
try it:

```bash
python app/server.py --demo
```

Then the screenshot, then the three-mode table that explains why this is not
another Whisper wrapper.

---

## Making it useful to other people

You said you want it to help people. Three things would do more for that than
any amount of polish:

**A release.** Tag `v0.1.0` and write release notes naming the three
contributions in plain language. GitHub surfaces releases; commits it does not.

```powershell
git tag -a v0.1.0 -m "First release"
git push origin v0.1.0
```

**Real numbers in the README.** Right now it explains a method. Once you have
run the pipeline over ten Oyez arguments, the results table is the thing that
makes someone trust it — particularly the ratio between aggregate word error
rate and legal-term word error rate, which nobody else is reporting.

**An answer to "can I use this for my domain?"** The confidentiality and
provenance conditions that hold for legal work hold equally for clinical
records, regulatory filings, and incident investigation. A short section
saying so, and saying what would need to change, converts a legal-AI project
into infrastructure that several fields can use. That is also the reusability
evidence your design principles currently lack, so it serves the paper at the
same time.

---

## One caution

The repository is public and the licence permits use. It is worth being
deliberate that `NOTICE` and the README both say plainly that this is not legal
advice, that outputs are an aid to review rather than a substitute for the
record, and that legally load-bearing terms are transcribed less accurately
than the aggregate error rate suggests.

Someone will eventually run privileged audio through this. The disclaimers, the
attestation, and the enforced loopback binding are what make that defensible
rather than reckless. Do not quietly relax any of them for convenience later.
