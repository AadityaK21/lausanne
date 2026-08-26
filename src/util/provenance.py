"""
Run-level provenance and confidentiality attestation (DP1).

The confidentiality claim in Section 8.3 is only as strong as its auditability.
A system that runs locally but keeps no record of where each output was produced
cannot support the claim when it is questioned - by a supervising partner, an
opposing party, a bar association, or a client. "We ran it locally" is an
assertion. This module produces the artefact that makes it a demonstration.

Every run emits a manifest recording:

  * a SHA-256 of the input audio, so the record is bound to a specific file
  * for every stage: the model, the device, and the ENDPOINT it executed against
  * a classification of every network endpoint contacted, separating
    DATA EGRESS (user content sent somewhere) from ASSET FETCH (model weights
    retrieved), because these are entirely different disclosures and conflating
    them makes the record useless
  * a SHA-256 of each artefact produced, so a later reader can verify the
    manifest describes the files actually on disk

The distinction in the third bullet is the one that matters and the one most
easily got wrong. Downloading Whisper weights from Hugging Face is outbound
network traffic that discloses nothing about the proceeding. Sending a
transcript to a hosted inference endpoint discloses everything. A manifest that
reports "network activity: yes" for both is worse than no manifest, because it
trains its reader to ignore it.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import socket
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from .logging import get_logger

log = get_logger("provenance")


# --------------------------------------------------------------------------
# Material classification
# --------------------------------------------------------------------------

PRIVILEGED = "privileged"
PUBLIC = "public"

VALID_CLASSES = (PRIVILEGED, PUBLIC)


class TierViolation(RuntimeError):
    """Raised when a run would send privileged material off-host."""


# --------------------------------------------------------------------------
# Endpoint classification
# --------------------------------------------------------------------------

_LOOPBACK_NAMES = {"localhost", "127.0.0.1", "::1", "0.0.0.0", "[::1]"}

# Ollama can serve models that execute on Ollama's infrastructure rather than
# on this machine. They are requested through the ordinary local endpoint, so
# the URL looks like loopback while the prompt travels to a third party.
#
# This is the most dangerous shape a confidentiality bug can take: the
# attestation would truthfully report a loopback endpoint and thereby report
# "no data egress" about a run that egressed everything. Endpoint locality is
# therefore necessary but not sufficient - the model identity has to be checked
# too.
_CLOUD_MODEL_MARKERS = (":cloud", "-cloud", "/cloud")


def is_cloud_model(model: Optional[str]) -> bool:
    """True when a model name denotes remote execution despite a local URL."""
    if not model:
        return False
    name = model.strip().lower()
    return any(marker in name for marker in _CLOUD_MODEL_MARKERS)


def is_local_endpoint(host_or_url: str) -> bool:
    """
    True when an endpoint cannot leave the machine.

    Deliberately strict: anything that is not unambiguously loopback is treated
    as remote. A LAN address is not loopback. A hostname that happens to resolve
    to loopback today is not loopback either, because that can change without
    the manifest changing.
    """
    if not host_or_url:
        return True  # in-process execution, no endpoint at all

    value = host_or_url.strip()
    if "://" in value:
        parsed = urlparse(value)
        hostname = (parsed.hostname or "").lower()
    else:
        hostname = value.split(":")[0].lower()

    return hostname in _LOOPBACK_NAMES


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> Optional[str]:
    """SHA-256 of a file, or None if it does not exist."""
    p = Path(path)
    if not p.exists() or not p.is_file():
        return None
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


# --------------------------------------------------------------------------
# Manifest
# --------------------------------------------------------------------------


class StageRecord(BaseModel):
    stage: str
    model: Optional[str] = None
    tier: Optional[str] = None
    device: Optional[str] = None
    # Where inference actually happened. Empty means in-process on this host.
    endpoint: str = ""
    endpoint_is_local: bool = True
    elapsed_seconds: Optional[float] = None
    artifact: Optional[str] = None
    artifact_sha256: Optional[str] = None


class RunManifest(BaseModel):
    run_id: str
    created_utc: str
    hostname: str
    platform: str

    material_class: str = PRIVILEGED
    input_path: str
    input_sha256: Optional[str] = None
    input_bytes: Optional[int] = None

    stages: List[StageRecord] = Field(default_factory=list)

    # Endpoints that received user content. Should be empty or loopback-only
    # for a privileged run.
    data_egress: List[str] = Field(default_factory=list)
    # Endpoints contacted to retrieve model weights. Discloses nothing about
    # the proceeding, but recorded for completeness.
    asset_fetch: List[str] = Field(default_factory=list)

    attestation: Optional[str] = None
    attestation_ok: Optional[bool] = None
    findings: List[str] = Field(default_factory=list)


def new_manifest(
    run_id: str, input_path: str, material_class: str = PRIVILEGED
) -> RunManifest:
    if material_class not in VALID_CLASSES:
        raise ValueError(
            f"material_class must be one of {VALID_CLASSES}, got {material_class!r}"
        )
    p = Path(input_path)
    return RunManifest(
        run_id=run_id,
        created_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        hostname=socket.gethostname(),
        platform=f"{platform.system()} {platform.release()} / Python "
                 f"{platform.python_version()}",
        material_class=material_class,
        input_path=str(p),
        input_sha256=sha256_file(p),
        input_bytes=p.stat().st_size if p.exists() else None,
    )


def record_stage(
    manifest: RunManifest,
    stage: str,
    *,
    model: Optional[str] = None,
    tier: Optional[str] = None,
    device: Optional[str] = None,
    endpoint: str = "",
    elapsed_seconds: Optional[float] = None,
    artifact: Optional[str] = None,
) -> None:
    """Append a stage record, hashing its artefact and classifying its endpoint."""
    # A cloud-hosted model makes the stage remote no matter how local the URL is.
    local = is_local_endpoint(endpoint) and not is_cloud_model(model)
    manifest.stages.append(
        StageRecord(
            stage=stage,
            model=model,
            tier=tier,
            device=device,
            endpoint=endpoint,
            endpoint_is_local=local,
            elapsed_seconds=elapsed_seconds,
            artifact=str(artifact) if artifact else None,
            artifact_sha256=sha256_file(artifact) if artifact else None,
        )
    )
    if not local:
        # This stage sent user content off this machine, either to a
        # non-loopback endpoint or through a cloud-hosted model.
        label = endpoint or "(local endpoint)"
        if is_cloud_model(model):
            label = f"{label} via cloud model {model!r}"
        if label not in manifest.data_egress:
            manifest.data_egress.append(label)


def record_asset_fetch(manifest: RunManifest, endpoint: str) -> None:
    if endpoint and endpoint not in manifest.asset_fetch:
        manifest.asset_fetch.append(endpoint)


# --------------------------------------------------------------------------
# Attestation
# --------------------------------------------------------------------------


def attest(manifest: RunManifest) -> RunManifest:
    """
    Decide and record whether this run honoured its material classification.

    A privileged run passes only if every stage executed either in-process or
    against a loopback endpoint. There is no partial credit and no warning
    tier: either the material stayed on the host or it did not.
    """
    findings: List[str] = []

    remote = [s for s in manifest.stages if not s.endpoint_is_local]
    for s in remote:
        findings.append(
            f"stage {s.stage!r} executed against non-local endpoint "
            f"{s.endpoint!r} (model {s.model})"
        )

    if manifest.material_class == PRIVILEGED:
        ok = not remote
        if ok:
            attestation = (
                "PRIVILEGED MATERIAL - NO DATA EGRESS. Every stage of this run "
                "executed on this host, either in-process or against a loopback "
                "endpoint. No audio, transcript, or derived note content was "
                "transmitted to any external service."
            )
        else:
            attestation = (
                "ATTESTATION FAILED. This run was classified as privileged but "
                "sent content to one or more non-local endpoints. The "
                "confidentiality property does NOT hold for these outputs."
            )
    else:
        ok = True
        if remote:
            attestation = (
                "PUBLIC MATERIAL. This run used one or more remote inference "
                "endpoints. That is permitted for this classification; the "
                "outputs carry no confidentiality guarantee."
            )
        else:
            attestation = (
                "PUBLIC MATERIAL, executed entirely on this host. No data "
                "egress occurred, though none was prohibited."
            )

    if manifest.asset_fetch:
        findings.append(
            f"model weights retrieved from {len(manifest.asset_fetch)} "
            f"endpoint(s): {', '.join(manifest.asset_fetch)}. This is inbound "
            f"asset retrieval and discloses no run content."
        )

    manifest.attestation = attestation
    manifest.attestation_ok = ok
    manifest.findings = findings
    return manifest


def guard_tier(material_class: str, tier_name: str, endpoint: str,
               model: Optional[str] = None) -> None:
    """
    Refuse, before any inference happens, to send privileged material off-host.

    DP1 states tier segregation as a design property. Stating it in
    documentation makes it a convention that a hurried user violates at 2am.
    Enforcing it at the point of execution makes it a property of the artefact.

    Two independent ways a run can leave the machine, and the second is the one
    that looks safe: a remote endpoint, or a cloud-hosted model reached through
    a local one.
    """
    if material_class != PRIVILEGED:
        return

    if is_cloud_model(model):
        raise TierViolation(
            f"Refusing to run model {model!r}.\n"
            f"Despite being reached at {endpoint or 'a local endpoint'}, a "
            f"cloud-tagged model executes on the provider's infrastructure, so "
            f"your transcript would be sent to a third party. The local URL "
            f"makes this look safe; it is not.\n\n"
            f"Use a locally-executing model, or re-run with --material public "
            f"if this recording is genuinely publishable."
        )

    if is_local_endpoint(endpoint):
        return

    raise TierViolation(
        f"Refusing to run tier {tier_name!r} against endpoint {endpoint!r}.\n"
        f"This run is classified as PRIVILEGED, and that endpoint is not on "
        f"this host.\n\n"
        f"If this material is genuinely public - a published court recording, "
        f"a training corpus - re-run with --material public. Do not use that "
        f"flag to silence this error for material you would not publish."
    )


# --------------------------------------------------------------------------
# Verification
# --------------------------------------------------------------------------


def verify(manifest_path: str | Path) -> tuple[bool, List[str]]:
    """
    Re-check a manifest against the files on disk.

    Answers a narrow question honestly: do the artefacts still hash to what the
    manifest says? It does not and cannot prove that no egress occurred - a
    manifest is a record produced by the system, not a network capture. What it
    prevents is a manifest being silently paired with outputs it does not
    describe.
    """
    path = Path(manifest_path)
    if not path.exists():
        return False, [f"manifest not found: {path}"]

    with open(path, "r", encoding="utf-8") as fh:
        manifest = RunManifest.model_validate(json.load(fh))

    problems: List[str] = []

    current_input = sha256_file(manifest.input_path)
    if manifest.input_sha256 and current_input != manifest.input_sha256:
        if current_input is None:
            problems.append(
                f"input file no longer present: {manifest.input_path}"
            )
        else:
            problems.append(
                f"input file has changed since the run "
                f"(manifest {manifest.input_sha256[:12]}..., "
                f"now {current_input[:12]}...)"
            )

    for s in manifest.stages:
        if not s.artifact or not s.artifact_sha256:
            continue
        current = sha256_file(s.artifact)
        if current is None:
            problems.append(f"artefact missing: {s.artifact}")
        elif current != s.artifact_sha256:
            problems.append(
                f"artefact modified since the run: {s.artifact} "
                f"(manifest {s.artifact_sha256[:12]}..., now {current[:12]}...)"
            )

    if manifest.attestation_ok is False:
        problems.append(
            "the run's own attestation FAILED - privileged material reached a "
            "non-local endpoint"
        )

    return (not problems), problems


def render(manifest: RunManifest) -> str:
    """Human-readable attestation, for pasting into a file note or an appendix."""
    lines: List[str] = []
    lines.append("=" * 70)
    lines.append("  RUN PROVENANCE AND CONFIDENTIALITY ATTESTATION")
    lines.append("=" * 70)
    lines.append("")
    lines.append(f"  Run           : {manifest.run_id}")
    lines.append(f"  Produced      : {manifest.created_utc}")
    lines.append(f"  Host          : {manifest.hostname}")
    lines.append(f"  Platform      : {manifest.platform}")
    lines.append(f"  Classification: {manifest.material_class.upper()}")
    lines.append("")
    lines.append(f"  Input         : {Path(manifest.input_path).name}")
    if manifest.input_sha256:
        lines.append(f"  SHA-256       : {manifest.input_sha256}")
    if manifest.input_bytes:
        lines.append(f"  Size          : {manifest.input_bytes:,} bytes")
    lines.append("")
    lines.append("  Stages")
    lines.append("  " + "-" * 66)
    for s in manifest.stages:
        where = "in-process (this host)" if not s.endpoint else (
            f"{s.endpoint} {'[LOCAL]' if s.endpoint_is_local else '[REMOTE]'}"
        )
        lines.append(f"    {s.stage:<18} {s.model or '(no model)'}")
        lines.append(f"    {'':<18} executed: {where}")
        if s.artifact_sha256:
            lines.append(
                f"    {'':<18} artefact: {Path(s.artifact).name} "
                f"({s.artifact_sha256[:16]}...)"
            )
    lines.append("")

    if manifest.data_egress:
        lines.append("  DATA EGRESS (run content sent off-host)")
        for e in manifest.data_egress:
            lines.append(f"    - {e}")
    else:
        lines.append("  DATA EGRESS: none")
    lines.append("")

    if manifest.asset_fetch:
        lines.append("  Asset retrieval (weights in; discloses no run content)")
        for e in manifest.asset_fetch:
            lines.append(f"    - {e}")
        lines.append("")

    lines.append("  " + "-" * 66)
    verdict = "PASS" if manifest.attestation_ok else "FAIL"
    lines.append(f"  ATTESTATION: {verdict}")
    lines.append("")
    for chunk in _wrap(manifest.attestation or "", 66):
        lines.append(f"  {chunk}")
    if manifest.findings:
        lines.append("")
        lines.append("  Notes")
        for f in manifest.findings:
            for i, chunk in enumerate(_wrap(f, 64)):
                lines.append(f"    {'- ' if i == 0 else '  '}{chunk}")
    lines.append("")
    lines.append("=" * 70)
    return "\n".join(lines)


def _wrap(text: str, width: int) -> List[str]:
    words, lines, cur = text.split(), [], ""
    for w in words:
        if cur and len(cur) + 1 + len(w) > width:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        lines.append(cur)
    return lines or [""]
