#!/usr/bin/env python3
"""Draft/run CPU interval measurement only; no automatic freeze or model calls."""
import argparse
import csv
import fcntl
import importlib
import io
import os
from pathlib import Path
import platform
import sys
import tempfile

import prepare_saraga_external_cohort_v1 as util

VERSION = "saraga_external103_materialization_v1"
STAGE = "saraga_external103_interval_measurement_only"
SELECTION = "preregistration/saraga_external103_selection_frozen_v1.json"
SELECTION_SHA = "270d14dc10cc5f16811f799197f6848a2ff8ee213a99d85e890b52b79c1ed56b"
HELPER_SHA = "795e1538521d0ccd08168e598bead4f6d4eb7a759a1eb625365a996559347e9c"
UTIL_SHA = "e7fd959bd7c4341fde6c544513ee696c2f55f103ff41853f134185b48f2131a3"
RD = Path("/mnt/nfs-data/users/yi/audio_phenomena_expansion_20260907")
CODE = Path(__file__).absolute()
CONFIG = {"workers": 1, "device": "cpu", "all_selected_required": True,
          "output_rate": 44100, "output_frames": 2646000, "output_channels": 2,
          "format": "WAV", "subtype": "FLOAT", "normalization": False,
          "dc_removal": False, "limiting": False, "replacement": False,
          "failure_policy": "retain proof; continue other fixed identities; no retry of failed item in same output"}


def path_ok(path):
    util.require(".." not in Path(path).parts, "Parent traversal forbidden")
    return util.no_symlinks(path)


def file_sha(path):
    return util.sha(util.read_bytes(path))


def get_backend():
    helper_path = CODE.with_name("saraga_interval_audio_v1.py")
    util.require(file_sha(helper_path) == HELPER_SHA, "Unreviewed interval helper")
    return importlib.import_module("saraga_interval_audio_v1")


def runtime(helper):
    return {"python": sys.version, "python_executable": sys.executable,
            "platform": platform.platform(), "numpy": helper.np.__version__,
            "scipy": helper.scipy.__version__, "soundfile": helper.sf.__version__,
            "libsndfile": helper.sf.__libsndfile_version__, "device": "cpu", "workers": 1,
            "thread_environment": {k: os.environ.get(k) for k in
                                   ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}}


def csv_bytes(rows):
    util.require(rows, "Empty CSV output forbidden")
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode()


def publish_directory(destination, files):
    """NFS: exclusive reservation, then atomic no-replace hardlink commit marker."""
    destination = path_ok(destination)
    util.require(destination.parent.is_dir() and not destination.exists(), "Publication target conflict")
    destination.mkdir()  # Reservation is intentionally retained if publication fails.
    for name, raw in files.items():
        util.require(Path(name).name == name and name != "COMMIT.json", "Invalid publication name")
        with (destination / name).open("xb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
    commit_directory(destination, set(files))


def commit_directory(destination, names):
    inventory = {name: file_sha(destination / name) for name in sorted(names)}
    commit = util.canonical({"status": "committed", "files_sha256": inventory})
    with tempfile.NamedTemporaryFile(dir=destination, prefix=".commit-", delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(commit)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(temporary, destination / "COMMIT.json", follow_symlinks=False)
    finally:
        temporary.unlink()


def verify_publication(destination, *, allow_extra=False):
    destination = path_ok(destination)
    marker = util.strict_json(util.read_bytes(destination / "COMMIT.json"))
    util.require(marker["status"] == "committed", "Uncommitted publication")
    names = set(marker["files_sha256"])
    util.require(all(Path(n).name == n and n != "COMMIT.json" for n in names), "Invalid committed filename")
    if not allow_extra:
        util.require({p.name for p in destination.iterdir()} == names | {"COMMIT.json"}, "Committed inventory mismatch")
    for name, digest in marker["files_sha256"].items():
        util.require(file_sha(destination / name) == digest, "Committed payload hash mismatch")
    return marker


def build_contract(root, output, *, _synthetic=None, _backend=None):
    """Metadata and library-version inspection only; never opens source audio."""
    root, output = path_ok(root), path_ok(output)
    synthetic = _synthetic is not None
    helper = _backend if synthetic and _backend is not None else get_backend()
    inventory = {}
    def bind(relative, expected=None):
        util.require(not Path(relative).is_absolute() and ".." not in Path(relative).parts, "Unsafe metadata binding")
        raw = util.read_bytes(root / relative)
        digest = util.sha(raw)
        util.require(expected is None or digest == expected, "Input hash mismatch: " + relative)
        inventory[relative] = digest
        return raw
    freeze = util.strict_json(bind(SELECTION, _synthetic["selection_sha256"] if synthetic else SELECTION_SHA))
    util.require(freeze["status"] == "frozen" and freeze["authorized_stage"] == "saraga_external103_cohort_selection_only"
                 and freeze["cohort_identities_and_intervals_frozen"] is True
                 and freeze["real_audio_measurement_accepted"] is False, "Selection freeze status/scope invalid")
    if synthetic:
        util.require(freeze.get("synthetic_test_only") is True, "Synthetic selection marker required")
    else:
        util.require(output.parent == RD and output.name.startswith("saraga_external103_"), "Output must be a new dedicated Saraga external103 directory under fixed RD")
        util.require(file_sha(Path(util.__file__)) == UTIL_SHA, "Selection utility implementation changed")
    selection_dir = freeze["selection_draft_directory"]
    selected_files = {name: bind(f"{selection_dir}/{name}", digest)
                      for name, digest in freeze["selection_files_sha256"].items()}
    draft = util.strict_json(selected_files["selection_draft.json"])
    util.require(draft == freeze["selection_draft"] and draft["status"] == "draft", "Frozen selection body mismatch")
    prior_inventory = util.strict_json(selected_files["input_inventory.json"])
    for relative, entry in prior_inventory.items():
        bind(relative, entry["sha256"])
    rows = util.csv_rows(selected_files["metadata.csv"])
    util.require(1 <= len(rows) <= 4 if synthetic else len(rows) == 103, "Fixed cohort size mismatch")
    util.require(len(util.keyed(rows, "item_id")) == len(rows), "Duplicate item identity")
    rows.sort(key=lambda r: r["item_id"])
    expected_ids = util.sha("\n".join(sorted(r["mbid"] for r in rows)).encode())
    util.require(synthetic or expected_ids == util.ELIGIBLE_ID_SHA, "Selection ID hash mismatch")
    for row in rows:
        util.ident(row["mbid"], synthetic)
        util.require(row["item_id"] == "saraga_hindustani_" + row["mbid"] and row["role"] == util.ROLE
                     and row["label"] == "0" and row["source_id"] == util.SOURCE
                     and row["classifier_admission"] == "False" and row["evaluation_allowed"] == "False", "Selection identity/role/admission mismatch")
        source = Path(row["source_audio_path"])
        expected_source = Path(_synthetic["raw_root"]) / (row["mbid"] + ".mp3") if synthetic else Path(util.RAW_ROOT) / (row["mbid"] + ".mp3")
        util.require(source == expected_source, "Noncanonical source path")
        receipt_name = f"{util.PHYSICAL}/receipts/{row['mbid']}.json"
        util.require(receipt_name in prior_inventory, "Physical receipt missing from frozen input inventory")
        receipt = util.strict_json(bind(receipt_name, prior_inventory[receipt_name]["sha256"]))
        record = receipt["record"]
        util.require(receipt["record_sha256"] == util.seal(record), "Physical record seal mismatch")
        util.require(record["item"]["mbid"] == row["mbid"], "Receipt item identity mismatch")
        measurement = record["measurement"]
        for field, key in (("native_sample_rate_hz", "sample_rate"), ("native_channels", "channels"),
                           ("actual_eof_frames", "actual_frames"), ("soundfile_header_frames", "header_frames")):
            util.require(str(measurement[key]) == row[field], "Selection/physical measurement mismatch: " + field)
        sr, frames = measurement["sample_rate"], measurement["actual_frames"]
        count, start = 60 * sr, (frames - 60 * sr) // 2
        util.require(sr in (44100, 48000) and frames >= count and measurement["channels"] == 2, "Invalid physical crop")
        for field, value in (("crop_start_frame", start), ("crop_frames", count), ("crop_end_frame_exclusive", start + count)):
            util.require(row[field] == str(value), "Frozen crop integer mismatch")
        util.require(round(float(row["source_offset_seconds"]) * sr) == start, "Frozen offset does not roundtrip")
        util.require(row["acquisition_raw_sha256"] == measurement["raw_hashes_before_decode"]["sha256"]
                     == measurement["raw_hashes_after_decode"]["sha256"], "Frozen raw hash mismatch")
    code_bindings = {CODE.name: file_sha(CODE), Path(util.__file__).name: file_sha(Path(util.__file__))}
    if not synthetic:
        code_bindings["saraga_interval_audio_v1.py"] = file_sha(CODE.with_name("saraga_interval_audio_v1.py"))
    rt = runtime(helper)
    contract = {"schema_version": 1, "version": VERSION, "purpose": "interval_measurement_only", "authorized_stage": STAGE,
                "synthetic_test_only": synthetic, "root": str(root), "output": str(output),
                "selection_freeze_sha256": inventory[SELECTION], "inputs_sha256": inventory,
                "code_sha256": code_bindings, "runtime": rt, "runtime_sha256": util.seal(rt),
                "configuration": CONFIG, "helper_configuration": helper.CONFIG,
                "rows": rows, "selected_ids": [r["item_id"] for r in rows],
                "classifier_admission": False, "fitting_authorized": False, "scoring_authorized": False,
                "neural_inference_authorized": False, "threshold_tuning_authorized": False,
                "raw_replacement_authorized": False}
    return contract


def create_draft(root, output, draft_dir, **kwargs):
    util.require(not path_ok(output).exists(), "Measurement output must not exist at draft stage")
    contract = build_contract(root, output, **kwargs)
    wrapper = {"status": "draft", "contract": contract, "contract_sha256": util.seal(contract),
               "audio_opened": False, "measurement_started": False, "requires_independent_root_freeze": True}
    publish_directory(draft_dir, {"measurement_contract_draft.json": util.canonical(wrapper)})
    return wrapper


def check_proof(row, record, proof):
    util.require(proof["status"] == "passed_interval_proof_not_classifier_admission"
                 and proof["physical_record_sha256"] == util.seal(record)
                 and proof["physical_contract_sha256"] == record["contract_sha256"]
                 and proof["source_audio_path"] == row["source_audio_path"] and proof["mbid"] == row["mbid"], "Helper source/record proof mismatch")
    for name in ("crop_start_frame", "crop_frames", "crop_end_frame_exclusive"):
        util.require(type(proof[name]) is int and proof[name] == int(row[name]), "Helper crop coordinate mismatch")
    util.require(proof["seek_proof"]["exact_bytes_equal"] is True and proof["seek_proof"]["exact_array_equal"] is True
                 and proof["real_empty_read_observed"] is True, "Missing sequential/fresh seek proof")


def verify_written(path, helper, sample_sha):
    path_ok(path)
    with helper.sf.SoundFile(path) as audio:
        util.require((audio.samplerate, audio.channels, audio.frames, audio.format, audio.subtype)
                     == (44100, 2, 2646000, "WAV", "FLOAT"), "Output WAV properties mismatch")
        data = audio.read(dtype="float32", always_2d=True)
    util.require(data.shape == (2646000, 2) and helper.np.isfinite(data).all(), "Output waveform invalid")
    util.require(helper._sample_sha(data, "<f4") == sample_sha, "FLOAT readback byte identity mismatch")
    return {"file_sha256": file_sha(path), "bytes": path.stat().st_size,
            "float32_sha256": sample_sha, "readback_verified": True}


def publish_item(root, output, row, contract, helper):
    item_id, digest = row["item_id"], util.seal(contract)
    destination = output / "items" / item_id
    receipt_name = f"{util.PHYSICAL}/receipts/{row['mbid']}.json"
    raw_receipt = util.read_bytes(root / receipt_name)
    util.require(util.sha(raw_receipt) == contract["inputs_sha256"][receipt_name], "Physical receipt changed before item")
    receipt = util.strict_json(raw_receipt)
    if destination.exists():
        verify_publication(destination)
        util.require(set(p.name for p in destination.iterdir()) == {"audio.wav", "proof.json", "COMMIT.json"}, "Resume item inventory mismatch")
        saved = util.strict_json(util.read_bytes(destination / "proof.json"))
        util.require(saved["contract_sha256"] == digest and saved["selection_row_sha256"] == util.seal(row)
                     and saved["status"] == "passed_interval_materialization_not_classifier_admission", "Resume item contract mismatch")
        check_proof(row, receipt["record"], saved["interval_proof"])
        raw, _ = helper._raw_hashes(row["source_audio_path"])
        util.require(raw == saved["interval_proof"]["raw_hashes_after"], "Resume source hash changed")
        observed = verify_written(destination / "audio.wav", helper, saved["interval_proof"]["standardized_float32_sha256"])
        util.require(observed == saved["wav"], "Resume output identity changed")
        return saved
    destination.mkdir()  # Incomplete reservations stay visible and fail closed on resume.
    proof = {}
    try:
        native, standard, proof = helper.verify_center_interval(row["source_audio_path"], receipt)
        check_proof(row, receipt["record"], proof)
        util.require(proof["configuration"] == helper.CONFIG, "Helper configuration parity mismatch")
        util.require(helper._sample_sha(standard, "<f4") == proof["standardized_float32_sha256"], "Returned standard samples mismatch")
        del native
        wav = destination / "audio.wav"
        helper.sf.write(wav, standard, 44100, format="WAV", subtype="FLOAT")
        written = verify_written(wav, helper, proof["standardized_float32_sha256"])
        saved = {"status": "passed_interval_materialization_not_classifier_admission", "item_id": item_id,
                 "contract_sha256": digest, "selection_row_sha256": util.seal(row), "role": util.ROLE,
                 "classifier_admission": False, "interval_proof": proof, "wav": written}
        with (destination / "proof.json").open("xb") as stream:
            stream.write(util.canonical(saved))
            stream.flush()
            os.fsync(stream.fileno())
        with wav.open("rb") as stream:
            os.fsync(stream.fileno())
        commit_directory(destination, {"audio.wav", "proof.json"})
        return saved
    except Exception as error:
        if not getattr(error, "audit", None):
            error.audit = proof
        raise


def native_row(row, saved):
    proof = saved["interval_proof"]
    return {"id": row["item_id"], "audio_path": row["source_audio_path"],
            "audio_offset_s": row["source_offset_seconds"], "native_sample_rate_hz": row["native_sample_rate_hz"],
            "duration_view": "60s", "duration_sec": 60, "registered_raw_sha256": row["acquisition_raw_sha256"],
            "label": 0, "source_id": util.SOURCE, "source_group": util.SOURCE, "role": util.ROLE, "group_id": row["group_id"],
            "physical_frames": proof["observed_actual_frames"], "physical_sample_rate_hz": proof["native_sample_rate_hz"],
            "physical_channels": 2, "sf_header_frames": proof["physical_header_frames"],
            "sf_actual_read_frames": proof["observed_actual_frames"], "crop_start_frame": proof["crop_start_frame"],
            "crop_frames": proof["crop_frames"], "crop_end_frame_exclusive": proof["crop_end_frame_exclusive"],
            "source_audio_sha256": row["acquisition_raw_sha256"], "native_crop_float64_sha256": proof["native_float64_sha256"],
            "native_crop_float32_sha256": proof["native_float32_sha256"], "center_seek_equals_sequential_float64": True,
            "classifier_admission_authorized": False, "evaluation_allowed": False}


def run(root, frozen_path, frozen_sha256, *, resume=False, _synthetic=None, _backend=None):
    """Require an independently frozen exact contract; continue fixed-row failures."""
    root, frozen_path = path_ok(root), path_ok(frozen_path)
    util.require(file_sha(frozen_path) == frozen_sha256, "Measurement freeze file pin mismatch")
    frozen = util.strict_json(util.read_bytes(frozen_path))
    util.require(frozen["status"] == "frozen" and frozen["authorized_stage"] == STAGE,
                 "New measurement freeze required; selection freeze cannot authorize run")
    review = frozen["independent_review"]
    util.require(review["approved"] is True and review["reviewer"] == "root", "Independent root review required")
    util.require(not Path(review["evidence"]).is_absolute() and ".." not in Path(review["evidence"]).parts, "Unsafe review evidence path")
    util.require(file_sha(root / review["evidence"]) == review["evidence_sha256"], "Root review evidence changed")
    contract = frozen["contract"]
    util.require(frozen["contract_sha256"] == util.seal(contract), "Measurement contract seal mismatch")
    output = path_ok(contract["output"])
    observed = build_contract(root, output, _synthetic=_synthetic, _backend=_backend)
    util.require(observed == contract, "Frozen contract/input/code/runtime mismatch")
    helper = _backend if _synthetic is not None and _backend is not None else get_backend()
    if output.exists():
        util.require(resume, "Output exists; explicit resume required")
        verify_publication(output, allow_extra=True)
        util.require(util.read_bytes(output / "frozen_measurement_contract.json") == util.read_bytes(frozen_path), "Resume freeze mismatch")
    else:
        util.require(not resume, "Resume output missing")
        publish_directory(output, {"frozen_measurement_contract.json": util.read_bytes(frozen_path)})
        for name in ("items", "failures", "reports"):
            (output / name).mkdir()
    lock_fd = os.open(path_ok(output / "writer.lock"), os.O_WRONLY | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    with os.fdopen(lock_fd, "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        util.require({p.name for p in output.iterdir()} <= {"frozen_measurement_contract.json", "COMMIT.json", "items", "failures", "reports", "writer.lock", "final"}, "Unexpected output root inventory")
        successes, failures = [], []
        expected = set(contract["selected_ids"])
        for name in ("items", "failures"):
            path_ok(output / name)
            util.require({p.name for p in (output / name).iterdir()} <= expected, "Unexpected prior item/failure inventory")
        for row in contract["rows"]:
            item_id = row["item_id"]
            failure_dir = output / "failures" / item_id
            if failure_dir.exists():
                verify_publication(failure_dir)
                prior = util.strict_json(util.read_bytes(failure_dir / "failure.json"))
                util.require(prior["contract_sha256"] == util.seal(contract) and prior["selection_row_sha256"] == util.seal(row), "Prior failure binding mismatch")
                failures.append(prior)
                continue
            try:
                successes.append(publish_item(root, output, row, contract, helper))
            except Exception as error:
                failure = {"status": "failed_no_replacement", "item_id": item_id, "contract_sha256": util.seal(contract),
                           "selection_row_sha256": util.seal(row), "error_type": type(error).__name__, "error": str(error),
                           "partial_interval_proof": getattr(error, "audit", {}), "classifier_admission": False}
                publish_directory(failure_dir, {"failure.json": util.canonical(failure)})
                failures.append(failure)
            print(f"{VERSION}: {len(successes) + len(failures)}/{len(contract['rows'])} accounted; {len(failures)} failures", flush=True)
        # Rehash every passed source and read back every committed output again
        # before final acceptance; do not infer batch-end freshness from old proofs.
        by_id = {row["item_id"]: row for row in contract["rows"]}
        final_successes = []
        for saved in successes:
            row = by_id[saved["item_id"]]
            try:
                final_successes.append(publish_item(root, output, row, contract, helper))
            except Exception as error:
                failure = {"status": "failed_no_replacement", "item_id": row["item_id"],
                           "contract_sha256": util.seal(contract), "selection_row_sha256": util.seal(row),
                           "error_type": type(error).__name__, "error": str(error), "failure_stage": "final_hash_recheck",
                           "partial_interval_proof": saved["interval_proof"], "classifier_admission": False}
                publish_directory(output / "failures" / row["item_id"], {"failure.json": util.canonical(failure)})
                failures.append(failure)
        successes = final_successes
        # No input/code/runtime changes may be hidden by a long measurement run.
        util.require(build_contract(root, output, _synthetic=_synthetic, _backend=_backend) == contract, "Bindings changed during measurement")
        summary = {"status": "passed_all_interval_materialization_not_classifier_admission" if not failures else "failed_incomplete_no_replacement",
                   "contract_sha256": util.seal(contract), "expected": len(contract["rows"]), "passed": len(successes), "failed": len(failures),
                   "failed_ids": [x["item_id"] for x in failures], "classifier_admission": False,
                   "final_passed_source_and_output_hash_recheck": True,
                   "neural_inference_performed": False, "classifier_fitted": False, "scoring_performed": False,
                   "synthetic_test_only": _synthetic is not None}
        files = {"materialization_summary.json": util.canonical(summary)}
        if not failures:
            util.require(len(successes) == len(contract["rows"]), "All selected must pass")
            native = [native_row(row, saved) for row, saved in zip(contract["rows"], successes)]
            manifest = [{"item_id": row["item_id"], "id": row["item_id"], "label": 0, "source_id": util.SOURCE,
                         "source_group": util.SOURCE, "group_id": row["group_id"], "role": util.ROLE,
                         "standardized_path": str(output / "items" / row["item_id"] / "audio.wav"),
                         "audio_path": str(output / "items" / row["item_id"] / "audio.wav"),
                         "audio_offset_s": 0, "requires_crop": 0, "duration": 60, "duration_sec": 60,
                         "native_sample_rate_hz": row["native_sample_rate_hz"], "native_sr": row["native_sample_rate_hz"],
                         "original_source_audio_path": row["source_audio_path"],
                         "original_source_audio_sha256": row["acquisition_raw_sha256"],
                         "standardized_sr": 44100, "standardized_frames": 2646000, "standardized_channels": 2,
                         "standardized_file_sha256": saved["wav"]["file_sha256"], "evaluation_allowed": False,
                         "classifier_admission_authorized": False} for row, saved in zip(contract["rows"], successes)]
            files["native_metadata_60s.csv"] = csv_bytes(native)
            files["inference_manifest.csv"] = csv_bytes(manifest)
            item_proofs = {saved["item_id"]: file_sha(output / "items" / saved["item_id"] / "proof.json") for saved in successes}
            files["item_proof_inventory.json"] = util.canonical(item_proofs)
            files["publication_receipt.json"] = util.canonical({"contract_sha256": util.seal(contract),
                 "outputs_sha256": {k: util.sha(v) for k, v in files.items()}, "item_proofs_sha256": util.seal(item_proofs)})
            final = output / "final"
            if final.exists():
                verify_publication(final)
                util.require(set(p.name for p in final.iterdir()) == set(files) | {"COMMIT.json"} and all(util.read_bytes(final / k) == v for k, v in files.items()), "Existing final publication conflict")
            else:
                publish_directory(final, files)
        else:
            report = output / "reports" / util.seal(summary)
            if not report.exists():
                publish_directory(report, files)
        return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="stage", required=True)
    draft = subs.add_parser("draft")
    draft.add_argument("--root", type=Path, required=True)
    draft.add_argument("--output", type=Path, required=True)
    draft.add_argument("--draft-dir", type=Path, required=True)
    launch = subs.add_parser("run")
    launch.add_argument("--root", type=Path, required=True)
    launch.add_argument("--frozen", type=Path, required=True)
    launch.add_argument("--frozen-sha256", required=True)
    launch.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.stage == "draft":
        result = create_draft(args.root, args.output, args.draft_dir)
    else:
        result = run(args.root, args.frozen, args.frozen_sha256, resume=args.resume)
    print(util.canonical(result).decode(), end="")
    return 1 if result.get("failed", 0) else 0


if __name__ == "__main__":
    raise SystemExit(main())
