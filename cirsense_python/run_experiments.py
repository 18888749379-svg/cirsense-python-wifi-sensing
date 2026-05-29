import argparse
import json
import subprocess
import sys
from pathlib import Path

from config import build_default_config
from learning_experiments import ALL_METHODS, DEFAULT_METHODS, run_learning_suite


CORE_RUNS = (
    ("bpm", None, "core_bpm"),
    ("distance", None, "core_distance"),
    ("bpm", "nlos_breathe", "core_nlos_bpm"),
    ("distance", "multitarget_distance", "core_multitarget_distance"),
)


def run_suite(
    suite: str,
    run_name: str,
    resume: bool = False,
    core_max_files: int = 3,
    learning_max_files: int = 6,
    epochs: int | None = None,
    methods: tuple[str, ...] = DEFAULT_METHODS,
    save_preprocessed: bool = False,
    use_preprocessed: bool = True,
    rebuild_dataset: bool = False,
    physics_run_name: str | None = None,
    distance_class_mode: str = "coarse",
    augment_bpm_windows: bool = False,
    bpm_signal_mode: str = "legacy",
) -> Path:
    config = build_default_config()
    smoke = suite in ("smoke", "compare_smoke")
    run_root = config.data.test_output_root if smoke else config.data.output_root
    suite_dir = run_root / run_name
    suite_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "suite": suite,
        "run_name": run_name,
        "run_root": str(run_root),
        "core_runs": [],
        "learning_run": None,
        "options": {
            "resume": resume,
            "core_max_files": core_max_files if smoke else None,
            "learning_max_files": learning_max_files if smoke else None,
            "epochs": epochs,
            "methods": list(methods),
            "save_preprocessed": save_preprocessed,
            "use_preprocessed": use_preprocessed,
            "rebuild_dataset": rebuild_dataset,
            "physics_run_name": physics_run_name,
            "distance_class_mode": distance_class_mode,
            "augment_bpm_windows": augment_bpm_windows,
            "bpm_signal_mode": bpm_signal_mode,
        },
    }

    if suite in ("smoke", "core_full", "full"):
        for task, subset, suffix in CORE_RUNS:
            child_run_name = f"{run_name}_{suffix}"
            command = [
                sys.executable,
                str(Path(__file__).resolve().parent / "main.py"),
                "--task",
                task,
                "--run-name",
                child_run_name,
                "--save-plots",
                "--plot-limit",
                "10" if smoke else "20",
            ]
            if subset is not None:
                command.extend(["--subset", subset])
            if smoke:
                command.extend(["--max-files", str(core_max_files)])
            if resume:
                command.append("--resume")
            if save_preprocessed:
                command.append("--save-preprocessed")
            if use_preprocessed:
                command.append("--use-preprocessed")
            _run_command(command)
            summary["core_runs"].append(
                {
                    "task": task,
                    "subset": subset or ("breathe" if task == "bpm" else "distance"),
                    "run_name": child_run_name,
                }
            )

    if suite in ("smoke", "compare_smoke", "full"):
        learning_epochs = epochs if epochs is not None else (2 if smoke else None)
        learning_run = run_learning_suite(
            f"{run_name}_comparison",
            tasks=("bpm", "distance"),
            methods=methods,
            max_files=learning_max_files if smoke else None,
            test_run=smoke,
            epochs=learning_epochs,
            save_preprocessed=save_preprocessed,
            use_preprocessed=use_preprocessed,
            rebuild_dataset=rebuild_dataset,
            physics_run_name=physics_run_name or (run_name if suite == "full" else "full_experiment_final"),
            distance_class_mode=distance_class_mode,
            augment_bpm_windows=augment_bpm_windows,
            bpm_signal_mode=bpm_signal_mode,
        )
        summary["learning_run"] = str(learning_run)

    summary_path = suite_dir / "suite_summary.json"
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
    print(f"Saved suite summary to: {summary_path}", flush=True)
    return suite_dir


def _run_command(command: list[str]) -> None:
    print("Running:", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Run CIRSense core and learning experiment suites.")
    parser.add_argument("--suite", choices=["smoke", "core_full", "compare_smoke", "full"], default="smoke")
    parser.add_argument("--run-name", default="cirsense_suite")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--core-max-files", type=int, default=3)
    parser.add_argument("--learning-max-files", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--methods", nargs="+", choices=list(ALL_METHODS), default=list(DEFAULT_METHODS))
    parser.add_argument("--save-preprocessed", action="store_true")
    parser.add_argument("--no-use-preprocessed", action="store_true")
    parser.add_argument("--rebuild-dataset", action="store_true")
    parser.add_argument(
        "--physics-run-name",
        default=None,
        help="Base run name for core physics CSVs used by residual/fusion methods.",
    )
    parser.add_argument("--distance-class-mode", choices=["coarse", "strict"], default="coarse")
    parser.add_argument("--bpm-window-augment", action="store_true")
    parser.add_argument(
        "--bpm-signal-mode",
        choices=["legacy", "quality"],
        default="legacy",
        help="Use the original BPM dynamic-path waveform or the newer spectrum-quality candidate search.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_suite(
        args.suite,
        args.run_name,
        resume=args.resume,
        core_max_files=args.core_max_files,
        learning_max_files=args.learning_max_files,
        epochs=args.epochs,
        methods=tuple(args.methods),
        save_preprocessed=args.save_preprocessed,
        use_preprocessed=not args.no_use_preprocessed,
        rebuild_dataset=args.rebuild_dataset,
        physics_run_name=args.physics_run_name,
        distance_class_mode=args.distance_class_mode,
        augment_bpm_windows=args.bpm_window_augment,
        bpm_signal_mode=args.bpm_signal_mode,
    )


if __name__ == "__main__":
    main()
