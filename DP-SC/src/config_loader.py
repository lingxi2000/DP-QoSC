from __future__ import annotations

import configparser
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union


CONFIG_FILENAME = "experiment.ini"
REFERENCE_FORMAT = "SLA-FullExactBoundary-v5"


def script_root() -> Path:
    return Path(__file__).resolve().parent


def config_path() -> Path:
    return script_root() / CONFIG_FILENAME


def _optional_int(
    parser: configparser.ConfigParser,
    section: str,
    option: str,
) -> Optional[int]:
    value = parser.get(section, option).strip()

    if value.lower() in {"none", "null", ""}:
        return None

    return int(value)


def _dataset_ga_section(
    parser: configparser.ConfigParser,
    dataset: Optional[str],
) -> Optional[str]:
    if dataset is None:
        return None

    target = f"GA_{str(dataset).strip()}".casefold()

    for section in parser.sections():
        if section.casefold() == target:
            return section

    raise ValueError(
        f"No dataset-specific GA section found for {dataset!r}. "
        f"Expected a section such as [GA_{dataset}]."
    )


@dataclass(frozen=True)
class ExperimentConfig:
    node_file: str
    service_file: str
    test_only: bool

    exact_max_tasks: Optional[int]
    exact_progress_every: int
    exact_save_every: int
    exact_auto_resume: bool

    ga_seed: int
    ga_population: int
    ga_max_generations: int
    ga_stagnation_patience: int
    ga_crossover_rate: float
    ga_mutation_probability: float
    ga_elite_count: int
    ga_hit_tolerance: float
    ga_max_exact_tasks: Optional[int]
    ga_mode: str

    exact_reference_dir: str
    result_dir: str
    ga_exact_result_suffix: str
    print_service_diagnostics: bool

    ga_dataset_section: Optional[str]

    def exact_reference_path(self, dataset: str) -> Path:
        dataset_safe = (
            str(dataset)
            .strip()
            .replace("/", "_")
            .replace("\\", "_")
        )
        node_stem = Path(self.node_file).stem
        filename = f"{dataset_safe}_{node_stem}_fullExactBoundary_v5.data"
        return script_root() / self.exact_reference_dir / filename

    def ga_exact_csv_path(self, dataset: str) -> Path:
        dataset_safe = (
            str(dataset)
            .strip()
            .replace("/", "_")
            .replace("\\", "_")
        )
        node_stem = Path(self.node_file).stem
        filename = (
            f"{dataset_safe}_{node_stem}"
            f"{self.ga_exact_result_suffix}"
        )
        return script_root() / self.result_dir / filename


def load_config(
    path: Optional[Union[str, Path]] = None,
    dataset: Optional[str] = None,
) -> ExperimentConfig:
    path = Path(path) if path is not None else config_path()

    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    parser = configparser.ConfigParser(
        interpolation=None,
        inline_comment_prefixes=(";", "#"),
    )
    parser.read(path, encoding="utf-8")

    required_sections = {"DATA", "EXACT", "GA", "OUTPUT"}
    missing = required_sections - set(parser.sections())

    if missing:
        raise ValueError(f"Missing INI sections: {sorted(missing)}")

    dataset_section = _dataset_ga_section(parser, dataset)
    population_section = dataset_section or "GA"
    generation_section = dataset_section or "GA"

    cfg = ExperimentConfig(
        node_file=parser.get("DATA", "node_file").strip(),
        service_file=parser.get("DATA", "service_file").strip(),
        test_only=parser.getboolean("DATA", "test_only"),
        exact_max_tasks=_optional_int(parser, "EXACT", "max_tasks"),
        exact_progress_every=parser.getint("EXACT", "progress_every"),
        exact_save_every=parser.getint("EXACT", "save_every"),
        exact_auto_resume=parser.getboolean("EXACT", "auto_resume"),
        ga_seed=parser.getint("GA", "seed"),
        ga_population=parser.getint(population_section, "population"),
        ga_max_generations=parser.getint(
            generation_section,
            "max_generations",
        ),
        ga_stagnation_patience=parser.getint(
            "GA",
            "stagnation_patience",
        ),
        ga_crossover_rate=parser.getfloat("GA", "crossover_rate"),
        ga_mutation_probability=parser.getfloat(
            "GA",
            "mutation_probability",
        ),
        ga_elite_count=parser.getint("GA", "elite_count"),
        ga_hit_tolerance=parser.getfloat("GA", "hit_tolerance"),
        ga_max_exact_tasks=_optional_int(
            parser,
            "GA",
            "max_exact_tasks",
        ),
        ga_mode=parser.get("GA", "mode").strip().lower(),
        exact_reference_dir=parser.get(
            "OUTPUT",
            "exact_reference_dir",
        ).strip(),
        result_dir=parser.get("OUTPUT", "result_dir").strip(),
        ga_exact_result_suffix=parser.get(
            "OUTPUT",
            "ga_exact_result_suffix",
        ).strip(),
        print_service_diagnostics=parser.getboolean(
            "OUTPUT",
            "print_service_diagnostics",
            fallback=False,
        ),
        ga_dataset_section=dataset_section,
    )

    validate_config(cfg)
    return cfg


def validate_config(cfg: ExperimentConfig) -> None:
    if not cfg.node_file:
        raise ValueError("[DATA] node_file cannot be empty.")

    if not cfg.service_file:
        raise ValueError("[DATA] service_file cannot be empty.")

    if cfg.exact_max_tasks is not None and cfg.exact_max_tasks <= 0:
        raise ValueError(
            "[EXACT] max_tasks must be positive or none."
        )

    if cfg.exact_progress_every <= 0:
        raise ValueError(
            "[EXACT] progress_every must be positive."
        )

    if cfg.exact_save_every <= 0:
        raise ValueError(
            "[EXACT] save_every must be positive."
        )

    if cfg.ga_population < 4:
        raise ValueError("GA population must be >= 4.")

    if cfg.ga_max_generations <= 0:
        raise ValueError(
            "GA max_generations must be positive."
        )

    if cfg.ga_stagnation_patience <= 0:
        raise ValueError(
            "[GA] stagnation_patience must be positive."
        )

    if cfg.ga_stagnation_patience > cfg.ga_max_generations:
        raise ValueError(
            "[GA] stagnation_patience cannot exceed "
            "the selected dataset max_generations."
        )

    if not 0.0 <= cfg.ga_crossover_rate <= 1.0:
        raise ValueError(
            "[GA] crossover_rate must be in [0, 1]."
        )

    if not 0.0 <= cfg.ga_mutation_probability <= 1.0:
        raise ValueError(
            "[GA] mutation_probability must be in [0, 1]."
        )

    if not 1 <= cfg.ga_elite_count < cfg.ga_population:
        raise ValueError(
            "[GA] elite_count must satisfy "
            "1 <= elite_count < selected dataset population."
        )

    if cfg.ga_hit_tolerance < 0.0:
        raise ValueError(
            "[GA] hit_tolerance must be >= 0."
        )

    if (
        cfg.ga_max_exact_tasks is not None
        and cfg.ga_max_exact_tasks <= 0
    ):
        raise ValueError(
            "[GA] max_exact_tasks must be positive or none."
        )

    if cfg.ga_mode not in {"exact_quality", "full_scale"}:
        raise ValueError(
            "[GA] mode must be exact_quality or full_scale."
        )