"""Tests for the PopManga / CoMix / local panel detection benchmark harness."""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest
from PIL import Image
from typer.testing import CliRunner

from manga2kindle.cli import app
from manga2kindle.cv.backends.base import DetectionResult
from manga2kindle.cv.benchmark import (
    DEFAULT_COMIX_DATASET_ID,
    DEFAULT_POPMANGA_DATASET_ID,
    BenchmarkDatasetKind,
    BenchmarkMetrics,
    BenchmarkSample,
    evaluate_predictions,
    load_comix_split,
    load_local_dataset,
    load_popmanga_split,
    run_benchmark,
)
from manga2kindle.models import DetectorKind, PanelBox, ReadingDirection

runner = CliRunner()


def _make_sample(page_id: str, boxes: list[PanelBox], tmp_path: Path) -> BenchmarkSample:
    img_path = tmp_path / f"{page_id}.png"
    if not img_path.exists():
        Image.new("RGB", (800, 1200), (255, 255, 255)).save(img_path, "PNG")
    return BenchmarkSample(page_id=page_id, image_path=img_path, ground_truth=boxes)


class TestEvaluatePredictions:
    """Unit tests for evaluate_predictions metric calculations."""

    def test_perfect_predictions_score_one(self, tmp_path: Path) -> None:
        gt = [
            PanelBox(x=420, y=40, w=340, h=500, ordinal=1),
            PanelBox(x=40, y=40, w=340, h=500, ordinal=2),
            PanelBox(x=40, y=580, w=720, h=560, ordinal=3),
        ]
        samples = [_make_sample("p1", gt, tmp_path)]
        preds = [[box.model_copy() for box in gt]]

        metrics = evaluate_predictions(
            samples,
            preds,
            backend=DetectorKind.CONTOUR,
            ms_per_page=12.5,
            peak_ram_mb=128.0,
            peak_vram_mb=0.0,
        )

        assert isinstance(metrics, BenchmarkMetrics)
        assert metrics.backend == DetectorKind.CONTOUR
        assert metrics.pages_evaluated == 1
        assert metrics.precision_50 == pytest.approx(1.0)
        assert metrics.recall_50 == pytest.approx(1.0)
        assert metrics.precision_75 == pytest.approx(1.0)
        assert metrics.recall_75 == pytest.approx(1.0)
        assert metrics.mean_iou == pytest.approx(1.0)
        assert metrics.nested_duplicate_rate == pytest.approx(0.0)
        assert metrics.exact_count_match == pytest.approx(1.0)
        assert metrics.pairwise_order_acc == pytest.approx(1.0)
        assert metrics.kendall_tau == pytest.approx(1.0)
        assert metrics.ms_per_page == pytest.approx(12.5)
        assert metrics.peak_ram_mb == pytest.approx(128.0)
        assert metrics.peak_vram_mb == pytest.approx(0.0)

    def test_iou_threshold_50_vs_75_distinction(self, tmp_path: Path) -> None:
        # GT: 100x100 at (0, 0) -> area 10_000
        # Pred: 100x100 at (20, 0) -> intersection 80x100 = 8_000, union = 12_000 -> IoU = 2/3 (~0.6667)
        gt = [PanelBox(x=0, y=0, w=100, h=100, ordinal=1)]
        pred = [PanelBox(x=20, y=0, w=100, h=100, ordinal=1)]
        samples = [_make_sample("p1", gt, tmp_path)]

        metrics = evaluate_predictions(samples, [pred], backend=DetectorKind.ONNX_YOLO)

        assert metrics.precision_50 == pytest.approx(1.0)
        assert metrics.recall_50 == pytest.approx(1.0)
        assert metrics.precision_75 == pytest.approx(0.0)
        assert metrics.recall_75 == pytest.approx(0.0)
        assert metrics.mean_iou == pytest.approx(2.0 / 3.0, abs=1e-4)

    def test_mismatched_counts_and_nested_duplicates(self, tmp_path: Path) -> None:
        # Page 1: 2 GT boxes; preds have 1 match, 1 nested duplicate inside it, and 1 FP -> 3 preds
        # Page 2: 1 GT box; preds have 1 exact match -> exact_count_match = 0.5
        gt1 = [
            PanelBox(x=20, y=20, w=400, h=400, ordinal=1),
            PanelBox(x=20, y=460, w=400, h=400, ordinal=2),
        ]
        pred1 = [
            PanelBox(x=20, y=20, w=400, h=400, ordinal=1),  # matches gt1[0]
            PanelBox(x=40, y=40, w=200, h=200, ordinal=2),  # 100% contained inside pred1[0]
            PanelBox(x=500, y=20, w=200, h=200, ordinal=3),  # false positive
        ]
        gt2 = [PanelBox(x=50, y=50, w=300, h=300, ordinal=1)]
        pred2 = [PanelBox(x=50, y=50, w=300, h=300, ordinal=1)]

        samples = [
            _make_sample("p1", gt1, tmp_path),
            _make_sample("p2", gt2, tmp_path),
        ]
        metrics = evaluate_predictions(samples, [pred1, pred2], backend=DetectorKind.CONTOUR)

        # Total TP@50 = 2 (1 on p1, 1 on p2); total preds = 4; total GT = 3
        assert metrics.precision_50 == pytest.approx(2.0 / 4.0)
        assert metrics.recall_50 == pytest.approx(2.0 / 3.0)
        # 1 nested duplicate out of 4 total predicted boxes
        assert metrics.nested_duplicate_rate == pytest.approx(1.0 / 4.0)
        assert metrics.exact_count_match == pytest.approx(0.5)

    def test_reversed_and_partial_reading_order(self, tmp_path: Path) -> None:
        gt = [
            PanelBox(x=420, y=40, w=300, h=300, ordinal=1),
            PanelBox(x=40, y=40, w=300, h=300, ordinal=2),
            PanelBox(x=40, y=400, w=680, h=300, ordinal=3),
        ]
        # Completely reversed order: [3, 2, 1] -> 0/3 concordant pairs, tau = -1.0
        reversed_preds = [
            PanelBox(x=420, y=40, w=300, h=300, ordinal=3),
            PanelBox(x=40, y=40, w=300, h=300, ordinal=2),
            PanelBox(x=40, y=400, w=680, h=300, ordinal=1),
        ]
        m_rev = evaluate_predictions(
            [_make_sample("p1", gt, tmp_path)],
            [reversed_preds],
            backend=DetectorKind.CONTOUR,
        )
        assert m_rev.pairwise_order_acc == pytest.approx(0.0)
        assert m_rev.kendall_tau == pytest.approx(-1.0)

        # One swap [2, 1, 3] -> pairs (2,1) discordant, (2,3) concordant, (1,3) concordant -> 2/3, tau = 1/3
        partial_preds = [
            PanelBox(x=420, y=40, w=300, h=300, ordinal=2),
            PanelBox(x=40, y=40, w=300, h=300, ordinal=1),
            PanelBox(x=40, y=400, w=680, h=300, ordinal=3),
        ]
        m_part = evaluate_predictions(
            [_make_sample("p2", gt, tmp_path)],
            [partial_preds],
            backend=DetectorKind.CONTOUR,
        )
        assert m_part.pairwise_order_acc == pytest.approx(2.0 / 3.0)
        assert m_part.kendall_tau == pytest.approx(1.0 / 3.0)

    def test_empty_samples_returns_zeros(self) -> None:
        metrics = evaluate_predictions([], [], backend=DetectorKind.CONTOUR)
        assert metrics.pages_evaluated == 0
        assert metrics.precision_50 == 0.0
        assert metrics.kendall_tau == 0.0

    def test_mismatched_lengths_raises(self, tmp_path: Path) -> None:
        samples = [_make_sample("p1", [], tmp_path)]
        with pytest.raises(ValueError, match="length"):
            evaluate_predictions(samples, [], backend=DetectorKind.CONTOUR)


class TestLocalDatasetLoader:
    """Tests for load_local_dataset supporting custom JSON and COCO JSON."""

    def test_load_custom_json_format(self, tmp_path: Path) -> None:
        pages_dir = tmp_path / "pages"
        pages_dir.mkdir()
        Image.new("RGB", (600, 900), (255, 255, 255)).save(pages_dir / "001.png")
        Image.new("RGB", (600, 900), (255, 255, 255)).save(pages_dir / "002.png")

        truth_file = tmp_path / "truth.json"
        truth_file.write_text(
            json.dumps(
                {
                    "pages": [
                        {
                            "page_id": "001",
                            "file_name": "001.png",
                            "panels": [
                                {"x": 300, "y": 20, "w": 260, "h": 400, "ordinal": 1},
                                {"x": 20, "y": 20, "w": 260, "h": 400, "ordinal": 2},
                            ],
                        },
                        {
                            "page_id": "002",
                            "file_name": "002.png",
                            "panels": [[20, 20, 560, 860]],
                        },
                    ]
                }
            ),
            encoding="utf-8",
        )

        samples = load_local_dataset(pages_dir, truth_file)
        assert len(samples) == 2
        assert samples[0].page_id == "001"
        assert samples[0].image_path == pages_dir / "001.png"
        assert len(samples[0].ground_truth) == 2
        assert samples[0].ground_truth[0].ordinal == 1
        assert samples[1].ground_truth[0] == PanelBox(x=20, y=20, w=560, h=860, ordinal=1)

    def test_load_coco_json_format(self, tmp_path: Path) -> None:
        pages_dir = tmp_path / "coco_pages"
        pages_dir.mkdir()
        Image.new("RGB", (800, 1200), (255, 255, 255)).save(pages_dir / "page_a.png")

        coco_file = tmp_path / "coco.json"
        coco_file.write_text(
            json.dumps(
                {
                    "images": [{"id": 10, "file_name": "page_a.png", "width": 800, "height": 1200}],
                    "categories": [
                        {"id": 1, "name": "panel"},
                        {"id": 2, "name": "text"},
                    ],
                    "annotations": [
                        {"id": 1, "image_id": 10, "category_id": 1, "bbox": [400, 50, 350, 500], "order": 1},
                        {"id": 2, "image_id": 10, "category_id": 2, "bbox": [420, 70, 80, 40]},
                        {"id": 3, "image_id": 10, "category_id": 1, "bbox": [50, 50, 330, 500], "order": 2},
                    ],
                }
            ),
            encoding="utf-8",
        )

        samples = load_local_dataset(pages_dir, coco_file)
        assert len(samples) == 1
        assert samples[0].image_path == pages_dir / "page_a.png"
        # Should filter out category_id=2 ("text") and keep only the 2 panels
        assert len(samples[0].ground_truth) == 2
        assert samples[0].ground_truth[0] == PanelBox(x=400, y=50, w=350, h=500, ordinal=1)
        assert samples[0].ground_truth[1] == PanelBox(x=50, y=50, w=330, h=500, ordinal=2)

    def test_missing_truth_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_local_dataset(tmp_path, tmp_path / "missing.json")

    def test_coco_none_ordinals_do_not_crash(self, tmp_path: Path) -> None:
        pages_dir = tmp_path / "pages"
        pages_dir.mkdir()
        Image.new("RGB", (800, 1200), (255, 255, 255)).save(pages_dir / "page_a.png", "PNG")

        coco = {
            "images": [{"id": 1, "file_name": "page_a.png", "width": 800, "height": 1200}],
            "categories": [{"id": 1, "name": "panel"}],
            "annotations": [
                {"id": 10, "image_id": 1, "category_id": 1, "bbox": [50, 50, 300, 400], "ordinal": None},
                {"id": 11, "image_id": 1, "category_id": 1, "bbox": [400, 50, 300, 400], "order": None, "reading_order": None},
            ],
        }
        truth = tmp_path / "coco_none.json"
        truth.write_text(json.dumps(coco), encoding="utf-8")

        samples = load_local_dataset(pages_dir, truth)
        assert len(samples) == 1
        assert [p.ordinal for p in samples[0].ground_truth] == [1, 2]

    def test_missing_file_name_and_image_raises_value_error(self, tmp_path: Path) -> None:
        pages_dir = tmp_path / "pages"
        pages_dir.mkdir()
        truth = tmp_path / "missing_key.json"
        truth.write_text(json.dumps([{"panels": [[10, 10, 100, 100]]}]), encoding="utf-8")

        with pytest.raises(ValueError, match="missing 'file_name' or 'image'"):
            load_local_dataset(pages_dir, truth)


class TestHfDatasetLoaders:
    """Tests for PopManga and CoMix loaders with mocked `datasets` library."""

    def test_default_dataset_ids_match_spec(self) -> None:
        assert DEFAULT_POPMANGA_DATASET_ID == "ragavsachdeva/popmanga_test"
        assert DEFAULT_COMIX_DATASET_ID == "emanuelevivoli/comix-v0_1-pages"

    def test_load_popmanga_split_and_comix_split(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        rows = [
            {
                "image": Image.new("RGB", (600, 900), (255, 255, 255)),
                "panels": [[300, 20, 250, 400], [20, 20, 250, 400]],
                "id": "pop_001",
            },
            {
                "image": Image.new("RGB", (600, 900), (240, 240, 240)),
                "panels": [[20, 20, 560, 860]],
                "id": "pop_002",
            },
        ]
        fake_datasets = types.ModuleType("datasets")
        fake_datasets.load_dataset = lambda *args, **kwargs: rows  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "datasets", fake_datasets)

        pop_samples = load_popmanga_split("test", cache_dir=tmp_path / "pop", limit=1)
        assert len(pop_samples) == 1
        assert pop_samples[0].page_id == "pop_001"
        assert pop_samples[0].image_path.is_file()
        assert len(pop_samples[0].ground_truth) == 2

        comix_samples = load_comix_split("test", cache_dir=tmp_path / "comix", limit=2)
        assert len(comix_samples) == 2
        assert comix_samples[1].page_id == "pop_002"


class TestRunBenchmarkAndCli:
    """Tests for run_benchmark and `manga2kindle benchmark` CLI command."""

    def test_run_benchmark_excludes_warmup_and_respects_ordered_flag(self, tmp_path: Path) -> None:
        gt = [
            PanelBox(x=420, y=40, w=340, h=500, ordinal=1),
            PanelBox(x=40, y=40, w=340, h=500, ordinal=2),
        ]
        samples = [_make_sample(f"p{i}", gt, tmp_path) for i in range(2)]

        class _UnorderedBackend:
            kind = DetectorKind.ONNX_YOLO

            def __init__(self) -> None:
                self.calls = 0

            def detect(self, image_path: Path) -> DetectionResult:
                self.calls += 1
                # Return panels in reverse order with ordered=False so PanelSorter(RTL) orders them
                return DetectionResult(
                    panels=[
                        PanelBox(x=40, y=40, w=340, h=500, ordinal=0),
                        PanelBox(x=420, y=40, w=340, h=500, ordinal=0),
                    ],
                    ordered=False,
                    backend=DetectorKind.ONNX_YOLO,
                    page_width=800,
                    page_height=1200,
                )

            def detect_batch(self, image_paths: list[Path]) -> list[DetectionResult]:
                return [self.detect(p) for p in image_paths]

        backend = _UnorderedBackend()
        metrics = run_benchmark(
            samples,
            backend,
            direction=ReadingDirection.RTL,
            warmup=1,
        )

        assert backend.calls == 3  # 1 warmup + 2 evaluated pages
        assert metrics.pages_evaluated == 2
        assert metrics.precision_50 == pytest.approx(1.0)
        assert metrics.pairwise_order_acc == pytest.approx(1.0)
        assert metrics.ms_per_page >= 0.0

    def test_cli_benchmark_local_dataset_writes_report(self, tmp_path: Path) -> None:
        pages_dir = tmp_path / "pages"
        pages_dir.mkdir()
        img = Image.new("RGB", (800, 1200), (255, 255, 255))
        from PIL import ImageDraw

        draw = ImageDraw.Draw(img)
        draw.rectangle([420, 60, 740, 560], fill=(230, 230, 230), outline=(0, 0, 0), width=8)
        draw.rectangle([60, 620, 740, 1140], fill=(210, 210, 210), outline=(0, 0, 0), width=8)
        img.save(pages_dir / "page_001.png")

        truth_file = tmp_path / "truth.json"
        truth_file.write_text(
            json.dumps(
                {
                    "pages": [
                        {
                            "page_id": "page_001",
                            "file_name": "page_001.png",
                            "panels": [
                                {"x": 420, "y": 60, "w": 320, "h": 500, "ordinal": 1},
                                {"x": 60, "y": 620, "w": 680, "h": 520, "ordinal": 2},
                            ],
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        out_json = tmp_path / "report.json"

        result = runner.invoke(
            app,
            [
                "benchmark",
                "--dataset",
                BenchmarkDatasetKind.LOCAL.value,
                "--pages-dir",
                str(pages_dir),
                "--truth",
                str(truth_file),
                "--backends",
                "contour",
                "--out",
                str(out_json),
            ],
        )

        assert result.exit_code == 0, result.output
        assert "contour" in result.output.lower()
        assert out_json.is_file()
        payload = json.loads(out_json.read_text(encoding="utf-8"))
        assert payload["dataset"] == "local"
        assert len(payload["results"]) == 1
        assert payload["results"][0]["backend"] == "contour"
        assert payload["results"][0]["pages_evaluated"] == 1

    def test_cli_benchmark_local_requires_pages_dir_and_truth(self) -> None:
        result = runner.invoke(app, ["benchmark", "--dataset", "local"])
        assert result.exit_code != 0

    def test_cli_benchmark_forces_merge_bridged_panels_false(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        pages_dir = tmp_path / "pages"
        pages_dir.mkdir()
        Image.new("RGB", (800, 1200), (255, 255, 255)).save(pages_dir / "page_001.png")
        truth_file = tmp_path / "truth.json"
        truth_file.write_text(
            json.dumps(
                {
                    "pages": [
                        {
                            "page_id": "page_001",
                            "file_name": "page_001.png",
                            "panels": [{"x": 50, "y": 50, "w": 300, "h": 400, "ordinal": 1}],
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )

        seen_configs: list[object] = []
        from manga2kindle.cv.backends import resolve_auto_chain as real_resolve

        def spy_resolve(cfg):
            seen_configs.append(cfg)
            return real_resolve(cfg)

        monkeypatch.setattr("manga2kindle.cli.resolve_auto_chain", spy_resolve)

        result = runner.invoke(
            app,
            [
                "benchmark",
                "--dataset",
                BenchmarkDatasetKind.LOCAL.value,
                "--pages-dir",
                str(pages_dir),
                "--truth",
                str(truth_file),
                "--backends",
                "contour",
            ],
        )
        assert result.exit_code == 0, result.output
        assert len(seen_configs) == 1
        assert seen_configs[0].merge_bridged_panels is False
