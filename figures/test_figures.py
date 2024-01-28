"""Offline checks for the frozen manuscript plotting inputs and conventions."""

import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

import gap_penalty
import hec
import retrieval
import roc
import scatter
from compute_heatmap_matrices import DATA as HEATMAP_DATA, residue_states
from _common import DATA, output_options, plt, save_figure


class FigureOutputTests(unittest.TestCase):
    def test_default_pdf_only(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "output"
            args = output_options("test", ["--output-dir", str(output)])
            self.assertEqual(args.format, "pdf")
            self.assertTrue(args.output_dir.is_dir())
            fig, ax = plt.subplots(figsize=(1, 1))
            ax.plot([0, 1])
            save_figure(fig, args.output_dir, "test")
            self.assertEqual({path.name for path in output.iterdir()}, {"test.pdf"})
            self.assertTrue((output / "test.pdf").read_bytes().startswith(b"%PDF-"))
            self.assertFalse(plt.fignum_exists(fig.number))

    def test_explicit_png_only(self):
        with tempfile.TemporaryDirectory() as directory:
            args = output_options("test", ["--output-dir", directory, "--format", "png"])
            fig, ax = plt.subplots(figsize=(1, 1))
            ax.plot([0, 1])
            save_figure(fig, args.output_dir, "test", args.format)
            self.assertEqual({path.name for path in args.output_dir.iterdir()}, {"test.png"})
            self.assertTrue((args.output_dir / "test.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n"))
            self.assertFalse(plt.fignum_exists(fig.number))

    def test_unsupported_format_rejected(self):
        with self.assertRaises(ValueError):
            save_figure(None, Path("."), "test", "svg")


class FrozenDataTests(unittest.TestCase):
    def test_results_directory(self):
        self.assertEqual(DATA.name, "results")
        self.assertEqual(HEATMAP_DATA, DATA)
        self.assertTrue(DATA.is_dir())
        self.assertFalse((DATA.parent / "data").exists())

    def test_result_manifest(self):
        repository = DATA.parent.parent
        manifest = json.loads((DATA / "manifest.json").read_text())
        self.assertEqual(len(manifest), 34)
        paths = {item["path"] for item in manifest}
        self.assertEqual(len(paths), len(manifest))
        references = {path for path in paths if path.startswith("data/")}
        self.assertEqual(references, {"data/malidup_hec.csv", "data/malisam_hec.csv",
                                      "data/scope40-test.fasta"})
        for item in manifest:
            with self.subTest(path=item["path"]):
                path = (repository / item["path"]).resolve()
                self.assertTrue(path.is_relative_to(repository.resolve()))
                content = path.read_bytes()
                self.assertEqual(len(content), item["bytes"])
                self.assertEqual(hashlib.sha256(content).hexdigest(), item["sha256"])

    def test_metric_tables_match_scatter_inputs(self):
        for dataset, count in [("malidup", 241), ("malisam", 130)]:
            with (DATA / f"{dataset}_tm_scores.csv").open(newline="") as handle:
                frozen = list(csv.DictReader(handle))
            tables = list((DATA / dataset).glob("*.txt"))
            self.assertEqual(len(tables), 12)
            for path in tables:
                with self.subTest(dataset=dataset, method=path.stem):
                    rows = [line.split() for line in path.read_text().splitlines()]
                    actual = {row[0]: float(row[4]) for row in rows}
                    expected = {row["pair_id"]: float(row[path.stem]) for row in frozen}
                    self.assertEqual(len(rows), count)
                    self.assertEqual(actual, expected)

    def test_t5_residue_extraction_keeps_first_and_excludes_eos(self):
        hidden = np.array([[10, 11], [20, 21], [30, 31], [999, 999]])
        np.testing.assert_array_equal(residue_states(hidden, 3), hidden[:3])
        with self.assertRaises(ValueError):
            residue_states(hidden, 4)

    def test_data_checksums(self):
        provenance = json.loads((DATA / "provenance.json").read_text())
        for item in provenance["outputs"]:
            with self.subTest(path=item["path"]):
                digest = hashlib.sha256((DATA / item["path"]).read_bytes()).hexdigest()
                self.assertEqual(digest, item["sha256"])

    def test_scatter_counts_and_means(self):
        for dataset, count, mean in [("malidup", 241, 0.496474), ("malisam", 130, 0.280015)]:
            values = scatter.load_scores(dataset)
            self.assertTrue(all(len(column) == count for column in values.values()))
            self.assertAlmostEqual(values["clalign-prott5"].mean(), mean, places=6)
        values = scatter.load_scores("malidup")["dali"]
        self.assertEqual(np.count_nonzero(values == 0), 7)
        self.assertAlmostEqual(values.mean(), 0.574891, places=6)
        self.assertLess(values.mean(), values[values > 0].mean())
        self.assertEqual(np.count_nonzero(scatter.load_scores("malisam")["dali"] == 0), 0)

    def test_fixed_values(self):
        self.assertEqual(hec.RESULTS[("malisam", "SOV")][0],
                         [0.361, 0.450, 0.381, 0.467, 0.472, 0.560])
        self.assertEqual(retrieval.RESULTS["MAP"],
                         [0.746, 0.757, 0.750, 0.689, 0.876, 0.838, 0.844])
        self.assertEqual(gap_penalty.RESULTS["Validation"],
                         [0.342, 0.335, 0.326, 0.316, 0.306, 0.302])
        for values, upper in hec.RESULTS.values():
            self.assertEqual(len(values), len(hec.METHODS))
            self.assertTrue(all(0 <= value <= upper for value in values))
        for values in retrieval.RESULTS.values():
            self.assertEqual(len(values), len(retrieval.METHODS))
            self.assertTrue(all(0 <= value <= 1 for value in values))

    def test_hec_aggregation_population(self):
        data = json.loads((DATA / "hec_aggregation.json").read_text())
        methods = ["plmblast", "plmalign", "eba", "deepblast", "tea-nw", "clalign-prott5"]
        self.assertEqual(data["malidup"]["valid_pairs"], 240)
        self.assertEqual(data["malisam"]["valid_pairs"], 129)
        for (dataset, metric), (values, _) in hec.RESULTS.items():
            expected = [round(data[dataset]["valid_pair_means"][method][metric], 3) for method in methods]
            self.assertEqual(values, expected)

    def test_curve_population_and_sampling(self):
        identifiers = None
        for method, _, _ in roc.METHODS:
            with (DATA / "scope40201" / f"{method}.rocx").open(newline="") as handle:
                actual = {row["NAME"] for row in csv.DictReader(handle, delimiter="\t")}
            if identifiers is None:
                identifiers = actual
            self.assertEqual(actual, identifiers)
            for values in roc.load_sensitivities(method).values():
                x, y = roc.curve_coordinates(values)
                self.assertEqual(len(x), 39)
                np.testing.assert_allclose(x, np.arange(1, 767, 20) / 766)
                self.assertTrue(np.all(np.diff(y) <= 0))
                self.assertTrue(np.all((y >= 0) & (y <= 1)))

    def test_heatmap_arrays_and_normalization(self):
        case = json.loads((DATA / "heatmap_case.json").read_text())
        metadata = json.loads((DATA / "heatmap_metadata.json").read_text())
        self.assertEqual(case["pair_id"], metadata["pair_id"])
        shape = (len(case["sequence1"]), len(case["sequence2"]))
        self.assertEqual(shape, (65, 71))
        self.assertEqual(list(shape), metadata["shape"])
        digest = hashlib.sha256((DATA / "heatmap_matrices.npz").read_bytes()).hexdigest()
        self.assertEqual(digest, metadata["matrices_sha256"])
        with np.load(DATA / "heatmap_matrices.npz", allow_pickle=False) as arrays:
            self.assertEqual(set(arrays.files), {"plmblast", "plmalign", "eba", "clalign"})
            for matrix in arrays.values():
                self.assertEqual(matrix.shape, shape)
                self.assertTrue(np.isfinite(matrix).all())
                self.assertGreater(float(matrix.std()), 0)
            for name in ["plmblast", "plmalign"]:
                self.assertAlmostEqual(float(arrays[name].mean()), 0, places=5)
                self.assertAlmostEqual(float(arrays[name].std(ddof=1)), 1, places=5)
            self.assertTrue(np.all(np.abs(arrays["clalign"]) <= 1.000001))
        self.assertEqual(metadata["eba_default"], "author-original (array key: eba)")


if __name__ == "__main__":
    unittest.main()
