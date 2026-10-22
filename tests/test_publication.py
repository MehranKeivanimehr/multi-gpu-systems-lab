"""Portable public-release regressions; no GPU or model imports."""
import copy
import ast
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest import mock

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
import a6000_evidence as ev
import build_showcase
import render_figures


class Publication(unittest.TestCase):
    def test_python_sources_parse_under_the_portable_contract(self):
        for p in ROOT.rglob('*.py'):
            ast.parse(p.read_text(encoding='utf-8-sig'), filename=str(p), feature_version=(3,10))

    def test_control_replays_from_real_arrays(self):
        r=ev.replay()
        self.assertTrue(r['all_pass'])
        self.assertEqual(r['slice_elements'],787968)
        self.assertEqual(r['payload_bytes'],12607488)
        self.assertEqual(len(r['endpoints']),6)

    def test_matched_cap_and_exact_repeat_cohorts(self):
        rows=ev.inference()
        self.assertEqual([round(r['dp2_advantage_pct'],2) for r in rows],[33.23,14.99])
        for r in rows:
            self.assertEqual(r['aggregate_cap'],512)
            self.assertEqual(r['arms']['TP2']['source'],'rerun')
            self.assertEqual(r['arms']['DP2']['n'],2)

    def test_unmatched_original_cap_is_not_selected(self):
        rows=ev.csv_rows(ROOT/'studies/a6000/results/rerun/vllm_tp_concurrency_comparison.csv')
        no_match=[r for r in rows if not (r['source']=='rerun' and r['tp']=='2' and r['total_concurrency_cap']=='512')]
        with mock.patch.object(ev,'csv_rows',return_value=no_match):
            with self.assertRaises(ev.EvidenceError): ev.inference()

    def test_no_duplicate_repeat_is_accepted(self):
        rows=ev.csv_rows(ROOT/'studies/a6000/results/rerun/vllm_tp_concurrency_comparison.csv')
        same=copy.deepcopy(rows)
        arm=[r for r in same if r['model']=='qwen2.5-1.5b' and r['source']=='rerun' and r['tp']=='2' and r['total_concurrency_cap']=='512']
        arm[1]['run']=arm[0]['run']
        with mock.patch.object(ev,'csv_rows',return_value=same):
            with self.assertRaises(ev.EvidenceError): ev.inference()

    def test_titan_contexts_remain_qualified(self):
        s=build_showcase.summary()
        self.assertEqual(s['titan']['training_contexts'],16)
        self.assertEqual(s['titan']['status_counts'],{'completed_with_supported_checks':17,'completed_with_qualifications':2})

    def test_generated_values_are_current(self):
        for name,content in build_showcase.artifacts().items():
            self.assertEqual((ROOT/name).read_bytes(),content.encode(),name)

    def test_svg_figures_are_data_driven_and_current(self):
        s=json.loads((ROOT/'results/showcase.json').read_text())
        for name,fn in render_figures.FIGURES.items():
            self.assertEqual((ROOT/'assets/figures'/f'{name}.svg').read_text(),fn(s).svg())

    def test_limitations_are_the_final_readme_section(self):
        sections=[s for s in (ROOT/'README.md').read_text().splitlines() if s.startswith('## ')]
        self.assertEqual(sections[-1],'## Limitations')

    def test_author_is_single_and_no_workflow_success_is_invented(self):
        s=(ROOT/'CITATION.cff').read_text()
        self.assertEqual(s.count('given-names:'),1)
        self.assertIn('Mehran',s)
        p=json.loads((ROOT/'provenance/publication.json').read_text())
        self.assertFalse(p['new_gpu_runs']); self.assertFalse(p['github_ci_run'])

    def test_damaged_or_resized_array_refuses(self):
        src=ROOT/'results/a6000_validation/evidence/70_final_review/calibration_slices'
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); d=root/'results/a6000_validation/evidence/70_final_review/calibration_slices'
            d.parent.mkdir(parents=True); shutil.copytree(src,d)
            p=d/'C5_grad_b1_slices.f32'
            p.write_bytes(p.read_bytes()+b'\0\0\0\0')
            # Even an updated file hash must not conceal a wrong coverage size.
            m=json.loads((d/'SLICES_MANIFEST.json').read_text()); m['files'][p.name]['sha256']=hashlib.sha256(p.read_bytes()).hexdigest()
            (d/'SLICES_MANIFEST.json').write_text(json.dumps(m))
            with self.assertRaises(ev.EvidenceError): ev.replay(root)


if __name__=='__main__': unittest.main()
