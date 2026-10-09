"""GPU/offscreen rendering regression, including the failing legacy control."""
from pathlib import Path
import tempfile
import unittest
from scripts.m16_visual_check import check
from supervisor.feeding_loading.config import load_config, validate_config


class DuploVisualTests(unittest.TestCase):
    def test_four_rendered_orientations_and_unchanged_physics(self):
        with tempfile.TemporaryDirectory(prefix='duplo-render-') as temp:
            result=check(Path(temp)/'evidence',Path('configs/feeding_loading/feeding_duplo_v2.json'))
            self.assertTrue(result['passed'])

    def test_visual_version_is_explicit_and_validated(self):
        legacy=load_config()
        self.assertNotIn('visual_version',legacy['object'])
        legacy['object']['visual_version']='misspelled-version'
        with self.assertRaises(ValueError):validate_config(legacy)


if __name__=='__main__':unittest.main()
