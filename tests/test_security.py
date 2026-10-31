"""Synthetic security regressions; no imaging stack, weights, or clinical data required."""
import importlib
import hashlib
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
import types
import unittest
from unittest.mock import patch
import zipfile

ROOT = Path(__file__).resolve().parents[1]


class Progress:
    def __enter__(self): return self
    def __exit__(self, *args): pass
    def add_task(self, *args, **kwargs): return 1
    def update(self, *args, **kwargs): pass


class Output:
    def log_update(self, *args): pass
    def console_update(self, *args): pass
    def create_file_progress_bar(self): return Progress()


class Response:
    status_code = 200
    headers = {}
    def __init__(self, body): self.body = body
    def __enter__(self): return self
    def __exit__(self, *args): pass
    def raise_for_status(self): pass
    def iter_content(self, **kwargs): yield self.body


class ModelDownloadSecurity(unittest.TestCase):
    def setUp(self):
        # Only replace the heavyweight application entry point/UI dependency.
        package = types.ModuleType('moosez')
        package.__path__ = [str(ROOT / 'moosez')]
        system = types.ModuleType('moosez.system')
        system.OutputManager = Output
        system.MODELS_DIRECTORY_PATH = 'unused'
        self.modules = patch.dict(sys.modules, {'moosez': package, 'moosez.system': system})
        self.modules.start()
        self.addCleanup(self.modules.stop)
        self.models = importlib.import_module('moosez.models')
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.model = self.models.Model.__new__(self.models.Model)
        self.model.model_identifier = 'synthetic'
        self.model.folder_name = 'Dataset001_body'
        self.model.base_directory = str(self.base)
        self.model.directory = str(self.base / 'Dataset001_body')
        self.model.url = 'https://example.invalid/model.zip'
        self.model.sha256 = None

    def download(self, members, pin_checksum=False):
        archive = io.BytesIO()
        with zipfile.ZipFile(archive, 'w') as zf:
            for name, value in members:
                zf.writestr(name, value)
        if pin_checksum:
            self.model.sha256 = hashlib.sha256(archive.getvalue()).hexdigest()
        with patch('requests.get', return_value=Response(archive.getvalue())):
            self.model._Model__download(Output())

    def test_rejects_sibling_model_overwrite_before_extracting_any_member(self):
        victim = self.base / 'OtherModel' / 'checkpoint.pth'
        victim.parent.mkdir()
        victim.write_bytes(b'trusted')
        with self.assertRaises(ValueError):
            self.download([('Dataset001_body/config.json', '{}'), ('OtherModel/checkpoint.pth', b'hostile')])
        self.assertEqual(victim.read_bytes(), b'trusted')
        self.assertFalse(Path(self.model.directory).exists())

    def test_download_does_not_follow_predictable_archive_symlink(self):
        victim = self.base / 'victim'
        victim.write_bytes(b'trusted')
        (self.base / 'model.zip').symlink_to(victim)
        self.download([('Dataset001_body/config.json', '{}')])
        self.assertEqual(victim.read_bytes(), b'trusted')

    def test_rejects_traversal_and_windows_paths(self):
        for name in ['../Dataset001_body/file', '/Dataset001_body/file', 'Dataset001_body/../file',
                     'Dataset001_body\\file', 'C:/Dataset001_body/file']:
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.download([(name, 'hostile')])

    def test_rejects_archive_symlink(self):
        member = zipfile.ZipInfo('Dataset001_body/link')
        member.create_system = 3
        member.external_attr = 0o120777 << 16
        with self.assertRaises(ValueError):
            self.download([(member, '../outside')])

    def test_failed_update_preserves_installed_model(self):
        installed = Path(self.model.directory)
        installed.mkdir()
        (installed / 'checkpoint.pth').write_bytes(b'trusted')
        with self.assertRaises(ValueError):
            self.download([('OtherModel/checkpoint.pth', b'hostile')])
        self.assertEqual((installed / 'checkpoint.pth').read_bytes(), b'trusted')

    def test_enhance_download_replaces_symlink_without_touching_target(self):
        download = importlib.import_module('moosez.download')
        victim = self.base / 'victim'
        victim.write_bytes(b'trusted')
        destination = self.base / 'enhance.zip'
        destination.symlink_to(victim)
        with patch.dict(download.constants.ENHANCE_URLS, {'synthetic': 'https://example.invalid/data'}):
            with patch('requests.get', return_value=Response(b'synthetic')):
                download.download_enhance_item('synthetic', str(destination), Output())
        self.assertEqual(victim.read_bytes(), b'trusted')
        self.assertFalse(destination.is_symlink())
        self.assertEqual(destination.read_bytes(), b'synthetic')

    def test_checksum_mismatch_is_rejected_before_install(self):
        self.model.sha256 = '0' * 64
        with self.assertRaises(ValueError):
            self.download([('Dataset001_body/checkpoint.pth', b'tampered')])
        self.assertFalse(Path(self.model.directory).exists())

    def test_valid_model_installs_and_records_version(self):
        self.download([('Dataset001_body/configuration/fold_0/checkpoint.pth', b'synthetic')], pin_checksum=True)
        self.assertEqual((Path(self.model.directory) / 'configuration/fold_0/checkpoint.pth').read_bytes(), b'synthetic')
        self.assertTrue((Path(self.model.directory) / 'model_version.json').is_file())


class WorkflowSecurity(unittest.TestCase):
    def test_tag_payload_cannot_execute_before_validation(self):
        workflow = (ROOT / '.github/workflows/release.yml').read_text()
        block = workflow.split('      - name: Extract release metadata\n', 1)[1].split('\n  build-python:', 1)[0]
        script = textwrap.dedent(block.split('        run: |\n', 1)[1])
        with tempfile.TemporaryDirectory() as tmp:
            marker = Path(tmp) / 'executed'
            payload = '$(touch ' + str(marker) + ')'
            replacements = {'github.event_name': 'workflow_dispatch', 'github.event.release.tag_name': '',
                            'github.event.inputs.tag': payload}
            for expression, value in replacements.items():
                script = script.replace('${{ ' + expression + ' }}', value)
            env = dict(os.environ, RELEASE_TAG='', DISPATCH_TAG=payload,
                       GITHUB_EVENT_NAME='workflow_dispatch', GITHUB_OUTPUT=str(Path(tmp) / 'output'))
            result = subprocess.run(['bash', '-c', script], env=env, capture_output=True, timeout=5)
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse(marker.exists(), result.stderr.decode())

    def test_valid_release_and_dispatch_tags_produce_expected_metadata(self):
        workflow = (ROOT / '.github/workflows/release.yml').read_text()
        block = workflow.split('      - name: Extract release metadata\n', 1)[1].split('\n  build-python:', 1)[0]
        script = textwrap.dedent(block.split('        run: |\n', 1)[1])
        for event in ('release', 'workflow_dispatch'):
            with self.subTest(event=event), tempfile.TemporaryDirectory() as tmp:
                output = Path(tmp) / 'output'
                env = dict(os.environ, RELEASE_TAG='moosez-v.3.3.0', DISPATCH_TAG='moosez-v.3.3.0',
                           GITHUB_EVENT_NAME=event, GITHUB_OUTPUT=str(output))
                result = subprocess.run(['bash', '-c', script], env=env, capture_output=True, timeout=5)
                self.assertEqual(result.returncode, 0, result.stderr.decode())
                self.assertEqual(output.read_text(), 'tag=moosez-v.3.3.0\nversion=3.3.0\ndocker_version=v3.3.0\n')
