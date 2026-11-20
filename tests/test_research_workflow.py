"""Exercise real command receipts, tamper detection, comparison and skipped gates."""
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from airbot_tt.research import run, compare, write_json

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('group_verify', ROOT/'scripts/verify.py')
verify=importlib.util.module_from_spec(spec)
spec.loader.exec_module(verify)


class ResearchTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        subprocess.run(['git','init','-q',str(self.root)],check=True)
        (self.root/'.gitignore').write_text('runs/\n')
        subprocess.run(['git','-C',str(self.root),'add','.gitignore'],check=True)
        subprocess.run(['git','-C',str(self.root),'-c','user.name=Test','-c','user.email=test@example.invalid',
                        'commit','-qm','Synthetic fixture'],check=True)
        self.protocol=self.root/'protocol.json'
        self.definition=dict(question='Synthetic fixture only',kind='experiment',common={'serves':'fixed'},method={'reward':'A'})
        write_json(self.protocol,self.definition)

    def trial(self,name='a',seed=1,code=None):
        result=dict(status='PASS',training_seed=seed,runtime={'python':sys.version},metrics={'score':.5})
        program=code or "import pathlib; pathlib.Path(r'{run_dir}/result.json').write_text("+repr(json.dumps(result))+")"
        directory=self.root/'runs'/name
        record=run(self.root,directory,self.protocol,[],[sys.executable,'-c',program],5)
        return directory,record

    def test_receipt_and_no_overwrite(self):
        path,record=self.trial()
        self.assertEqual(record['status'],'PASS')
        self.assertIn('protocol.json',record['source_sha256'])
        with self.assertRaises(FileExistsError): self.trial()
        result=compare([path],'score')
        self.assertEqual(result['scientific_conclusion'],'INCONCLUSIVE')
        self.assertIsNone(result['groups'][0]['sd'])

    def test_method_changes_allowed_common_changes_rejected(self):
        a,_=self.trial()
        self.definition['method']={'reward':'B'};write_json(self.protocol,self.definition)
        b,_=self.trial('b')
        self.assertEqual(len(compare([a,b],'score')['groups']),2)
        self.definition['common']={'serves':'different'};write_json(self.protocol,self.definition)
        c,_=self.trial('c')
        with self.assertRaisesRegex(ValueError,'common'): compare([a,c],'score')

    def test_duplicate_seed_and_tampering(self):
        a,_=self.trial();b,_=self.trial('b')
        with self.assertRaisesRegex(ValueError,'Duplicate'): compare([a,b],'score')
        (b/'result.json').write_text('{}')
        with self.assertRaisesRegex(ValueError,'artifact'): compare([b],'score')

    def test_failures_not_silently_excluded(self):
        a,_=self.trial();b,r=self.trial('b',code='raise SystemExit(3)')
        self.assertEqual(r['status'],'FAIL')
        report=compare([a,b],'score')
        self.assertEqual(report['status'],'INCOMPLETE')
        self.assertEqual(len(report['failed_or_incomplete']),1)

    def test_successful_process_without_result_is_incomplete(self):
        _,r=self.trial(code='pass')
        self.assertEqual(r['status'],'INCOMPLETE')

    def test_source_mutation_invalidates_receipt(self):
        _,r=self.trial(code="from pathlib import Path; Path('protocol.json').write_text('{}')")
        self.assertEqual(r['status'],'INCOMPLETE')
        self.assertIn('protocol.json',r['changed_sources'])

    def test_all_test_modules_are_assigned_once(self):
        assigned=[name for names in verify.GROUPS.values() for name in names]
        actual={p.stem for p in (ROOT/'tests').glob('test_*.py')}
        self.assertEqual(set(assigned),actual)
        self.assertEqual(len(assigned),len(set(assigned)))

    def test_timeout(self):
        r=run(self.root,self.root/'runs/timeout',self.protocol,[],
              [sys.executable,'-c','import time; time.sleep(30)'],.05)
        self.assertEqual(r['error'],'TIMEOUT')
        self.assertEqual(r['status'],'FAIL')

    def test_analysis_is_not_training_replication(self):
        self.definition['kind']='analysis';write_json(self.protocol,self.definition)
        a,_=self.trial()
        with self.assertRaisesRegex(ValueError,'independent'): compare([a],'score')

    def test_skips_and_empty_suite_are_incomplete(self):
        class Skip(unittest.TestCase):
            @unittest.skip('dependency unavailable')
            def test_missing(self): pass
        result=unittest.TextTestRunner(stream=io.StringIO()).run(unittest.defaultTestLoader.loadTestsFromTestCase(Skip))
        self.assertEqual(verify.status_for(result),('INCOMPLETE',2))
        self.assertEqual(verify.status_for(unittest.TestResult()),('INCOMPLETE',2))


if __name__=='__main__': unittest.main()
