"""End-to-end checks of source fidelity, citations, caching and CLI isolation."""
import argparse
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pymupdf

from report_parser.cli import main, parse_page_selection
from report_parser.pipeline import ParserConfig, parse_document


def make_report(path):
    with pymupdf.open() as doc:
        for number in range(1, 4):
            page = doc.new_page(width=612, height=792)
            page.insert_text((50, 25), 'ACME annual report', fontsize=9)
            page.insert_text((300, 777), str(number), fontsize=9)
            page.insert_text((50, 90), 'Operating review', fontsize=16)
            # Two columns, deliberately inserted row-by-row in the PDF stream.
            for index in range(5):
                page.insert_text((50, 120 + index*14), f'Left column sentence {index}.', fontsize=10)
                page.insert_text((330, 120 + index*14), f'Right column sentence {index}.', fontsize=10)
            if number == 2:
                page.insert_text((50, 230), 'In millions', fontsize=10)
                for index, row in enumerate([
                    ('Metric','2025','2024'), ('Revenue','$1,234','$1,105'),
                    ('Costs','(456)','(490)'), ('Margin','15.2%','14.1%')]):
                    for x,value in zip((55,330,450),row):
                        page.insert_text((x,275+index*25),value,fontsize=10)
                for x in (50,320,440,550):
                    page.draw_line((x,255),(x,355))
                for y in (255,280,305,330,355):
                    page.draw_line((50,y),(550,y))
        doc.new_page(width=612, height=792)  # An image-only/empty page must be flagged.
        doc.save(path)


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp_base = Path(__file__).resolve().parents[1]/'tmp'/'parser-tests'
        self.temp_base.mkdir(parents=True,exist_ok=True)
        self.directory = tempfile.TemporaryDirectory(dir=self.temp_base)
        self.root = Path(self.directory.name)
        self.pdf = self.root/'ACME_FY2025_annual_report.pdf'
        make_report(self.pdf)
        self.output = self.root/'parsed'
        self.metadata = {'ticker':'ACME','fiscal_year':2025,'company':'ACME'}

    def tearDown(self):
        self.assertTrue(self.root.resolve().is_relative_to(self.temp_base.resolve()))
        self.directory.cleanup()

    def parse(self, **kwargs):
        return parse_document(self.pdf,self.output,source_metadata=self.metadata,**kwargs)

    def test_citations_reading_order_tables_and_sparse_pages(self):
        meta=self.parse(config=ParserConfig(table_mode='all'))
        pages=[json.loads(line) for line in (self.output/'pages.jsonl').read_text(encoding='utf-8').splitlines()]
        tables=[json.loads(line) for line in (self.output/'tables.jsonl').read_text(encoding='utf-8').splitlines()]
        self.assertTrue(meta['complete'])
        self.assertEqual(meta['counts']['pages'],4)
        self.assertEqual(meta['counts']['tables'],1)
        self.assertEqual(meta['counts']['pages_needing_ocr_or_review'],1)
        first=pages[0]
        self.assertNotIn('ACME annual report',first['text'])
        self.assertTrue(any(line['text']=='ACME annual report' for line in first['removed_margins']))
        self.assertLess(first['text'].index('Left column sentence 4.'),first['text'].index('Right column sentence 0.'))
        table=tables[0]
        self.assertEqual(table['citation']['pdf_page'],2)
        self.assertEqual(table['citation']['ticker'],'ACME')
        self.assertEqual(table['citation']['sha256'],meta['sha256'])
        self.assertIn(['Costs','(456)','(490)'],table['rows'])
        self.assertEqual(table['bbox'],[50.0,255.0,550.0,355.0])
        table_blocks=[element for element in pages[1]['elements'] if element['kind']=='table']
        self.assertEqual(len(table_blocks),1)
        self.assertEqual(table_blocks[0]['table_id'],table['table_id'])
        self.assertEqual(pages[1]['text'].count('$1,234'),1)
        self.assertTrue(pages[1]['table_source_lines'])
        for page in pages:
            for element in page['elements']:
                self.assertEqual(element['citation']['pdf_page'],page['page_number'])
                self.assertEqual(element['citation']['bbox'],element['bbox'])
        self.assertTrue(pages[3]['warnings'])

    def test_resume_rebuilds_changed_config_metadata_and_corrupt_artifact(self):
        config=ParserConfig(table_mode='off')
        self.assertFalse(self.parse(config=config)['reused'])
        self.assertTrue(self.parse(config=config)['reused'])
        (self.output/'pages.jsonl').write_text('',encoding='utf-8')
        self.assertFalse(self.parse(config=config)['reused'])
        self.metadata['ticker']='NEW'
        self.assertFalse(self.parse(config=config)['reused'])
        self.assertIn('NEW',(self.output/'pages.jsonl').read_text(encoding='utf-8'))
        self.assertFalse(self.parse(config=ParserConfig(table_mode='off',pages=(2,)))['reused'])

    def test_partial_selection_keeps_original_page_numbers_and_rejects_out_of_range(self):
        meta=self.parse(config=ParserConfig(table_mode='off',pages=(2,4)))
        self.assertTrue(meta['partial'])
        self.assertEqual(meta['selected_pdf_pages'],[2,4])
        pages=[json.loads(s) for s in (self.output/'pages.jsonl').read_text(encoding='utf-8').splitlines()]
        self.assertEqual([p['page_number'] for p in pages],[2,4])
        with self.assertRaisesRegex(ValueError,'exceeds document length'):
            self.parse(config=ParserConfig(pages=(5,)))

    def test_table_failure_keeps_native_text_and_flags_page(self):
        with patch('report_parser.pipeline.extract_tables',side_effect=ValueError('fixture failure')):
            self.parse(config=ParserConfig(table_mode='all',pages=(2,)))
        page=json.loads((self.output/'pages.jsonl').read_text(encoding='utf-8'))
        self.assertIn('$1,234',page['text'])
        self.assertEqual(page['table_scan'],'error')
        self.assertTrue(any('fixture failure' in w for w in page['warnings']))

    def test_rotated_page_keeps_display_coordinates_and_reports_limit(self):
        with pymupdf.open() as doc:
            page=doc.new_page(width=612,height=792)
            page.insert_text((50,100),'Rotation preserves source coordinates',fontsize=12)
            page.set_rotation(90)
            rotated=self.root/'rotated.pdf'
            doc.save(rotated)
        parse_document(rotated,self.output,config=ParserConfig(table_mode='all'))
        page=json.loads((self.output/'pages.jsonl').read_text(encoding='utf-8'))
        self.assertEqual((page['width'],page['height']),(792,612))
        self.assertEqual(page['source_lines'][0]['direction'],[0,1])
        self.assertEqual(page['table_scan'],'error')
        self.assertTrue(any('rotated_page' in w for w in page['warnings']))
        box=page['elements'][0]['citation']['bbox']
        self.assertTrue(0<=box[0]<box[2]<=792)
        self.assertTrue(0<=box[1]<box[3]<=612)

    def test_cli_deduplicates_inputs_and_isolates_partial_outputs(self):
        with contextlib.redirect_stdout(io.StringIO()):
            result=main(['--input',str(self.pdf),str(self.pdf),'--manifest',str(self.root/'absent.json'),
                         '--output',str(self.output),'--workers','1','--pages','2','--table-mode','off'])
        self.assertEqual(result,0)
        registry=json.loads((self.output/'parse_manifest.json').read_text(encoding='utf-8'))
        self.assertEqual(len(registry['documents']),1)
        self.assertIn('__pages_',registry['documents'][0]['output_dir'])
        self.assertTrue(registry['documents'][0]['partial'])

    def test_cli_rejects_same_name_destination_collision(self):
        another=self.root/'another'
        another.mkdir()
        copy=another/self.pdf.name
        copy.write_bytes(self.pdf.read_bytes())
        with contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit) as error:
            main(['--input',str(self.pdf),str(copy),'--manifest',str(self.root/'absent.json'),
                  '--output',str(self.output),'--workers','1'])
        self.assertEqual(error.exception.code,2)

    def test_page_selection_validation(self):
        self.assertEqual(parse_page_selection('3,1-2,2'),(1,2,3))
        for value in ('0','3-1','a','1,,2','-1'):
            with self.assertRaises(argparse.ArgumentTypeError):
                parse_page_selection(value)


if __name__=='__main__':
    unittest.main()
