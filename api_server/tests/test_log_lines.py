import io
import unittest

from api_server.log_lines import LogLineSplitter

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - tqdm ships with the training image
    tqdm = None


class LogLineSplitterTests(unittest.TestCase):
    def test_holds_a_partial_line_until_its_newline(self):
        splitter = LogLineSplitter()
        self.assertEqual(splitter.feed('Loading base'), [])
        self.assertEqual(splitter.feed(' model\nSaved   \n\n'), [('Loading base model', False), ('Saved', False)])

    def test_streams_each_redraw_when_the_next_one_starts(self):
        splitter = LogLineSplitter()
        self.assertEqual(splitter.feed('\rjob:  1%| 2/200 [loss: 3.2e-01]'), [])
        self.assertEqual(splitter.feed('\rjob:  2%| 3/200 [loss: 3.1e-01]'), [('job:  1%| 2/200 [loss: 3.2e-01]', True)])
        self.assertEqual(splitter.feed('\n'), [('job:  2%| 3/200 [loss: 3.1e-01]', False)])

    def test_treats_crlf_split_across_writes_as_one_line_end(self):
        splitter = LogLineSplitter()
        self.assertEqual(splitter.feed('hello\r'), [])
        self.assertEqual(splitter.feed('\nworld\n'), [('hello', False), ('world', False)])

    @unittest.skipIf(tqdm is None, 'tqdm not installed')
    def test_a_real_tqdm_bar_yields_one_redraw_per_update(self):
        splitter = LogLineSplitter()
        lines = []

        class Sink(io.StringIO):
            def write(self, s):
                lines.extend(splitter.feed(s))
                return len(s)

        bar = tqdm(total=3, file=Sink(), mininterval=0, leave=True)
        for step in range(3):
            bar.set_postfix_str(f'lr: 1.0e-04 loss: {step + 1}.000e-01')
            bar.update(1)
        bar.close()

        redraws = [line for line, is_redraw in lines if is_redraw]
        self.assertGreaterEqual(len(redraws), 3)
        self.assertTrue(all('\r' not in line for line, _ in lines))
        self.assertIn('loss: 3.000e-01', lines[-1][0])
        self.assertFalse(lines[-1][1], 'the closed bar ends with a newline')


if __name__ == '__main__':
    unittest.main()
