"""Tests for ProcessMonitorWatcher._reconcile_category.

When extraction detects a different document type than the declared category
(e.g. a quote filed as 'Invoice'), the watcher must update process_monitor's
category to the detected type instead of leaving a mislabeled record.
"""
import sys
import unittest

sys.path.insert(0, "src")
sys.path.insert(0, ".")

from src.services.process_monitor_watcher import ProcessMonitorWatcher


class _Cursor:
    def __init__(self, log):
        self._log = log
    def __enter__(self):
        return self
    def __exit__(self, *a):
        return False
    def execute(self, sql, params=None):
        self._log.append((sql, params))


class _Conn:
    def __init__(self, log):
        self._log = log
        self.closed = False
    def cursor(self):
        return _Cursor(self._log)
    def close(self):
        self.closed = True


def _watcher(log):
    # Bypass __init__ (needs a real agent_nick); only _get_connection is used.
    w = ProcessMonitorWatcher.__new__(ProcessMonitorWatcher)
    w._get_connection = lambda autocommit=True: _Conn(log)
    return w


class TestReconcileCategory(unittest.TestCase):
    def test_quote_filed_as_invoice_is_reconciled(self):
        log = []
        _watcher(log)._reconcile_category(875, declared="Invoice", detected="quote")
        self.assertEqual(len(log), 1, "expected one UPDATE")
        sql, params = log[0]
        self.assertIn("UPDATE proc.process_monitor", sql)
        self.assertIn("category", sql)
        self.assertEqual(params[0], "Quote")
        self.assertEqual(params[2], 875)

    def test_matching_type_is_noop(self):
        log = []
        _watcher(log)._reconcile_category(1, declared="Quote", detected="quote")
        self.assertEqual(log, [])

    def test_po_aliases_match_no_update(self):
        log = []
        # 'PO' (category) and 'purchase_order' (engine doc_type) are the same type
        _watcher(log)._reconcile_category(1, declared="PO", detected="purchase_order")
        self.assertEqual(log, [])

    def test_unknown_detected_type_is_noop(self):
        log = []
        _watcher(log)._reconcile_category(1, declared="Invoice", detected="")
        self.assertEqual(log, [])

    def test_invoice_detected_under_quote_reconciles(self):
        log = []
        _watcher(log)._reconcile_category(2, declared="Quote", detected="invoice")
        self.assertEqual(log[0][1][0], "Invoice")


if __name__ == "__main__":
    unittest.main()
