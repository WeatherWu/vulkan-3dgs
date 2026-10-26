from array import array
import random
import unittest

import training_range_key_audit as audit


class DirectRangeKeyAuditTests(unittest.TestCase):
    def test_zero_and_one_do_not_require_sorted_scratch(self):
        self.assertEqual(list(audit.compare([], [], None, 3)[1]), [0] * 12)
        self.assertEqual(list(audit.compare([2], [0], None, 3)[1]),
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0])
        # The deliberately invalid/unproduced scratch must not be selected.
        self.assertEqual(audit.compare([2], [0], [999], 3)[1][9], 1)

    def test_empty_tiles_and_last_item_boundary(self):
        words, ranges = audit.compare([4, 1, 4, 1], [1, 3, 0, 2], [1, 1, 4, 4], 6)
        self.assertEqual(list(ranges), [0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0,
                                       0, 0, 0, 0, 2, 2, 0, 0, 0, 0, 0, 0])
        self.assertEqual(words[4 * 4 + 1], 4)

    def test_all_ties_and_unused_capacity(self):
        _, ranges = audit.compare([2] * 33 + [999] * 7, list(range(33)) + [999] * 7,
                                  [2] * 33 + [999] * 7, 4, 33)
        self.assertEqual(ranges[9], 33)
        self.assertEqual(ranges[10:12], array("I", [0, 0]))

    def test_warp_workgroup_and_radix_partition_boundaries(self):
        rng = random.Random(91217)
        for count in (2, 31, 32, 33, 255, 256, 257, 4095, 4096, 4097):
            original = [rng.randrange(17) for _ in range(count)]
            order = sorted(range(count), key=lambda i: original[i])
            keys = [original[i] for i in order]
            with self.subTest(count=count):
                _, ranges = audit.compare(original, order, keys, 17)
                self.assertEqual(sum(ranges[1::4]), count)

    def test_bad_permutations_ids_and_missing_producer_are_rejected(self):
        for original, order, keys, tiles in (([0, 1], [0, 0], [0, 1], 2),
                                             ([0, 1], [0, 2], [0, 1], 2),
                                             ([0, 2], [0, 1], [0, 2], 2),
                                             ([0, 1], [0, 1], None, 2),
                                             ([0, 1], [0, 1], [0], 2),
                                             ([0, 1], [0, 1], [1, 1], 2),
                                             ([0, 1], [1, 0], [1, 0], 2)):
            with self.assertRaises(ValueError):
                audit.compare(original, order, keys, tiles)

    def test_address_proxy_distinguishes_contiguous_and_scattered_keys(self):
        contiguous = audit.address_blocks(list(range(256)), 256)
        scattered = audit.address_blocks(list(range(0, 2048, 8)), 256)
        self.assertEqual(contiguous["sum_indirect_distinct_key_blocks"],
                         contiguous["sum_direct_distinct_key_blocks"])
        self.assertGreater(scattered["sum_indirect_distinct_key_blocks"],
                           scattered["sum_direct_distinct_key_blocks"])


if __name__ == "__main__":
    unittest.main()
