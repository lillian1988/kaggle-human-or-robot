# encoding: UTF-8

import pandas as pd
from prediction.stats import *
from nose.tools import *
from StringIO import StringIO
import unittest
from unittest import TestCase


class TestStats(TestCase):

    def setUp(self):
        data = """
                bid_id   bidder_id                               auction  merchandise     device  time                  country  ip               url
                785044   ffaf0a972a6dcb3910fd6b16045781e2ava5y   i1aya    sporting_goods  phone6  9763839947368421      de       19.5.213.239     yu9d3n3n8cae3gr
                1365341  ffaf0a972a6dcb3910fd6b16045781e2ava5y   2t9re    sporting_goods  phone8  9763839947368421      id       235.182.203.96   dpvxj185nd1q3jc
                2300650  ffaf0a972a6dcb3910fd6b16045781e2ava5y   6enkh    sporting_goods  phone8  9772616263157894      my       137.122.156.187  q0qfeuesqhqhv1x
        """

        self.bids = pd.read_csv(StringIO(data), sep='\s+')

    def test_get_mean_bid_per_bidder(self):
        data = get_mean_bid_per_bidder(self.bids)
        self.assertEqual(data, 3.0)

    def test_get_mean_bidder_per_auction(self):
        data = get_mean_bidder_per_auction(self.bids)
        self.assertEqual(data, 1.0)

    def test_get_bid_count_per_bidder_per_auction(self):
        data = get_bid_count_per_bidder_per_auction(self.bids)
        self.assertEqual(data, 1.0)

    def test_get_attr_distribution(self):
        data = get_attr_distribution(self.bids, "merchandise")
        self.assertEqual(data.keys()[0], "sporting_goods")
        self.assertEqual(data.values[0], 3)

    def test_get_ubiquity_score(self):
        data = get_ubiquity_score(self.bids)
        self.assertEqual(data.values[0], 1)

    def test_get_mean_attr_count_per_auction(self):
        data = get_mean_attr_count_per_auction(self.bids, "ip")
        self.assertEqual(data.values[0], 1)

    def test_get_ordered_time_list(self):
        data = get_ordered_time_list(self.bids)
        expected_list = [
            9763839947368421,
            9763839947368421,
            9772616263157894
        ]
        self.assertEqual(data.values[0], expected_list)

    def test_get_elapsed_times(self):
        data = get_elapsed_times(self.bids)
        expected_list = [
            0,
            8776315789473,
        ]
        self.assertEqual(data.values[0], expected_list)

        data = get_elapsed_times(
            pd.DataFrame([['1', 1]], columns=['bidder_id', 'time']))
        self.assertEqual(data.values[0], [])

    def test_get_mean_stats(self):
        data = get_mean_stats(self.bids)
        self.assertEqual(data['mean_nb_bid_id'].values[0], 1)
        self.assertEqual(data['mean_nb_ip'].values[0], 1)
        self.assertEqual(data['mean_nb_country'].values[0], 1)
        self.assertEqual(data['mean_nb_url'].values[0], 1)
        self.assertEqual(data['mean_nb_merchandise'].values[0], 1)

    def test_get_nunique_stats(self):
        data = get_nunique_stats(self.bids)
        self.assertEqual(data['nb_distinct_auction'].values[0], 3)
        self.assertEqual(data['nb_distinct_ip'].values[0], 3)
        self.assertEqual(data['nb_distinct_merchandise'].values[0], 1)
        self.assertEqual(data['nb_distinct_url'].values[0], 3)
        self.assertEqual(data['nb_distinct_device'].values[0], 2)

    def test_get_time_stats(self):
        data = get_time_stats(self.bids)
        self.assertEqual(data['elapsed_time_max'].values[0], 8776315789473)
        self.assertEqual(data['elapsed_time_min'].values[0], 0)
        self.assertEqual(data['time_range'].values[0], 8776315789473)
        self.assertEqual(data['ubiquity'].values[0], 1)
