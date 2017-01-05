# encoding: UTF-8

import numpy as np
import pandas as pd
from prediction.utils import tweak_max, tweak_min


def get_mean_bid_per_bidder(bids):
    return bids.groupby("bidder_id").bid_id.size().mean()


def get_mean_bidder_per_auction(bids):
    return bids.groupby("auction").bidder_id.size().mean()


def get_bid_count_per_bidder_per_auction(bids):
    return bids.groupby(["auction", "bidder_id"]).bid_id.size().mean()


def get_attr_distribution(bids, attribute):
    return bids.groupby(attribute).size().sort_values(ascending=False)


def get_ubiquity_score(bids):
    gb = bids.groupby("bidder_id")
    return gb.time.size() - gb.time.nunique()


def get_mean_attr_count_per_auction(bids, attribute):
    gb = bids.groupby(["bidder_id", "auction"])
    return gb[attribute].nunique().reset_index().groupby("bidder_id").mean()


def get_ordered_time_list(bids):
    time_gb = bids.sort_values('time').groupby('bidder_id')
    return time_gb.time.apply(list)


def get_elapsed_times(bids):
    return get_ordered_time_list(bids).apply(lambda a: [t - s for s, t in zip(a, a[1:])])


def get_mean_stats(bids):
    l_mean = ['bid_id', 'country', 'ip', 'merchandise', 'url']
    return {'mean_nb_%s' % f: get_mean_attr_count_per_auction(bids, f) for f in l_mean}


def get_nunique_stats(bids):
    l_nunique = ['auction', 'device', 'ip', 'merchandise', 'url']
    bidder_gb = bids.groupby("bidder_id")
    return {'nb_distinct_%s' % f: bidder_gb[f].nunique() for f in l_nunique}


def get_time_stats(bids):
    return {
        "time_std": bids.sort_values('time').groupby('bidder_id').time.std().fillna(0),
        "time_range": get_elapsed_times(bids).apply(sum),
        "ubiquity": get_ubiquity_score(bids),
        "elapsed_time_std": get_elapsed_times(bids).apply(np.std).fillna(0),
        "elapsed_time_min": get_elapsed_times(bids).apply(tweak_min).fillna(0),
        "elapsed_time_max": get_elapsed_times(bids).apply(tweak_max).fillna(0),
    }


def get_stats(bids):
    # Count bids
    features_dict = {
        "nb_bids": bids.groupby("bidder_id").bid_id.size()
    }
    features_dict.update(get_mean_stats(bids))
    features_dict.update(get_nunique_stats(bids))
    features_dict.update(get_time_stats(bids))

    stats = pd.concat(features_dict.values(), axis=1)
    stats.columns = features_dict.keys()

    return stats


def get_feature_target(stats, bidders):
    merged_data = pd.concat(
        [bidders.set_index("bidder_id"), stats],
        axis=1,
        join="inner"
    )
    return (merged_data[stats.columns],
            merged_data.get('outcome'))
