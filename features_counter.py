num_raw_channels=7
num_base_feats=8
num_windows=5

used_win=1
used_raw=1

counter_channel_feats = used_win * num_raw_channels * num_base_feats * num_windows
counter_raw_feats     = used_raw * (num_raw_channels * 1 * 1)

num_sync_feats=15
num_conv_feats=4

counter_sync_channel_feats = (num_sync_feats + num_conv_feats) * counter_channel_feats
counter_sync_raw_feats     = (num_sync_feats + num_conv_feats) * counter_raw_feats


# counter_conv_channel_feats = num_conv_feats * counter_channel_feats
# counter_conv_raw_feats     = num_conv_feats * counter_raw_feats

counter_pairwise_feats = (counter_sync_channel_feats + counter_sync_raw_feats)

num_group_feats=6

counter_group_feats = num_group_feats * counter_pairwise_feats

print("Total Pairs Features = " + str(counter_pairwise_feats))
print("Total Group Features = " + str(counter_group_feats))