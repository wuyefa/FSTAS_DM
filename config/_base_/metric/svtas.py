METRIC = dict(
    SVTAS = dict(
        name = "SVTASegmentationMetric",
        overlap = [.1, .25, .5],
        segment_windows_size = 64,
        actions_map_file_path = "./data/gtea/mapping.txt",
        file_output = False,
        score_output = False),
)