from lorahub.api.gpu_topology import GpuSlotInfo, homogeneous_slot_groups


def test_homogeneous_slot_groups_separates_different_gpu_models() -> None:
    groups = homogeneous_slot_groups(
        [
            GpuSlotInfo(0, "RTX 4080", 16, "8.9"),
            GpuSlotInfo(1, "Tesla V100", 32, "7.0"),
            GpuSlotInfo(2, "RTX 4080", 16, "8.9"),
        ]
    )

    assert sorted(groups) == [[0, 2], [1]]
