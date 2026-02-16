"""Tests for SamplingListManager and get_task_id_set_from_config."""

import pytest
from affine.core.sampling_list import SamplingListManager, get_task_id_set_from_config


class TestGetTaskIdSetFromConfig:
    """Test get_task_id_set_from_config helper."""

    def test_with_sampling_list(self):
        config = {'sampling_config': {'sampling_list': [1, 2, 3, 4, 5]}}
        result = get_task_id_set_from_config(config)
        assert result == {1, 2, 3, 4, 5}

    def test_with_empty_sampling_list(self):
        config = {'sampling_config': {'sampling_list': []}}
        result = get_task_id_set_from_config(config)
        assert result == set()

    def test_no_sampling_config(self):
        result = get_task_id_set_from_config({})
        assert result == set()

    def test_no_sampling_list_in_config(self):
        config = {'sampling_config': {'other_key': 'value'}}
        result = get_task_id_set_from_config(config)
        assert result == set()

    def test_deduplicates(self):
        config = {'sampling_config': {'sampling_list': [1, 1, 2, 2, 3]}}
        result = get_task_id_set_from_config(config)
        assert result == {1, 2, 3}


class TestSamplingListManagerInit:
    """Test SamplingListManager.initialize_sampling_list."""

    @pytest.mark.asyncio
    async def test_basic_init(self):
        manager = SamplingListManager()
        result = await manager.initialize_sampling_list("test_env", [[0, 100]], 10)
        assert len(result) == 10
        assert all(0 <= x < 100 for x in result)
        assert result == sorted(result)  # should be sorted

    @pytest.mark.asyncio
    async def test_init_larger_than_available(self):
        manager = SamplingListManager()
        result = await manager.initialize_sampling_list("test_env", [[0, 5]], 100)
        assert len(result) == 5
        assert sorted(result) == [0, 1, 2, 3, 4]

    @pytest.mark.asyncio
    async def test_init_empty_range(self):
        manager = SamplingListManager()
        result = await manager.initialize_sampling_list("test_env", [], 10)
        assert len(result) == 0

    @pytest.mark.asyncio
    async def test_init_multiple_ranges(self):
        manager = SamplingListManager()
        result = await manager.initialize_sampling_list("test_env", [[0, 5], [100, 105]], 8)
        assert len(result) == 8
        for x in result:
            assert (0 <= x < 5) or (100 <= x < 105)


class TestSamplingListManagerRotate:
    """Test SamplingListManager.rotate_sampling_list."""

    @pytest.mark.asyncio
    async def test_basic_rotation(self):
        manager = SamplingListManager()
        current = list(range(10))
        dataset_range = [[0, 100]]
        new_list, removed, added = await manager.rotate_sampling_list(
            "test_env", current, dataset_range,
            sampling_count=10, rotation_count=3
        )
        assert len(removed) == 3
        assert len(added) == 3
        assert len(new_list) == 10
        # Removed from front
        assert removed == [0, 1, 2]

    @pytest.mark.asyncio
    async def test_rotation_zero(self):
        """rotation_count=0 means no rotation, only size adjustment."""
        manager = SamplingListManager()
        current = list(range(10))
        new_list, removed, added = await manager.rotate_sampling_list(
            "test_env", current, [[0, 100]],
            sampling_count=10, rotation_count=0
        )
        assert removed == []
        assert added == []
        assert new_list == current

    @pytest.mark.asyncio
    async def test_negative_rotation_count(self):
        manager = SamplingListManager()
        current = list(range(10))
        new_list, removed, added = await manager.rotate_sampling_list(
            "test_env", current, [[0, 100]],
            sampling_count=10, rotation_count=-1
        )
        assert new_list == current
        assert removed == []
        assert added == []

    @pytest.mark.asyncio
    async def test_fill_mode(self):
        """When current < target, should only add."""
        manager = SamplingListManager()
        current = [0, 1, 2]
        new_list, removed, added = await manager.rotate_sampling_list(
            "test_env", current, [[0, 100]],
            sampling_count=10, rotation_count=3
        )
        assert removed == []
        assert len(added) == 7  # fill to 10
        assert len(new_list) == 10

    @pytest.mark.asyncio
    async def test_shrink_mode(self):
        """When current > target, should remove surplus + rotation_count."""
        manager = SamplingListManager()
        current = list(range(15))
        new_list, removed, added = await manager.rotate_sampling_list(
            "test_env", current, [[0, 100]],
            sampling_count=10, rotation_count=2
        )
        # surplus=5, remove 5+2=7, add 2
        assert len(removed) == 7
        assert len(added) == 2
        assert len(new_list) == 10

    @pytest.mark.asyncio
    async def test_safety_check_large_rotation(self):
        """Should skip if sampling_count + rotation_count > 80% of dataset."""
        manager = SamplingListManager()
        current = list(range(10))
        new_list, removed, added = await manager.rotate_sampling_list(
            "test_env", current, [[0, 12]],  # dataset size 12
            sampling_count=10, rotation_count=3  # 10+3=13 > 12*0.8=9.6
        )
        assert new_list == current
        assert removed == []
        assert added == []

    @pytest.mark.asyncio
    async def test_insufficient_available_ids(self):
        """Should skip if not enough available IDs for addition."""
        manager = SamplingListManager()
        current = list(range(95))
        new_list, removed, added = await manager.rotate_sampling_list(
            "test_env", current, [[0, 200]],
            sampling_count=95, rotation_count=110  # need 110, only 105 available
        )
        assert new_list == current

    @pytest.mark.asyncio
    async def test_added_ids_not_in_remaining(self):
        """Added IDs should not duplicate remaining IDs."""
        manager = SamplingListManager()
        current = list(range(50))
        new_list, removed, added = await manager.rotate_sampling_list(
            "test_env", current, [[0, 200]],
            sampling_count=50, rotation_count=10
        )
        remaining = current[10:]  # after removing 10 from front
        remaining_set = set(remaining)
        for a in added:
            assert a not in remaining_set

    @pytest.mark.asyncio
    async def test_prioritize_new(self):
        """With prioritize_new=True, additions should come from later segments."""
        manager = SamplingListManager()
        current = list(range(5))
        new_list, removed, added = await manager.rotate_sampling_list(
            "test_env", current, [[0, 10], [1000, 1010]],
            sampling_count=5, rotation_count=3, prioritize_new=True
        )
        # Added should come from [1000, 1010] segment first
        assert len(added) == 3
        assert all(1000 <= a < 1010 for a in added)
