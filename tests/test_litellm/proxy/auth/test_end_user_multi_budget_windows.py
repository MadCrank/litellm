"""
Unit tests for multi-budget-window enforcement on end users.
"""

from unittest.mock import AsyncMock, patch

import pytest

import litellm
from litellm.proxy._types import LiteLLM_EndUserTable
from litellm.proxy.auth.auth_checks import _end_user_multi_budget_check


def _make_end_user(**kwargs):
    defaults = dict(user_id="test-user", blocked=False, spend=0.0, budget_limits=[])
    defaults.update(kwargs)
    return LiteLLM_EndUserTable(**defaults)


@pytest.mark.asyncio
async def test_no_budget_limits_passes():
    """End users with empty budget_limits should pass without raising."""
    user = _make_end_user(budget_limits=[])
    # Should not raise
    await _end_user_multi_budget_check(end_user_object=user)


@pytest.mark.asyncio
async def test_under_budget_passes():
    """End user with spend under all windows should pass."""
    user = _make_end_user(
        budget_limits=[
            {"budget_duration": "24h", "max_budget": 10.0},
            {"budget_duration": "30d", "max_budget": 100.0},
        ]
    )
    with patch(
        "litellm.proxy.proxy_server.get_current_spend",
        new_callable=AsyncMock,
        return_value=1.0,  # well under both windows
    ):
        await _end_user_multi_budget_check(end_user_object=user)


@pytest.mark.asyncio
async def test_over_hourly_window_raises():
    """End user exceeding the 1h window should raise BudgetExceededError."""
    user = _make_end_user(
        user_id="test-user",
        budget_limits=[
            {"budget_duration": "1h", "max_budget": 5.0},
            {"budget_duration": "30d", "max_budget": 100.0},
        ],
    )

    spend_by_window = [6.0, 6.0]  # over 1h, under 30d

    call_count = 0

    async def fake_get_spend(counter_key, fallback_spend):
        nonlocal call_count
        val = spend_by_window[call_count]
        call_count += 1
        return val

    with patch(
        "litellm.proxy.proxy_server.get_current_spend", side_effect=fake_get_spend
    ):
        with pytest.raises(litellm.BudgetExceededError) as exc_info:
            await _end_user_multi_budget_check(end_user_object=user)

    err = exc_info.value
    assert err.status_code == 429
    assert "1h" in str(err)
    assert "End User" in str(err)


@pytest.mark.asyncio
async def test_over_monthly_window_raises():
    """End user exceeding only the 30d window should raise BudgetExceededError referencing 30d."""
    user = _make_end_user(
        user_id="test-user",
        budget_limits=[
            {"budget_duration": "1h", "max_budget": 50.0},
            {"budget_duration": "30d", "max_budget": 5.0},
        ],
    )

    spend_by_window = [1.0, 10.0]  # under 1h, over 30d

    call_count = 0

    async def fake_get_spend(counter_key, fallback_spend):
        nonlocal call_count
        val = spend_by_window[call_count]
        call_count += 1
        return val

    with patch(
        "litellm.proxy.proxy_server.get_current_spend", side_effect=fake_get_spend
    ):
        with pytest.raises(litellm.BudgetExceededError) as exc_info:
            await _end_user_multi_budget_check(end_user_object=user)

    err = exc_info.value
    assert err.status_code == 429
    assert "30d" in str(err)


@pytest.mark.asyncio
async def test_multiple_windows_independent():
    """End user can pass if one window is over budget but another is fine (requires exception to be raised for the over-budget one only)."""
    user = _make_end_user(
        user_id="test-user",
        budget_limits=[
            {"budget_duration": "1h", "max_budget": 50.0},
            {"budget_duration": "30d", "max_budget": 100.0},
        ],
    )
    # Both under their respective limits — should pass
    with patch(
        "litellm.proxy.proxy_server.get_current_spend",
        new_callable=AsyncMock,
        return_value=1.0,
    ):
        await _end_user_multi_budget_check(end_user_object=user)


@pytest.mark.asyncio
async def test_budget_limit_entry_objects_coerced():
    """BudgetLimitEntry Pydantic objects (not dicts) must be handled without KeyError.

    While budget_limits is normally serialized as List[dict], the auth check must
    tolerate BudgetLimitEntry objects in case they arrive without prior serialization.
    """
    from litellm.proxy._types import BudgetLimitEntry

    user = _make_end_user(budget_limits=[])
    # Bypass Pydantic validation to simulate BudgetLimitEntry objects reaching the check
    object.__setattr__(
        user,
        "budget_limits",
        [BudgetLimitEntry(budget_duration="24h", max_budget=10.0)],
    )

    with patch(
        "litellm.proxy.proxy_server.get_current_spend",
        new_callable=AsyncMock,
        return_value=1.0,
    ):
        # Should not raise TypeError / KeyError — model_dump() coerces the object
        await _end_user_multi_budget_check(end_user_object=user)
