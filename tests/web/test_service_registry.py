import pytest

from web.service_registry import ProcessingServiceRegistry, SessionCapacityError


class FakeService:
    def __init__(self):
        self.shutdown_calls = 0
        self.running = False

    def shutdown(self):
        self.shutdown_calls += 1


def make_registry(*, clock, max_services=3, idle_timeout_seconds=60, is_service_busy=None):
    created = []
    disposed_sessions = []

    def factory():
        service = FakeService()
        created.append(service)
        return service

    registry = ProcessingServiceRegistry(
        service_factory=factory,
        shutdown_service=lambda service: service.shutdown(),
        is_service_busy=is_service_busy,
        on_dispose=disposed_sessions.append,
        max_services=max_services,
        idle_timeout_seconds=idle_timeout_seconds,
        clock=clock,
    )
    return registry, created, disposed_sessions


def test_sessions_receive_independent_services_and_reuse_their_own_service():
    now = [0.0]
    registry, _, _ = make_registry(clock=lambda: now[0])

    first = registry.acquire(None)
    registry.release(first)
    second = registry.acquire(None)
    registry.release(second)

    assert first.session_id != second.session_id
    assert first.service is not second.service
    assert first.set_cookie is True
    assert second.set_cookie is True

    again = registry.acquire(first.session_id)
    registry.release(again)

    assert again.service is first.service
    assert again.set_cookie is False


def test_idle_session_is_shutdown_and_removed():
    now = [0.0]
    registry, created, disposed_sessions = make_registry(
        clock=lambda: now[0],
        idle_timeout_seconds=10,
    )

    lease = registry.acquire(None)
    registry.release(lease)
    now[0] = 11.0
    registry.cleanup()

    assert registry.service_count == 0
    assert created[0].shutdown_calls == 1
    assert disposed_sessions == [lease.session_id]


def test_registry_rejects_new_session_when_every_service_is_busy():
    now = [0.0]
    registry, _, _ = make_registry(clock=lambda: now[0], max_services=1)

    active = registry.acquire(None)

    with pytest.raises(SessionCapacityError):
        registry.acquire(None)

    registry.release(active)
    replacement = registry.acquire(None)
    registry.release(replacement)

    assert replacement.service is not active.service


def test_running_service_is_not_evicted_when_capacity_is_reached():
    now = [0.0]
    registry, _, _ = make_registry(
        clock=lambda: now[0],
        max_services=1,
        is_service_busy=lambda service: service.running,
    )

    running = registry.acquire(None)
    running.service.running = True
    registry.release(running)

    with pytest.raises(SessionCapacityError):
        registry.acquire(None)

    running.service.running = False
    replacement = registry.acquire(None)
    registry.release(replacement)

    assert replacement.service is not running.service
    assert running.service.shutdown_calls == 1