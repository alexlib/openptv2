"""EventBus unit tests (pure Python, no display) — the cross-view backbone."""
from openptv2.gui_tk.events import CameraClick, EventBus, ParamsChanged


def test_publish_delivers_to_type_subscribers_only():
    bus = EventBus()
    clicks, params = [], []
    bus.subscribe(CameraClick, clicks.append)
    bus.subscribe(ParamsChanged, params.append)

    n = bus.publish(CameraClick(cam=1, x=10.0, y=20.0))
    assert n == 1
    assert clicks and clicks[0].cam == 1 and clicks[0].x == 10.0
    assert params == []  # different type not delivered


def test_multiple_subscribers_all_receive():
    bus = EventBus()
    got = []
    for _ in range(3):
        bus.subscribe(CameraClick, lambda e, g=got: g.append(e.cam))
    assert bus.publish(CameraClick(cam=2, x=0, y=0)) == 3
    assert got == [2, 2, 2]


def test_unsubscribe_stops_delivery():
    bus = EventBus()
    got = []
    off = bus.subscribe(CameraClick, lambda e: got.append(e))
    off()
    assert bus.publish(CameraClick(cam=0, x=0, y=0)) == 0
    assert got == []


def test_faulty_subscriber_is_isolated():
    bus = EventBus()
    good = []

    def boom(_):
        raise ValueError("bad subscriber")

    bus.subscribe(CameraClick, boom)
    bus.subscribe(CameraClick, good.append)
    bus.publish(CameraClick(cam=1, x=1, y=1))
    assert len(good) == 1                 # good subscriber still ran
    assert len(bus.last_errors) == 1      # error captured, not raised
