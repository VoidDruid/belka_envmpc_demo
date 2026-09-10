class DummyViewer:
    """Headless viewer stub used by simulator runs without a real MuJoCo viewer."""

    def is_running(self):
        """Pretend that the viewer is always alive."""
        return True

    def sync(self):
        """No-op sync hook."""
        pass

    def close(self):
        """No-op close hook."""
        pass
