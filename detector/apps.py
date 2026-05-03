from django.apps import AppConfig


class DetectorConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "detector"

    def ready(self):
        """
        Pre-load both models when Django starts so the first
        user request is not slow.
        """
        # Only load in the main process (not the reloader child)
        import os
        if os.environ.get("RUN_MAIN") != "true":
            return

        try:
            from detector.scene_predictor import _load as load_scene
            load_scene()
        except Exception as e:
            print(f"[DetectorConfig] Scene model preload failed: {e}")

        try:
            from detector.object_detector import _load as load_det
            load_det()
        except Exception as e:
            print(f"[DetectorConfig] Object detector preload failed: {e}")
