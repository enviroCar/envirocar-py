
class ECConfig:

    class _ECConfig:

        CONFIG_PARAMS = {
            "ec_base_url": "https://envirocar.org/api/stable/",
            "ec_username": "",
            "ec_password": "",
            "number_of_processes": 1
        }

        def __init__(self):
            self.load_config()

        def load_config(self):
            for key, value in self.CONFIG_PARAMS.items():
                setattr(self, key, value)

    _instance = None

    def __init__(self):
        if not ECConfig._instance:
            ECConfig._instance = self._ECConfig()
        for item in self._instance.CONFIG_PARAMS:
            setattr(self, item, getattr(self._instance, item))
        