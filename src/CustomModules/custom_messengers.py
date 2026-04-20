# CREATED BY GEMINI 3.1

import jax
from numpyro.primitives import Messenger

class scale_sites(Messenger):
    """Custom handler to scale the log-probability of specific sites."""
    def __init__(self, fn=None, scale=1.0, sites=None):
        self.scale = scale
        # Accept a list of site names to target
        self.sites = sites if sites else []
        super().__init__(fn)

    def process_message(self, msg):
        # Intercept the message and check the site name
        if msg.get("name") in self.sites and msg.get("type") in ("sample", "param"):
            current_scale = msg.get("scale")
            
            # Apply the scale factor directly to the message
            if current_scale is None:
                msg["scale"] = self.scale
            else:
                msg["scale"] = self.scale * current_scale