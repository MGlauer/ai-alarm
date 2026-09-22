"""Run the alarm system with the simulated sensors: the API is on http://localhost:5000/api (see api.py), and the
frontend at http://localhost:5000 if it is built (`npm run build` in frontend/).

Start a demo from the frontend ("Simulate"), or with `curl -X POST localhost:5000/api/demos/intruder`.
"""
import logging

from ai_alarm.system import build_system

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    system = build_system()
    system.start()
    system.app.run(port=5000, threaded=True)
