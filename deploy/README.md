# Deployment Files

Helpers for running the intranet segmentation web app as a long-lived service.

| File | Host | Purpose |
| --- | --- | --- |
| [`microseg-web.service`](microseg-web.service) | Ubuntu / systemd Linux | Service unit that starts the app at boot and restarts it on failure |
| [`start_web_server.bat`](start_web_server.bat) | Windows | Double-click or Task Scheduler launcher |

Full instructions, including the offline wheelhouse install, are in
[`docs/intranet_web_app.md`](../docs/intranet_web_app.md).
