# Push notifications (opt-in)

Open Preferences > Push Notifications on the machine actually running the training server (including a pod).
Install the ntfy mobile app or use its web app, generate a random topic, subscribe to the same topic on the
receiving device, enable notifications, choose events, Save, then Send test notification.
A reserved private ntfy.sh topic may be used with a publishing access token. Ordinary ntfy topics are public
to anyone who knows the name; a random topic is not access control. Keep it secret. Run names are omitted
unless explicitly enabled. No dataset contents, file paths, images or training logs are sent.

Notifications are off by default. Delivery uses HTTPS to ntfy.sh, with a bounded queue and a background
worker, without extra dependencies. Requests have a five-second socket timeout; delivery is best effort,
not guaranteed (e.g. server termination or loss of internet). Credentials are stored locally in the
git-ignored user data, not included in normal preferences API responses; the token is not encrypted at rest.
An accepted test confirms the service accepted the request, not that the phone displayed it.
Settings are per running server, not synchronized across machines.

Initial events: run completion, terminal failure, exhausted memory recovery, queue drained (including
failed runs), manually paused/stopped, run started, cache stage completed and sample preview written.
Queue drained is not a claim that every queued job succeeded. Future candidates include validation,
best checkpoint, epoch completion, downloads, captioning, diagnostics and confirmed pod shutdown.
