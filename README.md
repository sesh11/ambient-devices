# ambient-devices

Connectors that let ambient-home use physical devices. Each connector is an independent package registered through the `ambient_device_sdk.devices` entry-point group.

## Reachy Mini Lite

Install the Reachy connector in the ambient-home environment:

```bash
uv add "ambient-reachy @ git+https://github.com/sesh11/ambient-devices.git@<branch>#subdirectory=packages/ambient-reachy"
```

Configure it in ambient-home:

```toml
[[devices]]
id = "reachy-desk"
kind = "reachy-mini-lite"

[devices.config]
camera = true
```

Start `reachy-mini-daemon` before ambient-home. The connector streams 16 kHz mono PCM16 audio, controls the robot body and exposes optional Pollen body tools. The provider-free movement and tool subset is vendored; see [`NOTICE`](packages/ambient-reachy/src/ambient_reachy/_vendor/pollen/NOTICE) for provenance.

For isolated hardware-test runs, pass per-run state paths to the hub with repeatable `--hub-env` options, for example `ambient-reachy-hw-test --hub-env AMBIENT_DATA_DIR=/tmp/reachy-data --hub-env AMBIENT_DEVICES_FILE=/tmp/reachy-devices.toml`.

The daemon LaunchAgent installer uses the existing `ai.ambient-home.daemon` label. Installing it replaces the old ambient-home daemon agent in place; the app agent remains managed by ambient-home.

Run diagnostics with `ambient-reachy-hw-test` and `ambient-reachy-mic-check`. Hardware checks need a Reachy Mini Lite and are not run automatically.
