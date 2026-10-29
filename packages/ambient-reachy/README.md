# ambient-reachy

Reachy Mini Lite connector for the ambient-device-sdk. The plugin is discovered as `reachy-mini-lite`; it does not connect to hardware until `Device.start()` is called.

The connector requires a running Reachy Mini daemon. It owns the daemon connection, audio streams, robot movement lifecycle, microphone recovery, and device-specific tools. Vision requests are delegated to the hub through the SDK's injected `describe_image` callback.

The daemon installer preserves the existing `ai.ambient-home.daemon` LaunchAgent label. Installing it replaces the old daemon agent; the app agent is not managed here.
