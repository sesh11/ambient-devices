"""Reachy Mini Lite implementation of ambient-device-sdk's Device protocol."""

from __future__ import annotations
import asyncio
import logging
from typing import Any, Protocol
from collections.abc import AsyncIterator

from ambient_device_sdk import (
    ToolSpec,
    EventSink,
    Capability,
    AudioFormat,
    DeviceHealth,
    SessionPhase,
    SessionState,
    DeviceContext,
)

from ambient_reachy.look import build_look_tool
from ambient_reachy.config import ReachyConfig
from ambient_reachy.body_tools import build_body_tools


logger = logging.getLogger(__name__)
SAMPLE_RATE = 16_000
_MIC_QUEUE_SIZE = 20


class ReachyUnavailableError(RuntimeError):
    """Raised when the Reachy daemon or USB robot cannot be reached."""


class HardwareFactory(Protocol):
    """Factory boundary that keeps device construction independent of hardware imports."""

    def make_robot(self) -> Any:
        """Connect to Reachy Mini."""

    def make_movement_manager(self, robot: Any) -> Any:
        """Create Pollen's movement manager."""

    def check_microphone(self, seconds: float, emit: EventSink) -> str:
        """Check audio input and optionally recover the XMOS processor."""


class _DefaultHardware:
    def make_robot(self) -> Any:
        from reachy_mini import ReachyMini

        return ReachyMini()

    def make_movement_manager(self, robot: Any) -> Any:
        from ambient_reachy._vendor.pollen.moves import MovementManager

        return MovementManager(current_robot=robot)

    def check_microphone(self, seconds: float, emit: EventSink) -> str:
        from ambient_reachy.mic_check import ensure_microphone_audio

        return ensure_microphone_audio(seconds, emit)


class ReachyDevice:
    """Reachy Mini Lite audio, movement, health, and tool integration."""

    def __init__(self, context: DeviceContext, *, hardware: HardwareFactory | None = None) -> None:
        """Create a configured device without opening hardware connections."""
        self.context = context
        self.config = ReachyConfig.from_mapping(context.config)
        self._hardware: HardwareFactory = hardware or _DefaultHardware()
        self._robot: Any | None = None
        self._movement_manager: Any | None = None
        self._started = False
        self._daemon_lost = False
        self._audio_fault: str | None = None
        self._mic_silent = False
        self._play_generation = 0
        self._mic_queue: asyncio.Queue[bytes | None] = asyncio.Queue()
        self._play_queue: asyncio.Queue[bytes] = asyncio.Queue()
        self._mic_task: asyncio.Task[None] | None = None
        self._play_task: asyncio.Task[None] | None = None
        self._watch_task: asyncio.Task[None] | None = None
        self._movement_started = False
        self._session_id: str | None = None
        self._silence_seconds = 0.0
        self._silence_event_sent = False
        self._resample_logged = False

    @property
    def id(self) -> str:
        """Return this device's configured identifier."""
        return self.context.device_id

    @property
    def kind(self) -> str:
        """Return the registered device kind."""
        return "reachy-mini-lite"

    @property
    def capabilities(self) -> frozenset[Capability]:
        """Return the capabilities enabled by this configuration."""
        capabilities = {Capability.LISTEN, Capability.SPEAK, Capability.MOVE}
        if self.config.camera:
            capabilities.add(Capability.SEE)
        return frozenset(capabilities)

    @property
    def mic_format(self) -> AudioFormat:
        """Return the microphone PCM format."""
        return AudioFormat(SAMPLE_RATE)

    @property
    def speaker_format(self) -> AudioFormat:
        """Return the speaker PCM format."""
        return AudioFormat(SAMPLE_RATE)

    async def start(self) -> None:
        """Start microphone/playback streams and device liveness monitoring."""
        if self._started:
            return
        self._daemon_lost = False
        self._audio_fault = None
        self._mic_silent = False
        self._silence_seconds = 0.0
        self._silence_event_sent = False
        self._resample_logged = False
        self._mic_queue = asyncio.Queue(maxsize=_MIC_QUEUE_SIZE)
        self._play_queue = asyncio.Queue()
        try:
            from ambient_reachy.daemon import wait_for_daemon

            if self.config.mic_autorecover:
                try:
                    result = await asyncio.to_thread(
                        self._hardware.check_microphone,
                        self.config.mic_check_seconds,
                        self.context.emit,
                    )
                    if result != "ok":
                        self._mic_silent = result == "silent"
                except Exception as exc:
                    logger.exception("Microphone check failed; continuing to daemon wait")
                    self.context.emit(
                        "mic.check_failed", "mic_check", "error", error=str(exc), exc_type=type(exc).__name__
                    )
            ready = await wait_for_daemon(
                self.config.daemon_status_url,
                self.config.daemon_wait_s,
                self.config.daemon_poll_s,
                emit=self.context.emit,
            )
            if not ready:
                raise ReachyUnavailableError("daemon not ready")
            try:
                robot = await asyncio.to_thread(self._hardware.make_robot)
            except (ConnectionError, TimeoutError) as exc:
                raise ReachyUnavailableError(str(exc)) from exc
            self._robot = robot
            self._movement_manager = await asyncio.to_thread(self._hardware.make_movement_manager, robot)
            await self._sleep_robot()
            media = robot.media
            await asyncio.to_thread(media.start_recording)
            await asyncio.to_thread(media.start_playing)
            self._started = True
            self._mic_task = asyncio.create_task(self._mic_worker(), name=f"{self.id}-mic")
            self._play_task = asyncio.create_task(self._play_worker(), name=f"{self.id}-play")
            self._watch_task = asyncio.create_task(self._watch_worker(), name=f"{self.id}-daemon-watch")
        except BaseException:
            await self._release_resources()
            raise

    async def stop(self) -> None:
        """Stop tasks and release the robot; repeated calls are safe."""
        self._started = False
        self._session_id = None
        await self._release_resources()

    async def _release_resources(self) -> None:
        tasks = [task for task in (self._mic_task, self._play_task, self._watch_task) if task is not None]
        self._mic_task = self._play_task = self._watch_task = None
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._end_mic_stream()
        await self._clear_playback()
        manager = self._movement_manager
        if manager is not None and self._movement_started:
            try:
                await asyncio.to_thread(manager.stop, reset_to_neutral=False)
            except Exception:
                logger.debug("Failed to stop movement manager", exc_info=True)
        self._movement_started = False
        robot, self._robot = self._robot, None
        self._movement_manager = None
        if robot is not None:
            media = getattr(robot, "media", None)
            if media is not None:
                for method_name in ("stop_recording", "stop_playing"):
                    method = getattr(media, method_name, None)
                    if method is not None:
                        try:
                            await asyncio.to_thread(method)
                        except Exception:
                            logger.debug("Failed to stop Reachy media method %s", method_name, exc_info=True)
            try:
                close = getattr(robot, "__exit__", None)
                if close is not None:
                    await asyncio.to_thread(close, None, None, None)
                else:
                    disconnect = getattr(getattr(robot, "client", None), "disconnect", None)
                    if disconnect is not None:
                        await asyncio.to_thread(disconnect)
            except Exception:
                logger.debug("Failed to release Reachy Mini client", exc_info=True)

    def _end_mic_stream(self) -> None:
        try:
            self._mic_queue.put_nowait(None)
        except asyncio.QueueFull:
            try:
                self._mic_queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
            self._mic_queue.put_nowait(None)

    def _drain_play_queue(self) -> None:
        while not self._play_queue.empty():
            try:
                self._play_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

    def _record_audio_fault(self, fault: str, event: str, exc: Exception) -> None:
        self._audio_fault = fault
        logger.exception("Reachy audio worker failed (%s)", fault)
        try:
            self.context.emit(event, "live", "error", exc_type=type(exc).__name__)
        finally:
            self._drain_play_queue()
            self._end_mic_stream()

    async def _mic_worker(self) -> None:
        try:
            import numpy as np

            from ambient_reachy.audio import resample_linear, frame_to_mono_int16

            assert self._robot is not None
            media = self._robot.media
            source_rate = await asyncio.to_thread(media.get_input_audio_samplerate)
            while True:
                frame = await asyncio.to_thread(media.get_audio_sample)
                if self._audio_fault is not None or self._daemon_lost or not self._started:
                    return
                if frame is None:
                    await asyncio.sleep(0.01)
                    continue
                samples = frame_to_mono_int16(np.asarray(frame))
                if source_rate != SAMPLE_RATE:
                    if not self._resample_logged:
                        logger.info("Resampling Reachy microphone from %d Hz to %d Hz", source_rate, SAMPLE_RATE)
                        self._resample_logged = True
                    samples = resample_linear(samples, int(source_rate), SAMPLE_RATE)
                if not np.any(samples):
                    self._silence_seconds += samples.size / SAMPLE_RATE
                    if self._silence_seconds >= self.config.silence_alarm_s and not self._silence_event_sent:
                        logger.error(
                            "Microphone has delivered only zeros for %.0f s. Known Reachy Mini Lite XMOS startup fault "
                            "(pollen-robotics/reachy_mini#770). Run `ambient-reachy-mic-check --reboot-xmos`, then "
                            "restart ambient-home; if still silent, check the microphone FPC cable.",
                            self.config.silence_alarm_s,
                        )
                        self.context.emit("mic.silent_stream", "live", "error", seconds=self._silence_seconds)
                        self._mic_silent = True
                        self._silence_event_sent = True
                else:
                    self._silence_seconds = 0.0
                    self._mic_silent = False
                pcm = samples.astype("<i2", copy=False).tobytes()
                if self._mic_queue.full():
                    try:
                        self._mic_queue.get_nowait()
                    except asyncio.QueueEmpty:
                        pass
                self._mic_queue.put_nowait(pcm)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._record_audio_fault("mic_failed", "mic.stream_failed", exc)

    async def mic(self) -> AsyncIterator[bytes]:
        """Yield PCM16 chunks until stopped or the daemon connection is lost."""
        queue = self._mic_queue
        while True:
            chunk = await queue.get()
            if chunk is None:
                return
            yield chunk

    async def _play_worker(self) -> None:
        try:
            import numpy as np

            assert self._robot is not None
            media = self._robot.media
            while True:
                pcm = await self._play_queue.get()
                generation = self._play_generation
                samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
                if self._play_generation != generation:
                    continue
                await asyncio.to_thread(media.push_audio_sample, samples)
                if self._play_generation != generation:
                    await self._clear_media_player(media)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._record_audio_fault("playback_failed", "speaker.playback_failed", exc)

    async def play(self, pcm16: bytes) -> None:
        """Queue a complete PCM16 chunk without waiting for playback."""
        if self._audio_fault is not None:
            raise RuntimeError(f"Reachy audio is faulted: {self._audio_fault}")
        if not self._started or self._daemon_lost:
            raise RuntimeError("Reachy device is not available")
        if not self.speaker_format.is_whole_frames(pcm16):
            raise ValueError("PCM16 chunk must contain whole frames")
        self._play_queue.put_nowait(pcm16)

    async def stop_playback(self) -> None:
        """Discard queued audio and clear the current media-player buffer."""
        await self._clear_playback()

    async def _clear_playback(self) -> None:
        self._play_generation += 1
        self._drain_play_queue()
        if self._robot is None:
            return
        await self._clear_media_player(self._robot.media)

    async def _clear_media_player(self, media: Any) -> None:
        audio = getattr(media, "audio", media)
        clear = getattr(audio, "clear_player", None) or getattr(audio, "clear_output_buffer", None)
        if clear is not None:
            try:
                await asyncio.to_thread(clear)
            except Exception:
                logger.debug("Failed to clear Reachy audio player", exc_info=True)

    async def requests(self) -> AsyncIterator[str]:
        """Reachy does not provide a typed-text input channel."""
        if False:
            yield ""

    def tools(self) -> list[ToolSpec]:
        """Return camera and configured Pollen body tools while connected."""
        if not self._started or self._robot is None or self._movement_manager is None:
            return []
        result: list[ToolSpec] = []
        look = build_look_tool(self.context, self._robot, camera_enabled=self.config.camera)
        if look is not None:
            result.append(look)
        if self.config.body_tools:
            try:
                result.extend(
                    build_body_tools(
                        self.config.body_tools,
                        self._robot,
                        self._movement_manager,
                        camera_enabled=self.config.camera,
                        motion_duration_s=self.config.motion_duration_s,
                    )
                )
            except (ImportError, ModuleNotFoundError) as exc:
                logger.warning("Pollen body tools unavailable: %s", exc)
        return result

    async def on_session(self, state: SessionState) -> None:
        """Mirror hub session changes into the robot's body movements."""
        if not self._started or self._robot is None or self._movement_manager is None:
            return
        if state.phase is SessionPhase.WAKE:
            self._session_id = state.session_id
            try:
                await asyncio.to_thread(self._robot.enable_motors)
                await asyncio.to_thread(self._robot.wake_up)
                self._movement_started = True
                await asyncio.to_thread(self._movement_manager.start)
                await asyncio.to_thread(self._movement_manager.set_listening, False)
            except Exception as exc:
                logger.exception("Failed to wake robot")
                self.context.emit("robot.wake_failed", "live", "error", exc_type=type(exc).__name__)
            return
        if self._session_id != state.session_id:
            return
        if state.phase is SessionPhase.SLEEP:
            await self._sleep_robot()
            self._session_id = None
            return
        try:
            if state.phase is SessionPhase.INTERRUPTED:
                await asyncio.to_thread(self._movement_manager.set_speaking, False)
                await asyncio.to_thread(self._movement_manager.set_listening, True)
            elif state.phase in {SessionPhase.LISTENING, SessionPhase.SPEAKING, SessionPhase.IDLE}:
                await asyncio.to_thread(self._movement_manager.set_listening, state.listening)
                await asyncio.to_thread(self._movement_manager.set_speaking, state.speaking)
        except Exception:
            logger.exception("Failed to update Reachy session movement")

    async def _sleep_robot(self) -> None:
        robot = self._robot
        movement_manager = self._movement_manager
        if robot is None or movement_manager is None:
            return
        try:
            disable_wobbling = getattr(robot, "disable_wobbling", None)
            if disable_wobbling is not None:
                try:
                    await asyncio.to_thread(disable_wobbling)
                except Exception:
                    logger.debug("Failed to disable robot wobbling before sleep", exc_info=True)
            await asyncio.to_thread(movement_manager.stop, reset_to_neutral=False)
            self._movement_started = False
            await asyncio.to_thread(robot.goto_sleep)
        except Exception as exc:
            logger.exception("Failed to put robot to sleep")
            self.context.emit("robot.sleep_failed", "live", "error", exc_type=type(exc).__name__)

    async def _watch_worker(self) -> None:
        from ambient_reachy.daemon import watch_daemon

        await watch_daemon(
            self.config.daemon_status_url,
            self.config.daemon_watchdog_s,
            self.config.daemon_watchdog_failures,
            self._daemon_failed,
            emit=self.context.emit,
        )

    async def _daemon_failed(self, state: str) -> None:
        self._daemon_lost = True
        mic_task = self._mic_task
        if mic_task is not None:
            if mic_task is not asyncio.current_task():
                mic_task.cancel()
                await asyncio.gather(mic_task, return_exceptions=True)
            if self._mic_task is mic_task:
                self._mic_task = None
        self._end_mic_stream()
        await self.stop_playback()
        logger.error("Reachy daemon lost (state=%s); device will be restarted by the hub", state)

    async def health(self) -> DeviceHealth:
        """Report lifecycle, daemon, and microphone health without raising."""
        if not self._started:
            return DeviceHealth.down("not_started")
        if self._daemon_lost:
            return DeviceHealth.down("daemon_lost")
        if self._audio_fault is not None:
            return DeviceHealth.down(self._audio_fault)
        if self._mic_silent:
            return DeviceHealth.degraded("mic_silent")
        return DeviceHealth.ok()
