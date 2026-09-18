"""Bluetooth Low Energy heart rate collection.

Two collection modes are supported:

- ``gatt``: connect to the device and subscribe to notifications of the
  standard Heart Rate Measurement characteristic (Bluetooth SIG 0x2A37).
  This is the mode used by most chest straps and wrist bands.
- ``advertise``: passively scan advertisements and read heart rate values
  carried in the Service Data of the Heart Rate Service (0x180D), without
  establishing a GATT connection.

The ``bleak`` package is imported lazily so that the plugin can still be
loaded with a friendly error when the dependency is unavailable.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass

from astrbot.api import logger

HR_SERVICE_UUID = "0000180d-0000-1000-8000-00805f9b34fb"
HR_MEASUREMENT_UUID = "00002a37-0000-1000-8000-00805f9b34fb"
HR_SERVICE_UUID_SHORT = "180d"


def parse_heart_rate(data: bytes | bytearray) -> int | None:
    """Parse a Heart Rate Measurement value (Bluetooth SIG 0x2A37).

    The first byte is the flags field. Bit 0 selects the heart rate value
    format: 0 = UINT8, 1 = UINT16 (little-endian).

    Args:
        data: Raw characteristic value or heart-rate service-data payload.

    Returns:
        Heart rate in bpm, or None when the payload is too short.
    """
    if not data or len(data) < 2:
        return None
    flags = data[0]
    if flags & 0x01:
        return int.from_bytes(data[1:3], byteorder="little", signed=False)
    return int(data[1])


def _is_hr_service_uuid(uuid: str) -> bool:
    """Report whether an advertised UUID string is the Heart Rate Service."""
    lowered = uuid.lower()
    return lowered == HR_SERVICE_UUID or lowered == HR_SERVICE_UUID_SHORT


def _extract_hr_from_service_data(service_data: dict | None) -> int | None:
    """Extract a bpm value from BLE advertisement Service Data, if present."""
    if not service_data:
        return None
    for uuid, raw in service_data.items():
        if _is_hr_service_uuid(uuid):
            return parse_heart_rate(raw)
    return None


@dataclass
class ScannedDevice:
    """A device discovered during a BLE scan."""

    address: str
    name: str
    rssi: int
    has_hr_service: bool
    advertised_hr: int | None


async def scan_devices(timeout: int) -> list[ScannedDevice]:
    """Scan nearby BLE devices and report heart-rate related ones.

    Args:
        timeout: Scan duration in seconds.

    Returns:
        A list of discovered devices that either advertise the Heart Rate
        Service UUID or carry heart rate data in advertisement Service Data.
        If none are found, all discovered devices are returned so the user
        can still inspect device names and addresses.
    """
    from bleak import BleakScanner

    found: dict[str, ScannedDevice] = {}

    def _on_detection(device, advertisement_data) -> None:
        service_uuids = advertisement_data.service_uuids or []
        hr_service = any(_is_hr_service_uuid(u) for u in service_uuids)
        advertised_hr = _extract_hr_from_service_data(advertisement_data.service_data)
        previous = found.get(device.address)
        if previous:
            # Keep the richest information seen across advertisement packets.
            found[device.address] = ScannedDevice(
                address=device.address,
                name=device.name or previous.name,
                rssi=advertisement_data.rssi or previous.rssi,
                has_hr_service=previous.has_hr_service or hr_service,
                advertised_hr=previous.advertised_hr or advertised_hr,
            )
        else:
            found[device.address] = ScannedDevice(
                address=device.address,
                name=device.name or "",
                rssi=advertisement_data.rssi or 0,
                has_hr_service=hr_service,
                advertised_hr=advertised_hr,
            )

    scanner = BleakScanner(detection_callback=_on_detection)
    await scanner.start()
    try:
        await asyncio.sleep(timeout)
    finally:
        await scanner.stop()

    devices = list(found.values())
    hr_devices = [d for d in devices if d.has_hr_service or d.advertised_hr]
    return hr_devices or devices


class HeartRateMonitor:
    """Background BLE heart rate collector with automatic reconnect."""

    def __init__(self) -> None:
        self.running = False
        self.connected = False
        self.scanning = False
        self.mode = "gatt"
        self.latest_hr: int | None = None
        self.last_hr_ts = 0.0
        self.device_address = ""
        self.device_name = ""
        self.last_error = ""

        self._settings: dict = {}
        self._stop_event: asyncio.Event | None = None
        self._worker: asyncio.Task | None = None

    async def start(self, settings: dict) -> bool:
        """Start the collector worker.

        Args:
            settings: BLE settings (mode, device_address, device_name,
                scan_timeout, reconnect_interval).

        Returns:
            False when the collector is already running.
        """
        if self.running:
            return False
        self._settings = settings
        self.mode = settings.get("mode", "gatt")
        self.latest_hr = None
        self.last_hr_ts = 0.0
        self.device_address = settings.get("device_address", "") or ""
        self.device_name = ""
        self.connected = False
        self.scanning = False
        self.last_error = ""
        self._stop_event = asyncio.Event()
        self.running = True
        self._worker = asyncio.create_task(self._run())
        return True

    async def stop(self) -> None:
        """Stop the collector and wait for the worker to release the radio."""
        if not self.running:
            return
        self.running = False
        if self._stop_event:
            self._stop_event.set()
        worker = self._worker
        if worker:
            try:
                await asyncio.wait_for(asyncio.shield(worker), timeout=10)
            except (TimeoutError, asyncio.CancelledError):
                worker.cancel()
        self._worker = None
        self.connected = False
        self.scanning = False

    async def _run(self) -> None:
        try:
            if self.mode == "advertise":
                await self._run_advertise()
            else:
                await self._run_gatt()
        except asyncio.CancelledError:
            raise
        except Exception as e:
            self.last_error = str(e)
            logger.exception(f"Heart rate monitor stopped with an error: {e!s}")

    async def _interruptible_sleep(self, seconds: float) -> None:
        """Sleep that returns immediately when the stop event is set."""
        if not self._stop_event:
            return
        try:
            await asyncio.wait_for(self._stop_event.wait(), timeout=seconds)
        except TimeoutError:
            pass

    def _on_hr_notify(self, _sender, data: bytearray) -> None:
        """Handle a Heart Rate Measurement GATT notification."""
        hr = parse_heart_rate(data)
        if hr is not None and 0 < hr <= 300:
            self.latest_hr = hr
            self.last_hr_ts = time.time()

    async def _discover_device(self) -> str | None:
        """Discover a target heart rate device.

        Selection priority: configured address > name keyword match > first
        device advertising the Heart Rate Service.

        Returns:
            The device address, or None when no target is found.
        """
        from bleak import BleakScanner

        want_address = (self._settings.get("device_address") or "").strip()
        want_name = (self._settings.get("device_name") or "").strip().lower()
        timeout = int(self._settings.get("scan_timeout", 12))

        candidates: dict[str, ScannedDevice] = {}

        def _on_detection(device, advertisement_data) -> None:
            service_uuids = advertisement_data.service_uuids or []
            hr_service = any(_is_hr_service_uuid(u) for u in service_uuids)
            candidates[device.address] = ScannedDevice(
                address=device.address,
                name=device.name or "",
                rssi=advertisement_data.rssi or 0,
                has_hr_service=hr_service,
                advertised_hr=None,
            )

        scanner = BleakScanner(detection_callback=_on_detection)
        await scanner.start()
        try:
            deadline = time.time() + timeout
            while time.time() < deadline:
                if want_address and want_address in candidates:
                    return want_address
                if want_name:
                    for dev in candidates.values():
                        if want_name in dev.name.lower():
                            return dev.address
                else:
                    for dev in candidates.values():
                        if dev.has_hr_service:
                            return dev.address
                await self._interruptible_sleep(0.5)
                if self._stop_event and self._stop_event.is_set():
                    return None
        finally:
            await scanner.stop()
        return None

    async def _run_gatt(self) -> None:
        """Connect to the device and subscribe to HR notifications."""
        from bleak import BleakClient

        reconnect_interval = int(self._settings.get("reconnect_interval", 5))
        while self.running:
            address = self.device_address or await self._discover_device()
            if not address:
                self.last_error = "No heart rate device found during scan"
                logger.warning("Heart rate device not found, retrying...")
                await self._interruptible_sleep(reconnect_interval)
                continue

            self.device_address = address
            self.last_error = ""
            client = None
            try:
                async with BleakClient(address) as client:
                    self.device_name = address
                    await client.start_notify(HR_MEASUREMENT_UUID, self._on_hr_notify)
                    self.connected = True
                    logger.info(f"Subscribed to heart rate notifications: {address}")
                    while self.running and client.is_connected:
                        await self._interruptible_sleep(1)
            except Exception as e:
                self.last_error = str(e)
                logger.warning(f"Heart rate GATT connection error: {e!s}")
            finally:
                self.connected = False
                client = None

            if self.running:
                logger.info(
                    f"Heart rate device disconnected, reconnecting in "
                    f"{reconnect_interval}s..."
                )
                await self._interruptible_sleep(reconnect_interval)

    async def _run_advertise(self) -> None:
        """Passively receive heart rate values from BLE advertisements."""
        from bleak import BleakScanner

        reconnect_interval = int(self._settings.get("reconnect_interval", 5))

        def _on_detection(device, advertisement_data) -> None:
            hr = _extract_hr_from_service_data(advertisement_data.service_data)
            if hr is None or not (0 < hr <= 300):
                return
            want_address = (self._settings.get("device_address") or "").strip()
            want_name = (self._settings.get("device_name") or "").strip().lower()
            if want_address and device.address != want_address:
                return
            if want_name and want_name not in (device.name or "").lower():
                return
            self.latest_hr = hr
            self.last_hr_ts = time.time()
            self.device_address = device.address
            self.device_name = device.name or device.address

        while self.running:
            try:
                scanner = BleakScanner(detection_callback=_on_detection)
                await scanner.start()
                self.scanning = True
                self.last_error = ""
                logger.info("Listening for heart rate BLE advertisements...")
                try:
                    if self._stop_event:
                        await self._stop_event.wait()
                finally:
                    self.scanning = False
                    await scanner.stop()
            except Exception as e:
                self.scanning = False
                self.last_error = str(e)
                logger.warning(f"Heart rate advertisement scan error: {e!s}")
                await self._interruptible_sleep(reconnect_interval)

    def status(self) -> dict:
        """Return a snapshot of the collector state."""
        return {
            "running": self.running,
            "mode": self.mode,
            "connected": self.connected,
            "scanning": self.scanning,
            "latest_hr": self.latest_hr,
            "last_hr_ts": self.last_hr_ts,
            "device_address": self.device_address,
            "device_name": self.device_name,
            "last_error": self.last_error,
        }
